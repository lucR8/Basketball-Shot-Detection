from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import math


@dataclass
class BallState:
    """
    Tracked ball state consumed by higher-level reasoning modules.

    Coordinates:
      - cx, cy: image coordinates in pixels

    Motion:
      - vx, vy: finite-difference velocity estimates in pixels/frame

    Confidence:
      - conf: detector confidence when observed; 0.0 when the state is a short-horizon
        prediction during occlusion (i.e., not observed in the current frame).
    """
    frame_idx: int
    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0
    conf: float = 0.0


class BallTracker:
    """
    Lightweight single-object tracker for the basketball.

    Why this exists:
    - The ball is small, fast, and frequently missed by the detector.
    - Raw detections are noisy; downstream FSM logic needs a temporally consistent
      sequence of ball points.

    Core assumptions:
    - At most one relevant ball is present in the view.
    - A true detection is usually near the last estimated position.
      This enables gating against false positives when confidence is low.

    Inputs / outputs:
    - Input: YOLO detections (optionally pre-filtered by confidence and fake-ball suppression)
    - Output: BallState (or None if the track is fully lost)

    Occlusion handling:
    - When the ball is temporarily missing, we can optionally *predict forward*
      using a constant-velocity model. This prevents the track from "freezing"
      and improves reacquisition (distance gating is applied against a moving estimate).
    """

    def __init__(
        self,
        ball_class_name: str = "ball",
        max_lost: int = 10,
        ema_alpha: float = 0.6,
        max_jump_px: float = 120.0,
        max_lost_near_rim: int = 25,
        rim_near_px: float = 140.0,
        max_jump_near_rim_px: float = 200.0,
        *,
        pred_during_lost: bool = True,
        jump_per_lost_frame: float = 12.0,
    ):
        """
        Parameters (pixels, frames):

        max_lost:
          Max consecutive missing frames tolerated before resetting the track
          (when far from the rim).

        max_jump_px:
          Base gating threshold against "teleportation" to an unrelated false positive.

        Near-rim variants:
          Near the rim/backboard, occlusions are common; we allow a longer missing window
          and larger admissible jumps to improve reacquisition in that area.

        pred_during_lost:
          If True, produce predicted BallState during occlusions (conf=0.0), using a
          constant-velocity model. This keeps the estimate moving and reduces
          reacquisition failures caused by stale last positions.

        jump_per_lost_frame:
          Reacquisition slack: the longer the ball has been missing, the larger the
          admissible jump becomes. This compensates for the increasing uncertainty
          of the last known position.
        """
        self.ball_class_name = ball_class_name

        self.max_lost = int(max_lost)
        self.ema_alpha = float(ema_alpha)
        self.max_jump_px = float(max_jump_px)

        self.max_lost_near_rim = int(max_lost_near_rim)
        self.rim_near_px = float(rim_near_px)
        self.max_jump_near_rim_px = float(max_jump_near_rim_px)

        self.pred_during_lost = bool(pred_during_lost)
        self.jump_per_lost_frame = float(jump_per_lost_frame)

        self._lost = 0
        self._history: List[BallState] = []
        self._last_smoothed: Optional[Tuple[float, float]] = None

        # Frame index when the ball was last observed by the detector (not predicted).
        self._last_seen_frame: int = -10**9

    # -----------------------------
    # Geometry helpers
    # -----------------------------
    @staticmethod
    def _center(det: Dict[str, Any]) -> Tuple[float, float]:
        cx = (float(det["x1"]) + float(det["x2"])) / 2.0
        cy = (float(det["y1"]) + float(det["y2"])) / 2.0
        return cx, cy

    @staticmethod
    def _bbox_area(det: Dict[str, Any]) -> float:
        return max(0.0, float(det["x2"]) - float(det["x1"])) * max(0.0, float(det["y2"]) - float(det["y1"]))

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def reset(self) -> None:
        """Clear state when the track is considered lost."""
        self._lost = 0
        self._history.clear()
        self._last_smoothed = None
        self._last_seen_frame = -10**9

    def history(self) -> List[BallState]:
        return self._history

    def last(self) -> Optional[BallState]:
        return self._history[-1] if self._history else None

    @property
    def last_seen_frame(self) -> int:
        """Frame index when the ball was last detected (not merely predicted)."""
        return self._last_seen_frame

    # -----------------------------
    # Candidate filtering & gating
    # -----------------------------
    def _filter_ball_dets(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select only detections of the configured ball class."""
        want = self.ball_class_name.lower()
        return [d for d in detections if str(d.get("name", "")).lower() == want]

    def _dynamic_max_lost(self, rim_center: Optional[Tuple[float, float]]) -> int:
        """
        Increase tolerance to missing frames when the current track is near the rim.

        Rationale:
        - Near the rim/backboard, occlusions are common and detector jitter increases.
        """
        if rim_center is None or self._last_smoothed is None:
            return self.max_lost
        if self._dist(self._last_smoothed, rim_center) <= self.rim_near_px:
            return self.max_lost_near_rim
        return self.max_lost

    def _dynamic_max_jump(self, rim_center: Optional[Tuple[float, float]]) -> float:
        """
        Compute the admissible jump threshold for the current frame.

        Components:
        - Base jump gate (larger near the rim where jitter/occlusion is common)
        - Reacquisition slack that grows with the number of consecutive lost frames
          (the last known position becomes stale as time passes).
        """
        base = self.max_jump_px
        if rim_center is not None and self._last_smoothed is not None:
            if self._dist(self._last_smoothed, rim_center) <= self.rim_near_px:
                base = self.max_jump_near_rim_px

        return float(base + self._lost * self.jump_per_lost_frame)

    def _select_best(
        self,
        ball_dets: List[Dict[str, Any]],
        rim_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Pick the best ball candidate for this frame.

        Design choice (anti-teleportation):
        - If we have a previous estimate, choose the closest candidate.
        - If even the closest candidate is too far (max_jump), treat as occlusion (None).
        - Bootstrap: if no estimate yet, pick highest-confidence (tie-break by area).

        Note:
        - This is intentionally simple (single-object association) to keep reasoning
          transparent for academic evaluation.
        """
        if not ball_dets:
            return None

        if self._last_smoothed is None:
            # Bootstrap: no history -> highest confidence (tie-break by area)
            return sorted(
                ball_dets,
                key=lambda d: (float(d.get("conf", 0.0)), self._bbox_area(d)),
                reverse=True,
            )[0]

        last_pos = self._last_smoothed
        max_jump_now = self._dynamic_max_jump(rim_center)

        # Score by distance first; then prefer higher confidence; then prefer larger area.
        # (area is only a weak tie-breaker; it tends to penalize tiny spurious boxes)
        scored: List[Tuple[float, float, float, Dict[str, Any]]] = []
        for d in ball_dets:
            cx, cy = self._center(d)
            dist = self._dist((cx, cy), last_pos)
            conf = float(d.get("conf", 0.0))
            area = self._bbox_area(d)
            scored.append((dist, -conf, -area, d))  # area desc

        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        best_dist, _, _, best_det = scored[0]

        if best_dist > max_jump_now:
            return None

        return best_det

    def _ema(self, prev: Tuple[float, float], new: Tuple[float, float]) -> Tuple[float, float]:
        """Exponential moving average for position smoothing."""
        a = self.ema_alpha
        return (a * new[0] + (1 - a) * prev[0], a * new[1] + (1 - a) * prev[1])

    # -----------------------------
    # Update
    # -----------------------------
    def update(
        self,
        frame_idx: int,
        detections: List[Dict[str, Any]],
        rim_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[BallState]:
        """
        Update the tracker with current frame detections.

        Returns:
        - BallState when a track exists (including short occlusions if pred_during_lost=True)
        - None when the track is fully lost (after max_lost / max_lost_near_rim)
        """
        ball_dets = self._filter_ball_dets(detections)
        best = self._select_best(ball_dets, rim_center=rim_center)

        # -----------------------------
        # No accepted detection: occlusion / detector miss
        # -----------------------------
        if best is None:
            self._lost += 1
            max_lost_now = self._dynamic_max_lost(rim_center)
            if self._lost > max_lost_now:
                self.reset()
                return None

            last = self.last()
            if last is None or not self.pred_during_lost:
                return last

            # Predict forward using constant velocity (short-horizon approximation).
            # We keep the estimate moving to avoid reacquisition failures caused by
            # a stale last position.
            pred_x = last.cx + last.vx
            pred_y = last.cy + last.vy

            if self._last_smoothed is None:
                self._last_smoothed = (pred_x, pred_y)
            else:
                self._last_smoothed = self._ema(self._last_smoothed, (pred_x, pred_y))

            smx, smy = self._last_smoothed

            state = BallState(
                frame_idx=int(frame_idx),
                cx=float(smx),
                cy=float(smy),
                vx=float(last.vx),
                vy=float(last.vy),
                conf=0.0,  # not observed this frame
            )
            self._history.append(state)
            return state

        # -----------------------------
        # Accepted detection: update filter + velocity
        # -----------------------------
        self._lost = 0
        cx, cy = self._center(best)
        self._last_seen_frame = int(frame_idx)

        if self._last_smoothed is None:
            sm_cx, sm_cy = cx, cy
        else:
            sm_cx, sm_cy = self._ema(self._last_smoothed, (cx, cy))
        self._last_smoothed = (sm_cx, sm_cy)

        vx, vy = 0.0, 0.0
        prev = self.last()
        if prev is not None:
            dt = max(1, int(frame_idx) - int(prev.frame_idx))
            vx = (sm_cx - prev.cx) / dt
            vy = (sm_cy - prev.cy) / dt

        state = BallState(
            frame_idx=int(frame_idx),
            cx=float(sm_cx),
            cy=float(sm_cy),
            vx=float(vx),
            vy=float(vy),
            conf=float(best.get("conf", 0.0)),
        )
        self._history.append(state)
        return state
