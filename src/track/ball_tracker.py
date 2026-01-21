from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import math


@dataclass
class BallState:
    """
    Tracked ball state used by higher-level reasoning modules.

    cx, cy are image coordinates (pixels).
    vx, vy are simple finite-difference estimates (pixels/frame).
    """
    frame_idx: int
    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0
    conf: float = 0.0


class BallTracker:
    """
    Lightweight tracker that turns per-frame ball detections into a stable trajectory.

    Why this exists:
    - The ball is small and often missed; raw detections are noisy.
    - Event logic (attempt/outcome) is temporal; it needs consistent ball points.

    Core assumptions:
    - There is at most one relevant ball in view at a time.
    - A true ball detection is usually close to the last known position.
      (Used to reject false positives when YOLO conf is low.)

    Interaction with the pipeline:
    - Input: detections from YOLO (after filtering / fake-ball suppression)
    - Output: BallState (or None if fully lost)
    - Optional rim_center makes the tracker more tolerant to short occlusions near the rim.
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
    ):
        """
        Parameters are expressed in pixels and frames.

        max_lost:
          Number of consecutive missing frames tolerated before resetting the track
          (when far from the rim).

        max_jump_px:
          Hard gating against “teleportation” to a false positive (hands/heads/etc.).

        Near-rim variants:
          The ball is frequently occluded near the rim/backboard; we allow a larger
          missing window and larger jump tolerance in that region.
        """
        self.ball_class_name = ball_class_name
        self.max_lost = max_lost
        self.ema_alpha = ema_alpha
        self.max_jump_px = max_jump_px

        self.max_lost_near_rim = max_lost_near_rim
        self.rim_near_px = rim_near_px
        self.max_jump_near_rim_px = max_jump_near_rim_px

        self._lost = 0
        self._history: List[BallState] = []
        self._last_smoothed: Optional[Tuple[float, float]] = None

        # Used by higher-level logic that may need “last known ball time”.
        self._last_seen_frame: int = -10**9

    @staticmethod
    def _center(det: Dict[str, Any]) -> Tuple[float, float]:
        cx = (det["x1"] + det["x2"]) / 2.0
        cy = (det["y1"] + det["y2"]) / 2.0
        return cx, cy

    @staticmethod
    def _bbox_area(det: Dict[str, Any]) -> float:
        return max(0.0, (det["x2"] - det["x1"])) * max(0.0, (det["y2"] - det["y1"]))

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
        """Frame index when the ball was last detected (not merely carried forward)."""
        return self._last_seen_frame

    def _filter_ball_dets(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select only detections of the configured ball class."""
        return [d for d in detections if str(d.get("name", "")).lower() == self.ball_class_name.lower()]

    def _dynamic_max_lost(self, rim_center: Optional[Tuple[float, float]]) -> int:
        """Increase tolerance to missing frames when the current track is near the rim."""
        if rim_center is None or self._last_smoothed is None:
            return self.max_lost
        if self._dist(self._last_smoothed, rim_center) <= self.rim_near_px:
            return self.max_lost_near_rim
        return self.max_lost

    def _dynamic_max_jump(self, rim_center: Optional[Tuple[float, float]]) -> float:
        """Allow larger motion jumps near the rim where detections jitter/occlude."""
        if rim_center is None or self._last_smoothed is None:
            return self.max_jump_px
        if self._dist(self._last_smoothed, rim_center) <= self.rim_near_px:
            return self.max_jump_near_rim_px
        return self.max_jump_px

    def _select_best(
        self,
        ball_dets: List[Dict[str, Any]],
        rim_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Pick the best ball candidate for this frame.

        Design choice (anti-teleportation):
        - If we have a previous position, we choose the closest candidate.
        - If even the closest candidate is too far (max_jump), we return None
          and treat it as occlusion (do not jump to a likely false positive).
        """
        if not ball_dets:
            return None

        if self._last_smoothed is None:
            # Bootstrap: no history yet -> highest confidence (tie-break by area)
            return sorted(
                ball_dets,
                key=lambda d: (float(d.get("conf", 0.0)), -self._bbox_area(d)),
                reverse=True
            )[0]

        last_pos = self._last_smoothed
        max_jump_now = self._dynamic_max_jump(rim_center)

        scored: List[Tuple[float, float, float, Dict[str, Any]]] = []
        for d in ball_dets:
            cx, cy = self._center(d)
            dist = self._dist((cx, cy), last_pos)
            conf = float(d.get("conf", 0.0))
            area = self._bbox_area(d)
            scored.append((dist, -conf, area, d))

        scored.sort(key=lambda t: (t[0], t[1], t[2]))  # dist asc, conf desc, area asc
        best_dist, _, _, best_det = scored[0]

        if best_dist > max_jump_now:
            return None

        return best_det

    def _ema(self, prev: Tuple[float, float], new: Tuple[float, float]) -> Tuple[float, float]:
        """Exponential moving average for position smoothing."""
        a = self.ema_alpha
        return (a * new[0] + (1 - a) * prev[0], a * new[1] + (1 - a) * prev[1])

    def update(
        self,
        frame_idx: int,
        detections: List[Dict[str, Any]],
        rim_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[BallState]:
        """
        Update the tracker with current frame detections.

        Returns:
        - BallState when a track exists (including during short occlusions, where the
          last known state is carried forward).
        - None when the track is fully lost (after max_lost / max_lost_near_rim).
        """
        ball_dets = self._filter_ball_dets(detections)
        best = self._select_best(ball_dets, rim_center=rim_center)

        if best is None:
            self._lost += 1
            max_lost_now = self._dynamic_max_lost(rim_center)
            if self._lost > max_lost_now:
                self.reset()
                return None
            return self.last()

        self._lost = 0
        cx, cy = self._center(best)
        self._last_seen_frame = frame_idx

        if self._last_smoothed is None:
            sm_cx, sm_cy = cx, cy
        else:
            sm_cx, sm_cy = self._ema(self._last_smoothed, (cx, cy))
        self._last_smoothed = (sm_cx, sm_cy)

        vx, vy = 0.0, 0.0
        prev = self.last()
        if prev is not None:
            dt = max(1, frame_idx - prev.frame_idx)
            vx = (sm_cx - prev.cx) / dt
            vy = (sm_cy - prev.cy) / dt

        state = BallState(
            frame_idx=frame_idx,
            cx=sm_cx,
            cy=sm_cy,
            vx=vx,
            vy=vy,
            conf=float(best.get("conf", 0.0)),
        )
        self._history.append(state)
        return state
