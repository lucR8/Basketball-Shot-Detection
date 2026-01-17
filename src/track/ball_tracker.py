from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import math


@dataclass
class BallState:
    frame_idx: int
    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0
    conf: float = 0.0


class BallTracker:
    """
    Lightweight ball tracker for YOLO detections.

    Input each frame:
      - list of detections dicts (from yolo.py)
    Output:
      - current ball state (position + velocity) or None

    Strategy:
      1) Filter detections to keep only class "ball" (by name or class id).
      2) Pick best detection using:
           - proximity to last position (if available)
           - confidence
           - reasonable bbox size
      3) Smooth position with exponential moving average (EMA).
      4) Estimate velocity from smoothed positions.
      5) Handle temporary misses (occlusion) with max_lost frames.

    Improvement (rim-aware tolerance):
      - When the ball is close to the rim, detections often drop (occlusion/backboard).
      - We allow a larger "lost" window near the rim, but keep the normal one elsewhere.

    Key fix (anti-teleportation):
      - When global confidence is lowered, false positives can appear (hands/heads).
      - We MUST prioritize proximity to last position, and if all candidates are too far,
        treat as "no detection" (occlusion) instead of jumping to the wrong object.
    """

    def __init__(
        self,
        ball_class_name: str = "ball",
        max_lost: int = 10,
        ema_alpha: float = 0.6,
        max_jump_px: float = 120.0,
        # New: rim-aware lost tolerance
        max_lost_near_rim: int = 25,
        rim_near_px: float = 140.0,
        # New: allow bigger jumps near rim (ball moves fast + occlusions)
        max_jump_near_rim_px: float = 200.0,
    ):
        """
        ball_class_name: class name in detection dicts ("ball")
        max_lost: allowed consecutive frames without ball detection before state reset (far from rim)
        ema_alpha: smoothing factor (0..1). higher = less smoothing, more reactive.
        max_jump_px: reject detections too far from last known position (helps reduce false positives)

        max_lost_near_rim: allowed consecutive frames without ball detection when ball is near rim
        rim_near_px: distance threshold (in pixels) to consider ball "near rim"

        max_jump_near_rim_px: when near rim, allow larger jump gate (fast motion + jitter)
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

        # New: keep track of when we last saw the ball (useful for attempt fallback logic)
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
        """Frame index when the ball was last detected (not just predicted)."""
        return self._last_seen_frame

    def _filter_ball_dets(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Prefer name-based filtering
        balls = [d for d in detections if str(d.get("name", "")).lower() == self.ball_class_name.lower()]
        return balls

    def _dynamic_max_lost(self, rim_center: Optional[Tuple[float, float]]) -> int:
        """
        Pick max_lost depending on whether the current tracked ball is near rim.
        If we don't have rim_center or we don't have a last position -> use default max_lost.
        """
        if rim_center is None or self._last_smoothed is None:
            return self.max_lost

        if self._dist(self._last_smoothed, rim_center) <= self.rim_near_px:
            return self.max_lost_near_rim

        return self.max_lost

    def _dynamic_max_jump(self, rim_center: Optional[Tuple[float, float]]) -> float:
        """
        Pick max_jump depending on whether the current tracked ball is near rim.
        Near rim, allow bigger jumps (fast motion + jitter + occlusion).
        """
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
        Select best ball detection.

        IMPORTANT:
          - If we have a previous position, we prioritize proximity.
          - If ALL candidates are too far (beyond max_jump), we return None
            (treat as occlusion) instead of "teleporting" to a false positive.
        """
        if not ball_dets:
            return None

        # If no previous position, take highest confidence (tie -> smaller area tends to be ball-like)
        if self._last_smoothed is None:
            return sorted(ball_dets, key=lambda d: (float(d.get("conf", 0.0)), -self._bbox_area(d)), reverse=True)[0]

        last_pos = self._last_smoothed
        max_jump_now = self._dynamic_max_jump(rim_center)

        # Prefer the closest detection (with a mild confidence tie-break)
        scored: List[Tuple[float, float, float, Dict[str, Any]]] = []
        for d in ball_dets:
            cx, cy = self._center(d)
            dist = self._dist((cx, cy), last_pos)
            conf = float(d.get("conf", 0.0))
            area = self._bbox_area(d)
            scored.append((dist, -conf, area, d))

        scored.sort(key=lambda t: (t[0], t[1], t[2]))  # dist asc, conf desc, area asc
        best_dist, _, _, best_det = scored[0]

        # Hard gate: if the closest candidate is still too far, do NOT jump
        if best_dist > max_jump_now:
            return None

        return best_det

    def _ema(self, prev: Tuple[float, float], new: Tuple[float, float]) -> Tuple[float, float]:
        a = self.ema_alpha
        return (a * new[0] + (1 - a) * prev[0], a * new[1] + (1 - a) * prev[1])

    def update(
        self,
        frame_idx: int,
        detections: List[Dict[str, Any]],
        rim_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[BallState]:
        """
        Update tracker with detections of current frame.

        rim_center:
          - Optional (cx, cy) of rim. If provided, we become more permissive
            to temporary ball loss near the rim (occlusions).

        Returns:
          BallState if tracked (even if predicted during short occlusion, we keep last state),
          None if completely lost (after max_lost).
        """
        ball_dets = self._filter_ball_dets(detections)
        best = self._select_best(ball_dets, rim_center=rim_center)

        if best is None:
            # no detection (or rejected by gate) -> occlusion
            self._lost += 1

            # New: dynamic tolerance (near rim -> allow more missing frames)
            max_lost_now = self._dynamic_max_lost(rim_center)

            if self._lost > max_lost_now:
                self.reset()
                return None

            # keep last known state (do not add new points)
            return self.last()

        # We have a detection
        self._lost = 0
        cx, cy = self._center(best)
        self._last_seen_frame = frame_idx

        # Smooth
        if self._last_smoothed is None:
            sm_cx, sm_cy = cx, cy
        else:
            sm_cx, sm_cy = self._ema(self._last_smoothed, (cx, cy))
        self._last_smoothed = (sm_cx, sm_cy)

        # Velocity (difference of smoothed positions)
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
