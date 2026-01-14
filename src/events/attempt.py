from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math

from src.track.ball_tracker import BallState


@dataclass
class AttemptEvent:
    frame_idx: int
    ball_cx: float
    ball_cy: float
    rim_cx: float
    rim_cy: float
    distance_px: float


class AttemptDetector:
    """
    Detect shot attempts using ball trajectory + rim position.

    Trigger (simple MVP):
      - rim detected
      - ball tracked
      - ball enters a radius around rim (enter_radius_px)
      - ball is descending (vy > vy_min)
      - cooldown to prevent double counts
    """

    def __init__(
        self,
        enter_radius_px: float = 85.0,
        vy_min: float = 0.5,
        cooldown_frames: int = 25,
        require_approach: bool = True,
        approach_window: int = 6,
    ):
        """
        enter_radius_px: distance threshold to consider ball "near rim"
        vy_min: minimum downward velocity to consider a shot (image coords: down is +y)
        cooldown_frames: frames to wait after an attempt before allowing another
        require_approach: if True, require distance to rim decreasing over a short window
        approach_window: how many recent frames to check for decreasing distance
        """
        self.enter_radius_px = enter_radius_px
        self.vy_min = vy_min
        self.cooldown_frames = cooldown_frames
        self.require_approach = require_approach
        self.approach_window = approach_window

        self._cooldown = 0
        self._dist_hist: List[Tuple[int, float]] = []  # (frame_idx, distance)

    @staticmethod
    def _center(det: Dict[str, Any]) -> Tuple[float, float]:
        cx = (det["x1"] + det["x2"]) / 2.0
        cy = (det["y1"] + det["y2"]) / 2.0
        return cx, cy

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _pick_rim(self, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # Keep best rim detection by confidence
        rims = [d for d in detections if str(d.get("name", "")).lower() == "rim"]
        if not rims:
            return None
        return max(rims, key=lambda d: float(d.get("conf", 0.0)))

    def _is_approaching(self) -> bool:
        """
        True if distance has been decreasing overall in the last approach_window points.
        """
        if len(self._dist_hist) < self.approach_window:
            return True  # not enough history -> don't block

        last = self._dist_hist[-self.approach_window:]
        d0 = last[0][1]
        d1 = last[-1][1]
        # allow small noise
        return (d1 < d0 - 2.0)

    def update(
        self,
        frame_idx: int,
        detections: List[Dict[str, Any]],
        ball_state: Optional[BallState],
    ) -> Optional[AttemptEvent]:
        """
        Call once per frame.

        Returns AttemptEvent when a new attempt is detected, else None.
        """
        # Handle cooldown
        if self._cooldown > 0:
            self._cooldown -= 1

        if ball_state is None:
            return None

        rim_det = self._pick_rim(detections)
        if rim_det is None:
            return None

        rim_cx, rim_cy = self._center(rim_det)
        ball_pos = (ball_state.cx, ball_state.cy)
        rim_pos = (rim_cx, rim_cy)

        dist = self._dist(ball_pos, rim_pos)
        self._dist_hist.append((frame_idx, dist))

        # Keep history small
        if len(self._dist_hist) > 50:
            self._dist_hist = self._dist_hist[-50:]

        # Conditions
        near_rim = dist <= self.enter_radius_px
        descending = ball_state.vy >= self.vy_min  # +y is down
        approaching = self._is_approaching() if self.require_approach else True

        # Ready if cooldown is over, the cooldown is useful to avoid multiple detection for same attempt
        # For example when the ball is still near the rim for multiple frames = shot bouncing on the rim
        ready = self._cooldown == 0

        if near_rim and descending and approaching and ready:
            self._cooldown = self.cooldown_frames
            return AttemptEvent(
                frame_idx=frame_idx,
                ball_cx=ball_state.cx,
                ball_cy=ball_state.cy,
                rim_cx=rim_cx,
                rim_cy=rim_cy,
                distance_px=dist,
            )

        return None
