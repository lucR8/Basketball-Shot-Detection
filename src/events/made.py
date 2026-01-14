from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math

from src.track.ball_tracker import BallState
from src.events.attempt import AttemptEvent


@dataclass
class MadeEvent:
    frame_idx: int
    outcome: str  # "made" | "miss" | "unknown"
    attempt_frame_idx: int
    rim_cx: float
    rim_cy: float
    details: str = ""


class MadeDetector:
    """
    Decide MADE / MISS after an AttemptEvent.

    Logic (simple & effective):
      - After attempt triggers, monitor the next N frames.
      - MADE if we observe a "crossing" near the rim:
          ball goes from ABOVE the rim (cy < rim_cy - y_margin)
          to BELOW the rim (cy > rim_cy + y_margin)
        while x stays close to rim center (|cx - rim_cx| < x_tol)
      - If time window expires:
          - MISS if ball was seen but never crossed
          - UNKNOWN if ball got lost too much (occlusion)
    """

    def __init__(
        self,
        window_frames: int = 45,
        x_tol_px: float = 55.0,
        y_margin_px: float = 10.0,
        min_after_attempt: int = 2,
        max_lost_after_attempt: int = 10,
        use_dynamic_rim: bool = True,
    ):
        """
        window_frames: how many frames to look after attempt
        x_tol_px: how close ball must be to rim center in x to count as "in cylinder"
        y_margin_px: margin around rim center for 'above' and 'below' checks
        min_after_attempt: ignore the first few frames after attempt to avoid immediate noise
        max_lost_after_attempt: if ball is missing too long -> unknown
        use_dynamic_rim: if True, update rim position from detections every frame (if available)
        """
        self.window_frames = window_frames
        self.x_tol_px = x_tol_px
        self.y_margin_px = y_margin_px
        self.min_after_attempt = min_after_attempt
        self.max_lost_after_attempt = max_lost_after_attempt
        self.use_dynamic_rim = use_dynamic_rim

        self._active_attempt: Optional[AttemptEvent] = None
        self._start_frame: int = -1

        # rim state (can be updated dynamically)
        self._rim_cx: float = 0.0
        self._rim_cy: float = 0.0

        # tracking state after attempt
        self._lost: int = 0
        self._ball_pts: List[Tuple[int, float, float]] = []  # (frame_idx, cx, cy)

        # flags for crossing logic
        self._seen_above: bool = False
        self._seen_below: bool = False

    @staticmethod
    def _center(det: Dict[str, Any]) -> Tuple[float, float]:
        cx = (det["x1"] + det["x2"]) / 2.0
        cy = (det["y1"] + det["y2"]) / 2.0
        return cx, cy

    def _pick_rim(self, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        rims = [d for d in detections if str(d.get("name", "")).lower() == "rim"]
        if not rims:
            return None
        return max(rims, key=lambda d: float(d.get("conf", 0.0)))

    def start(self, attempt: AttemptEvent) -> None:
        """Start monitoring for made/miss right after an attempt."""
        self._active_attempt = attempt
        self._start_frame = attempt.frame_idx

        self._rim_cx = attempt.rim_cx
        self._rim_cy = attempt.rim_cy

        self._lost = 0
        self._ball_pts.clear()
        self._seen_above = False
        self._seen_below = False

    def _finish(self, frame_idx: int, outcome: str, details: str = "") -> MadeEvent:
        evt = MadeEvent(
            frame_idx=frame_idx,
            outcome=outcome,
            attempt_frame_idx=self._active_attempt.frame_idx if self._active_attempt else -1,
            rim_cx=self._rim_cx,
            rim_cy=self._rim_cy,
            details=details,
        )
        # reset
        self._active_attempt = None
        self._start_frame = -1
        self._lost = 0
        self._ball_pts.clear()
        self._seen_above = False
        self._seen_below = False
        return evt

    def active(self) -> bool:
        return self._active_attempt is not None

    def update(
        self,
        frame_idx: int,
        detections: List[Dict[str, Any]],
        ball_state: Optional[BallState],
        new_attempt: Optional[AttemptEvent] = None,
    ) -> Optional[MadeEvent]:
        """
        Call once per frame.

        - If new_attempt is not None, starts a monitoring window.
        - If monitoring is active, may return MadeEvent when decided.
        - Otherwise returns None.
        """
        # If a new attempt occurs, start/override (MVP behavior)
        if new_attempt is not None:
            self.start(new_attempt)

        if self._active_attempt is None:
            return None

        # Optional: refine rim position from detections while active
        if self.use_dynamic_rim:
            rim_det = self._pick_rim(detections)
            if rim_det is not None:
                self._rim_cx, self._rim_cy = self._center(rim_det)

        # Window bounds
        dt = frame_idx - self._start_frame
        if dt < self.min_after_attempt:
            return None

        if dt > self.window_frames:
            # Decide miss/unknown at end of window
            if self._lost > self.max_lost_after_attempt:
                return self._finish(frame_idx, "unknown", "ball lost too often after attempt")
            return self._finish(frame_idx, "miss", "no crossing detected in window")

        # Track ball after attempt
        if ball_state is None:
            self._lost += 1
            # if ball is lost a lot, we may end as unknown later
            return None

        # ball visible
        self._lost = max(0, self._lost - 1)  # gentle recovery
        cx, cy = ball_state.cx, ball_state.cy
        self._ball_pts.append((frame_idx, cx, cy))
        if len(self._ball_pts) > 120:
            self._ball_pts = self._ball_pts[-120:]

        # Check "in cylinder" (x near rim center)
        in_x = abs(cx - self._rim_cx) <= self.x_tol_px

        # Above / below checks (remember: y increases downward)
        above = cy < (self._rim_cy - self.y_margin_px)
        below = cy > (self._rim_cy + self.y_margin_px)

        # We only accept above/below if x is near rim center to avoid side false positives
        if in_x and above:
            self._seen_above = True
        if in_x and below:
            self._seen_below = True

        # MADE condition: seen above then below (order matters by construction)
        if self._seen_above and self._seen_below:
            return self._finish(frame_idx, "made", "above->below crossing near rim center")

        return None
