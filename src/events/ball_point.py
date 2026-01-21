from __future__ import annotations

from typing import Optional, Dict, Any, Tuple


def _center(det: Dict[str, Any]) -> Tuple[float, float]:
    return (
        (float(det["x1"]) + float(det["x2"])) / 2.0,
        (float(det["y1"]) + float(det["y2"])) / 2.0,
    )


def _area(det: Dict[str, Any]) -> float:
    w = max(0.0, float(det["x2"]) - float(det["x1"]))
    h = max(0.0, float(det["y2"]) - float(det["y1"]))
    return w * h


class BallPointResolver:
    """
    Resolve a single ball point (bx, by, src) each frame.

    Why this exists:
    - Downstream logic (FSM gates) needs *one* ball point per frame.
    - The ball may be detected by YOLO, predicted by the tracker, or missing.
    - We want a stable priority order and a short fallback memory for brief dropouts.

    Resolution order:
    1) Tracker point (ball_state), if available.
       - This provides continuity and smoothing across missed detections.
    2) YOLO ball detection overrides tracker *if valid* (size filter).
       - When YOLO is good, we use it to correct tracker drift.
    3) Short memory of the last valid point.
       - Handles 1–N frame holes without inventing new motion.

    Important engineering note:
    - This component does *not* apply semantic gates like "close_to_person".
      Those checks remain in AttemptDetector gates, where they belong.
    """

    def __init__(
        self,
        memory_frames: int = 6,
        enable_size_filter: bool = True,
        area_min_px2: float = 180.0,
        area_max_px2: float = 12000.0,
    ):
        self.memory_frames = int(memory_frames)
        self.enable_size_filter = bool(enable_size_filter)
        self.area_min_px2 = float(area_min_px2)
        self.area_max_px2 = float(area_max_px2)

        self._last_ball_xy: Optional[Tuple[float, float]] = None
        self._last_ball_frame: int = -10**9
        self._last_src: str = "none"

    def update(self, frame_idx: int, ball_state, ball_det: Optional[Dict[str, Any]], person_bbox=None):
        """
        Returns:
        - (bx, by, src) when a ball point can be produced
        - None when no reliable source exists (no tracker, no valid detection, memory expired)

        The `person_bbox` parameter is intentionally not used here:
        person-relative constraints are evaluated in higher-level gates.
        """
        bx = by = None
        src = "none"

        # 1) Tracker candidate (preferred baseline for temporal continuity).
        if ball_state is not None:
            bx, by = float(ball_state.cx), float(ball_state.cy)
            src = "tracker"

        # 2) YOLO detection overrides tracker if it passes the size validity check.
        if ball_det is not None:
            ok = True
            if self.enable_size_filter:
                a = _area(ball_det)
                ok = (self.area_min_px2 <= a <= self.area_max_px2)
            if ok:
                bx, by = _center(ball_det)
                src = "yolo" if ball_state is not None else "yolo_only"

        # 3) Short memory fallback for brief dropouts.
        if (bx is None or by is None) and self._last_ball_xy is not None:
            if (frame_idx - self._last_ball_frame) <= self.memory_frames:
                bx, by = self._last_ball_xy
                src = "memory"

        if bx is None or by is None:
            return None

        self._last_ball_xy = (bx, by)
        self._last_ball_frame = frame_idx
        self._last_src = src
        return (bx, by, src)
