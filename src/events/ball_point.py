from __future__ import annotations

from typing import Optional, Dict, Any, Tuple


def _center(det: Dict[str, Any]) -> Tuple[float, float]:
    return ((float(det["x1"]) + float(det["x2"])) / 2.0,
            (float(det["y1"]) + float(det["y2"])) / 2.0)


def _area(det: Dict[str, Any]) -> float:
    w = max(0.0, float(det["x2"]) - float(det["x1"]))
    h = max(0.0, float(det["y2"]) - float(det["y1"]))
    return w * h


class BallPointResolver:
    """
    Résout un point balle (bx,by,src) à partir de:
      - YOLO ball (si valide) prioritaire
      - sinon tracker (ball_state)
      - sinon mémoire courte
    Important: ne PAS filtrer ici "close_to_person" (ça doit rester un gate dans Attempt).
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
        bx = by = None
        src = "none"

        # 1) tracker
        if ball_state is not None:
            bx, by = float(ball_state.cx), float(ball_state.cy)
            src = "tracker"

        # 2) yolo overrides if valid
        if ball_det is not None:
            ok = True
            if self.enable_size_filter:
                a = _area(ball_det)
                ok = (self.area_min_px2 <= a <= self.area_max_px2)
            if ok:
                bx, by = _center(ball_det)
                src = "yolo" if ball_state is not None else "yolo_only"

        # 3) memory
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
