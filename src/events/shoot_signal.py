from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple


def _pick_best(detections: List[Dict[str, Any]], cls_name: str) -> Optional[Dict[str, Any]]:
    """Select the highest-confidence detection for a given class."""
    cls = cls_name.lower()
    cand = [d for d in detections if str(d.get("name", "")).lower() == cls]
    if not cand:
        return None
    return max(cand, key=lambda d: float(d.get("conf", 0.0)))


def _bbox(det: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]))


class ShootSignalTracker:
    """
    Convert YOLO "shoot" detections into a temporally consistent signal.

    Why this exists:
    - "shoot" (ball release) is an instantaneous / short-lived visual cue.
    - YOLO may miss it for a few frames (especially at long distances).
    - The FSM benefits from:
        - shoot_now: a stable boolean ("shoot is currently active")
        - shoot_rise: the transition moment (False -> True), used for arming logic
        - shoot_streak: how many consecutive frames shoot_now has been true
        - shoot_bbox memory: keep a recent bbox during short dropouts

    Invariant:
    - If shoot_now is True, shoot_bbox is guaranteed to be not None.
      (either from the current frame or from memory)
    """

    def __init__(self, conf_min: float = 0.18, memory_frames: int = 8):
        self.conf_min = float(conf_min)
        self.memory_frames = int(memory_frames)

        self._last_shoot_det: Optional[Dict[str, Any]] = None
        self._last_shoot_bbox: Optional[Tuple[float, float, float, float]] = None
        self._last_shoot_frame: int = -10**9

        self._shoot_active = False
        self._shoot_streak = 0

    def update(self, frame_idx: int, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns a dict of shoot-related signals consumed by AttemptDetector.

        The returned values are intentionally simple:
        they are "perception-side" signals; gating and attempt decisions happen elsewhere.
        """
        raw = _pick_best(detections, "shoot")

        # Update memory only with confident detections.
        if raw is not None and float(raw.get("conf", 0.0)) >= self.conf_min:
            self._last_shoot_det = raw
            self._last_shoot_bbox = _bbox(raw)
            self._last_shoot_frame = frame_idx

        from_memory = False
        shoot_bbox = None
        shoot_det = None
        shoot_conf = 0.0

        # Prefer current strong detection; otherwise fall back to short memory.
        if raw is not None and float(raw.get("conf", 0.0)) >= self.conf_min:
            shoot_det = raw
            shoot_bbox = _bbox(raw)
            shoot_conf = float(raw.get("conf", 0.0))
        else:
            if self._last_shoot_bbox is not None and (frame_idx - self._last_shoot_frame) <= self.memory_frames:
                from_memory = True
                shoot_bbox = self._last_shoot_bbox
                shoot_det = self._last_shoot_det
                shoot_conf = float(shoot_det.get("conf", 0.0)) if shoot_det is not None else 0.0

        shoot_now = shoot_bbox is not None
        shoot_rise = shoot_now and (not self._shoot_active)

        self._shoot_active = shoot_now
        self._shoot_streak = (self._shoot_streak + 1) if shoot_now else 0

        return {
            "shoot_now": bool(shoot_now),
            "shoot_rise": bool(shoot_rise),
            "shoot_streak": int(self._shoot_streak),
            "shoot_conf": float(shoot_conf),
            "shoot_bbox": shoot_bbox,          # guaranteed when shoot_now=True
            "shoot_det": shoot_det,            # may be None if only memory bbox is available
            "shoot_from_memory": bool(from_memory),
        }
