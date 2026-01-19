from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple


def _pick_best(detections: List[Dict[str, Any]], cls_name: str) -> Optional[Dict[str, Any]]:
    cls = cls_name.lower()
    cand = [d for d in detections if str(d.get("name", "")).lower() == cls]
    if not cand:
        return None
    return max(cand, key=lambda d: float(d.get("conf", 0.0)))


def _bbox(det: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]))


class ShootSignalTracker:
    """
    Détecte shoot_now/shoot_rise + mémoire courte de la bbox shoot.
    Invariant important:
      - si shoot_now=True => shoot_bbox != None
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
        raw = _pick_best(detections, "shoot")

        if raw is not None and float(raw.get("conf", 0.0)) >= self.conf_min:
            self._last_shoot_det = raw
            self._last_shoot_bbox = _bbox(raw)
            self._last_shoot_frame = frame_idx

        from_memory = False
        shoot_bbox = None
        shoot_det = None
        shoot_conf = 0.0

        # Use current strong detection first, else memory
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
            "shoot_bbox": shoot_bbox,          # <= IMPORTANT
            "shoot_det": shoot_det,            # may be None if memory bbox only (rare)
            "shoot_from_memory": bool(from_memory),
        }
