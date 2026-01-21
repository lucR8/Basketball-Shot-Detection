from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import math


@dataclass
class RimStable:
    """
    Stabilized rim reference.

    This is intentionally separate from raw detections:
    downstream modules consume RimStable, while YOLO boxes remain unchanged.
    """
    cx: float
    cy: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    conf: float = 0.0
    last_seen_frame: int = -10**9


class RimStabilizer:
    """
    Temporal smoothing of rim detections for mostly static-camera videos.

    Why this exists:
    - Rim detections jitter frame-to-frame (small box, partial occlusions).
    - Outcome reasoning needs a stable geometric reference (rim center + rim plane).

    Design choices:
    - This module never edits `dets` (separation of perception vs. reasoning).
    - It outputs a stable reference once enough consistent detections are seen (warmup).
    - It can keep (“hold”) the last stable reference briefly when detections disappear.
    """

    def __init__(
        self,
        alpha: float = 0.12,
        conf_min: float = 0.35,
        hold_frames: int = 60,
        warmup_min_hits: int = 8,
        max_step_px: float = 0.0,
    ):
        """
        alpha:
          EMA smoothing factor (higher = more reactive).

        warmup_min_hits:
          Number of confident rim observations required before declaring the rim “stable”.
          This avoids locking on a single spurious detection.

        hold_frames:
          If the rim disappears temporarily, keep the last stable reference for this many frames.

        max_step_px:
          Optional clamp on per-frame motion of the stabilized rim.
          Default is 0 (disabled) because the rim should not move significantly; EMA is enough.
        """
        self.alpha = float(alpha)
        self.conf_min = float(conf_min)
        self.hold_frames = int(hold_frames)
        self.warmup_min_hits = int(warmup_min_hits)
        self.max_step_px = float(max_step_px)

        self._stable: Optional[RimStable] = None
        self._hits: int = 0

    @staticmethod
    def _pick_best_rim(dets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        rims = [d for d in dets if str(d.get("name", "")).lower() == "rim"]
        if not rims:
            return None
        return max(rims, key=lambda d: float(d.get("conf", 0.0)))

    @staticmethod
    def _clip_step(prev: float, new: float, max_step: float) -> float:
        """Optional per-coordinate clamp (disabled when max_step <= 0)."""
        if max_step <= 0:
            return new
        d = new - prev
        if d > max_step:
            return prev + max_step
        if d < -max_step:
            return prev - max_step
        return new

    @staticmethod
    def _finite(x: float) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(float(x))

    def reset(self) -> None:
        self._stable = None
        self._hits = 0

    def update(self, frame_idx: int, dets: List[Dict[str, Any]]) -> Optional[RimStable]:
        """
        Update and (optionally) return a stabilized rim reference.

        Returns None during warmup (until enough hits are collected),
        then returns a RimStable and continues returning it during short dropouts.
        """
        rim = self._pick_best_rim(dets)

        if rim is not None:
            conf = float(rim.get("conf", 0.0))
            if conf >= self.conf_min:
                x1 = float(rim["x1"])
                y1 = float(rim["y1"])
                x2 = float(rim["x2"])
                y2 = float(rim["y2"])
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                if not all(self._finite(v) for v in (x1, y1, x2, y2, cx, cy)):
                    rim = None
                else:
                    if self._stable is None:
                        self._stable = RimStable(
                            cx=cx,
                            cy=cy,
                            bbox=(x1, y1, x2, y2),
                            conf=conf,
                            last_seen_frame=int(frame_idx),
                        )
                    else:
                        a = self.alpha
                        sx1, sy1, sx2, sy2 = self._stable.bbox
                        scx, scy = self._stable.cx, self._stable.cy

                        cx_c = self._clip_step(scx, cx, self.max_step_px)
                        cy_c = self._clip_step(scy, cy, self.max_step_px)
                        x1_c = self._clip_step(sx1, x1, self.max_step_px)
                        y1_c = self._clip_step(sy1, y1, self.max_step_px)
                        x2_c = self._clip_step(sx2, x2, self.max_step_px)
                        y2_c = self._clip_step(sy2, y2, self.max_step_px)

                        scx = (1.0 - a) * scx + a * cx_c
                        scy = (1.0 - a) * scy + a * cy_c
                        sx1 = (1.0 - a) * sx1 + a * x1_c
                        sy1 = (1.0 - a) * sy1 + a * y1_c
                        sx2 = (1.0 - a) * sx2 + a * x2_c
                        sy2 = (1.0 - a) * sy2 + a * y2_c

                        self._stable = RimStable(
                            cx=scx,
                            cy=scy,
                            bbox=(sx1, sy1, sx2, sy2),
                            conf=conf,
                            last_seen_frame=int(frame_idx),
                        )

                    self._hits += 1
                    if self._hits >= self.warmup_min_hits:
                        return self._stable
                    return None

        if self._stable is None:
            return None

        age = int(frame_idx) - int(self._stable.last_seen_frame)
        if age <= self.hold_frames and self._hits >= self.warmup_min_hits:
            return self._stable

        return None
