from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

BBox = Tuple[float, float, float, float]

@dataclass(frozen=True)
class RimScale:
    rim_w: float
    rim_h: float

    def px(self, rel: float, *, min_px: float = 0.0) -> float:
        # distance horizontale typiquement liée à rim_w
        return max(float(min_px), float(rel) * float(self.rim_w))

    def py(self, rel: float, *, min_px: float = 0.0) -> float:
        # distance verticale parfois mieux liée à rim_h
        return max(float(min_px), float(rel) * float(self.rim_h))

def from_rim_bbox(rim_bbox: Optional[BBox]) -> Optional[RimScale]:
    if rim_bbox is None:
        return None
    x1, y1, x2, y2 = map(float, rim_bbox)
    return RimScale(rim_w=max(1.0, x2-x1), rim_h=max(1.0, y2-y1))
