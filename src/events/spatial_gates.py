from __future__ import annotations

from typing import Tuple
import math

BBox = Tuple[float, float, float, float]


def clip_bbox(b: BBox) -> BBox:
    """
    Normalize a bbox so that (x1 <= x2) and (y1 <= y2).

    This protects geometric computations from occasional swapped coordinates.
    """
    x1, y1, x2, y2 = b
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (float(x1), float(y1), float(x2), float(y2))


def point_in_bbox(x: float, y: float, b: BBox, pad: float = 0.0) -> bool:
    """Test point inclusion in bbox, with optional padding (tolerance)."""
    x1, y1, x2, y2 = clip_bbox(b)
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def dist_point_to_bbox(px: float, py: float, b: BBox) -> float:
    """
    Euclidean distance from a point to a bbox (0 if the point is inside).

    This is used as a stable alternative to "distance to bbox center":
    it better reflects whether the ball is close to the player's silhouette.
    """
    x1, y1, x2, y2 = clip_bbox(b)
    dx = 0.0
    if px < x1:
        dx = x1 - px
    elif px > x2:
        dx = px - x2

    dy = 0.0
    if py < y1:
        dy = y1 - py
    elif py > y2:
        dy = py - y2

    return math.hypot(dx, dy)


def bbox_area(b: BBox) -> float:
    x1, y1, x2, y2 = clip_bbox(b)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection(a: BBox, b: BBox) -> float:
    """Intersection area between two bboxes."""
    ax1, ay1, ax2, ay2 = clip_bbox(a)
    bx1, by1, bx2, by2 = clip_bbox(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-Union (IoU), standard overlap metric in detection pipelines."""
    inter = bbox_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    ua = bbox_area(a)
    ub = bbox_area(b)
    denom = ua + ub - inter
    return float(inter / denom) if denom > 1e-9 else 0.0


def expand_bbox(b: BBox, pad: float) -> BBox:
    """Expand bbox by `pad` pixels on all sides."""
    x1, y1, x2, y2 = clip_bbox(b)
    return (x1 - pad, y1 - pad, x2 + pad, y2 + pad)


def person_cover_ratio(person_bbox: BBox, shoot_bbox: BBox, pad: float = 0.0) -> float:
    """
    Coverage ratio = area(intersection(person, expand(shoot,pad))) / area(person)

    Used as a robust proxy for "the shoot cue belongs to this person":
    we measure how much of the person's box overlaps the shoot box (with tolerance).
    """
    pb = clip_bbox(person_bbox)
    sb = expand_bbox(shoot_bbox, pad)
    inter = bbox_intersection(pb, sb)
    a = max(1e-9, bbox_area(pb))
    return float(inter / a)
