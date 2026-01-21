from __future__ import annotations

from typing import Optional, Tuple
import math

BBox = Tuple[float, float, float, float]


def point_in_bbox(x: float, y: float, bbox: BBox) -> bool:
    x1, y1, x2, y2 = bbox
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def expanded_rim_bbox(rim_bbox: Optional[BBox], expand_factor: float) -> Optional[BBox]:
    """Enlarge rim bbox to tolerate jitter / slightly tight boxes."""
    if rim_bbox is None:
        return None
    x1, y1, x2, y2 = rim_bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, (x2 - x1)) * float(expand_factor)
    h = max(1.0, (y2 - y1)) * float(expand_factor)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def near_rim_now(
    *,
    ball_xy: Tuple[float, float],
    rim_xy: Tuple[float, float],
    near_rim_dist_px: float,
    rim_bbox: Optional[BBox],
    rim_expand_factor: float,
) -> bool:
    """
    Decide whether the current ball point is "near the hoop".

    This predicate is used for:
    - grace windows: avoid deciding too early when the ball is still around the rim
    - early-miss: detect obvious misses when the ball leaves far after approaching

    When rim bbox exists, we use an *elliptical* gate derived from an expanded bbox.
    This better matches hoop geometry (wider horizontally) and tolerates localization jitter.
    """
    cx, cy = ball_xy
    rx, ry = rim_xy

    # (1) Distance-to-center fallback (works even if bbox is missing).
    d = math.hypot(cx - rx, cy - ry)
    if d <= float(near_rim_dist_px):
        return True

    exp = expanded_rim_bbox(rim_bbox, rim_expand_factor)
    if exp is None:
        return False

    x1, y1, x2, y2 = exp
    ex_cx = 0.5 * (x1 + x2)
    ex_cy = 0.5 * (y1 + y2)
    ex_w = max(1.0, x2 - x1)
    ex_h = max(1.0, y2 - y1)

    # (2) Ellipse around expanded bbox (slightly more permissive than the box itself).
    slack = 1.15
    erx = 0.5 * ex_w * slack
    ery = 0.5 * ex_h * slack

    dx = (cx - ex_cx) / erx
    dy = (cy - ex_cy) / ery
    if (dx * dx + dy * dy) <= 1.0:
        return True

    # (3) Expanded bbox containment as a simple fallback.
    return point_in_bbox(cx, cy, exp)


def inside_rim_gate(
    *,
    x: float,
    y: float,
    rim_cx: float,
    y_line: float,
    radii: Optional[Tuple[float, float]],
    fallback_x_px: float,
    fallback_y_px: float,
) -> bool:
    """
    Validate that a rim-plane crossing occurs close enough to the rim center.

    If radii (rx, ry) are available (rim bbox exists), use an ellipse test.
    Otherwise fall back to conservative pixel thresholds.
    """
    if radii is None:
        return (abs(x - rim_cx) <= float(fallback_x_px)) and (abs(y - y_line) <= float(fallback_y_px))

    rx, ry = radii
    dx = (x - rim_cx) / float(rx)
    dy = (y - y_line) / float(ry)
    return (dx * dx + dy * dy) <= 1.0


def find_plane_crossing(
    *,
    prev_xy: Tuple[float, float],
    cur_xy: Tuple[float, float],
    y_line: float,
    eps: float,
) -> Optional[Tuple[float, float]]:
    """
    Detect a downward crossing of the rim plane:
      prev.y <= y_line - eps  AND  cur.y >= y_line + eps

    Returns:
    - (x_cross, y_line) if a crossing occurs
    - None otherwise
    """
    x0, y0 = prev_xy
    x1, y1 = cur_xy
    if not ((y0 <= y_line - eps) and (y1 >= y_line + eps)):
        return None

    t = (y_line - y0) / max(1e-6, (y1 - y0))
    x_cross = x0 + t * (x1 - x0)
    return (float(x_cross), float(y_line))


def below_gate_hit(
    *,
    ball_xy: Tuple[float, float],
    rim_xy: Tuple[float, float],
    rim_bbox: Optional[BBox],
    radius_rel: float,
    min_px: float,
) -> bool:
    """
    Additional "swish-like" evidence under the rim:
    ball point close to rim center after it went below the hoop.
    """
    cx, cy = ball_xy
    rx, ry = rim_xy

    if rim_bbox is None:
        thr = float(min_px)
        return math.hypot(cx - rx, cy - ry) <= thr

    x1, y1, x2, y2 = rim_bbox
    rim_w = max(1.0, x2 - x1)
    thr = max(float(min_px), float(radius_rel) * rim_w)
    return math.hypot(cx - rx, cy - ry) <= thr
