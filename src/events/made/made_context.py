from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class MadeContext:
    """
    Per-frame geometric context for outcome reasoning.

    This object contains only measurements, not decisions.
    It allows MadeDetector to keep the update() logic readable and auditable.
    """

    frame_idx: int

    # Ball observation
    ball_xy: Tuple[float, float]

    # Rim reference
    rim_xy: Tuple[float, float]
    rim_bbox: Optional[BBox]

    # Rim reference lines
    y_line: float
    y_line_from_bbox: bool
    below_line: Optional[float]

    # Convenience measurements
    dist_to_rim: float


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_y_line(rim_cy: float, rim_bbox: Optional[BBox], rim_line_rel_y: float) -> Tuple[float, bool]:
    """
    Compute the rim-plane y reference.

    If rim_bbox is available:
    - place y_line at a relative vertical position inside the bbox.
    Otherwise:
    - fall back to rim_cy (less precise); caller should use higher epsilon tolerance.
    """
    if rim_bbox is None:
        return float(rim_cy), False

    x1, y1, x2, y2 = rim_bbox
    h = max(1.0, (y2 - y1))
    rel = min(0.60, max(0.05, float(rim_line_rel_y)))
    return float(y1 + rel * h), True


def compute_below_line(rim_bbox: Optional[BBox], below_rim_rel_y: float) -> Optional[float]:
    """
    Secondary y-line used to confirm the ball is clearly below the rim.

    Only meaningful if rim_bbox exists (we need rim bbox height).
    """
    if rim_bbox is None:
        return None
    x1, y1, x2, y2 = rim_bbox
    h = max(1.0, (y2 - y1))
    rel = min(0.98, max(0.40, float(below_rim_rel_y)))
    return float(y1 + rel * h)


def rim_size(rim_bbox: Optional[BBox]) -> Optional[Tuple[float, float]]:
    if rim_bbox is None:
        return None
    x1, y1, x2, y2 = rim_bbox
    return (max(1.0, x2 - x1), max(1.0, y2 - y1))


def center_x_gate_thr(rim_bbox: Optional[BBox], center_gate_radius_rel: float, center_gate_min_px: float) -> Optional[float]:
    """Horizontal threshold (px) for 'center trajectory evidence'."""
    size = rim_size(rim_bbox)
    if size is None:
        return None
    rim_w, _ = size
    return float(max(center_gate_min_px, float(center_gate_radius_rel) * rim_w))


def rim_gate_radii(
    rim_bbox: Optional[BBox],
    gate_rx_rel: float,
    gate_ry_rel: float,
) -> Optional[Tuple[float, float]]:
    """
    Elliptical gate radii around rim center for validating plane-crossing x.

    If rim_bbox is missing, caller should use a conservative pixel fallback.
    """
    if rim_bbox is None:
        return None
    x1, y1, x2, y2 = rim_bbox
    rim_w = max(1.0, x2 - x1)
    rim_h = max(1.0, y2 - y1)
    rx = max(6.0, float(gate_rx_rel) * rim_w)
    ry = max(3.0, float(gate_ry_rel) * rim_h)
    return float(rx), float(ry)


# --------------------------------------------------------------------
# Rim-size scaling helpers (turn pixel thresholds into rim-relative)
# --------------------------------------------------------------------
def rim_width(rim_bbox: Optional[BBox]) -> Optional[float]:
    """Return rim bbox width in pixels, if available."""
    size = rim_size(rim_bbox)
    if size is None:
        return None
    return float(size[0])


def rim_scale_from_bbox(
    rim_bbox: Optional[BBox],
    *,
    rim_ref_w: float,
    scale_min: float,
    scale_max: float,
) -> float:
    """
    Convert pixel thresholds into "rim-size-relative" thresholds.

    We estimate a per-video scale factor from the observed rim width:
        scale = rim_w / rim_ref_w

    - rim_ref_w: expected rim bbox width (px) at which pixel thresholds were tuned.
    - scale is clamped to stabilize behavior under noisy bbox estimates.
    """
    w = rim_width(rim_bbox)
    if w is None or rim_ref_w <= 1e-6:
        return 1.0
    s = float(w) / float(rim_ref_w)
    return float(min(max(s, float(scale_min)), float(scale_max)))


def scale_px(px: float, scale: float) -> float:
    """Scale a pixel threshold by the current rim scale factor."""
    return float(px) * float(scale)


def build_context(
    *,
    frame_idx: int,
    ball_xy: Tuple[float, float],
    rim_xy: Tuple[float, float],
    rim_bbox: Optional[BBox],
    rim_line_rel_y: float,
    below_rim_rel_y: float,
) -> MadeContext:
    rim_cx, rim_cy = rim_xy
    y_line, y_line_from_bbox = compute_y_line(rim_cy, rim_bbox, rim_line_rel_y)
    below_line = compute_below_line(rim_bbox, below_rim_rel_y)
    d = dist(ball_xy, rim_xy)

    return MadeContext(
        frame_idx=int(frame_idx),
        ball_xy=(float(ball_xy[0]), float(ball_xy[1])),
        rim_xy=(float(rim_cx), float(rim_cy)),
        rim_bbox=rim_bbox,
        y_line=float(y_line),
        y_line_from_bbox=bool(y_line_from_bbox),
        below_line=float(below_line) if below_line is not None else None,
        dist_to_rim=float(d),
    )
