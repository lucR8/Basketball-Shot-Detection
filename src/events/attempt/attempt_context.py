from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.events.spatial_gates import point_in_bbox, dist_point_to_bbox, person_cover_ratio

BBox = Tuple[float, float, float, float]


def pick_best(detections: List[Dict[str, Any]], cls_name: str) -> Optional[Dict[str, Any]]:
    cls = cls_name.lower()
    cand = [d for d in detections if str(d.get("name", "")).lower() == cls]
    if not cand:
        return None
    return max(cand, key=lambda d: float(d.get("conf", 0.0)))


def det_bbox(det: Dict[str, Any]) -> BBox:
    return (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]))


def det_center(det: Dict[str, Any]) -> Tuple[float, float]:
    return (
        (float(det["x1"]) + float(det["x2"])) / 2.0,
        (float(det["y1"]) + float(det["y2"])) / 2.0,
    )


def ball_rel_y_in_person(bx: float, by: float, person_bbox: BBox) -> float:
    x1, y1, x2, y2 = person_bbox
    h = max(1.0, (y2 - y1))
    return float((by - y1) / h)  # 0=haut, 1=bas


@dataclass(frozen=True)
class AttemptContext:
    frame_idx: int

    # inputs
    shoot_info: Dict[str, Any]
    shoot_bbox: BBox
    person_bbox: BBox
    ball_xy: Tuple[float, float]
    ball_src: str

    # scaling
    scale: float
    pad_person: float
    pad_ball: float
    dist_thr: float
    sep_thr: float

    # derived
    cover: float
    person_in_shoot: bool
    ball_in_shoot: bool
    d_ball_person: float
    ball_rel_y: float
    rel_y_rise: float


def build_context(
    *,
    frame_idx: int,
    shoot_info: Dict[str, Any],
    shoot_bbox: BBox,
    person_bbox: BBox,
    ball_point: Tuple[float, float, str],
    scale: float,
    person_in_shoot_min_cover: float,
    person_in_shoot_pad_px: float,
    ball_in_shoot_pad_px: float,
    ball_person_max_dist_px: float,
    release_sep_increase_px: float,
    armed_ball_rel_y: Optional[float],
) -> AttemptContext:
    bx = float(ball_point[0])
    by = float(ball_point[1])
    src = str(ball_point[2])

    pad_person = float(person_in_shoot_pad_px) * float(scale)
    pad_ball = float(ball_in_shoot_pad_px) * float(scale)
    dist_thr = float(ball_person_max_dist_px) * float(scale)
    sep_thr = float(release_sep_increase_px) * float(scale)

    cover = float(person_cover_ratio(person_bbox, shoot_bbox, pad=pad_person))
    person_in_shoot = bool(cover >= float(person_in_shoot_min_cover))

    ball_in_shoot = bool(point_in_bbox(bx, by, shoot_bbox, pad=pad_ball))
    d_bp = float(dist_point_to_bbox(bx, by, person_bbox))

    rel_y = float(ball_rel_y_in_person(bx, by, person_bbox))

    rel_y_rise = 0.0
    if armed_ball_rel_y is not None:
        rel_y_rise = float(armed_ball_rel_y - rel_y)

    return AttemptContext(
        frame_idx=int(frame_idx),
        shoot_info=shoot_info,
        shoot_bbox=shoot_bbox,
        person_bbox=person_bbox,
        ball_xy=(bx, by),
        ball_src=src,
        scale=float(scale),
        pad_person=float(pad_person),
        pad_ball=float(pad_ball),
        dist_thr=float(dist_thr),
        sep_thr=float(sep_thr),
        cover=float(cover),
        person_in_shoot=bool(person_in_shoot),
        ball_in_shoot=bool(ball_in_shoot),
        d_ball_person=float(d_bp),
        ball_rel_y=float(rel_y),
        rel_y_rise=float(rel_y_rise),
    )
