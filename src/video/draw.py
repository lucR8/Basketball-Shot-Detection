from __future__ import annotations

import cv2
from typing import List, Tuple, Optional

from src.events.attempt import AttemptEvent
from src.events.made import MadeEvent


def draw_boxes(frame, detections, show_label: bool = True):
    """
    detections: list of dict with keys: x1,y1,x2,y2,conf,cls,name
    """
    for d in detections:
        x1, y1, x2, y2 = map(int, (d["x1"], d["y1"], d["x2"], d["y2"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if show_label:
            label = f'{d.get("name","cls")} {float(d.get("conf",0.0)):.2f}'
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    return frame


def draw_ball_trace(
    frame,
    trace: List[Tuple[float, float]],
    max_length: int = 30,
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 3,
):
    """Draw recent trajectory of the ball."""
    if not trace:
        return frame

    pts = trace[-max_length:]

    for i, (x, y) in enumerate(pts):
        alpha = (i + 1) / len(pts)
        r = max(1, int(radius * alpha))
        cv2.circle(frame, (int(x), int(y)), r, color, -1)

    for i in range(1, len(pts)):
        p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p2 = (int(pts[i][0]), int(pts[i][1]))
        cv2.line(frame, p1, p2, color, 2)

    return frame


def _draw_text_panel(
    frame,
    lines: List[str],
    x: int,
    y_top: int,
    font_scale: float = 0.65,
    thickness: int = 2,
    line_h: int = 22,
    pad: int = 10,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.65,
) -> Tuple[int, int, int, int]:
    """
    Draw a semi-transparent rectangle + multiple lines of text.
    Returns (x1, y1, x2, y2) panel bbox.
    """
    if not lines:
        return (x, y_top, x, y_top)

    # Measure width
    max_w = 0
    for t in lines:
        (w, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_w = max(max_w, w)

    panel_w = max_w + 2 * pad
    panel_h = len(lines) * line_h + 2 * pad

    x1 = x
    y1 = y_top
    x2 = x + panel_w
    y2 = y_top + panel_h

    # alpha background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0, frame)

    # draw text
    y = y_top + pad + int(line_h * 0.8)
    for t in lines:
        cv2.putText(
            frame,
            t,
            (x + pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_h

    return (x1, y1, x2, y2)


def draw_attempt_debug(
    frame,
    evt: Optional[AttemptEvent],
    attempt_count: int,
    radius_px: int = 85,
):
    """Visual debug for a detected shot attempt (only when evt is not None)."""
    if evt is None:
        return frame

    rim_pt = (int(evt.rim_cx), int(evt.rim_cy))
    ball_pt = (int(evt.ball_cx), int(evt.ball_cy))

    cv2.circle(frame, rim_pt, int(radius_px), (0, 255, 255), 2)
    cv2.circle(frame, ball_pt, 6, (0, 0, 255), -1)
    cv2.line(frame, ball_pt, rim_pt, (255, 255, 0), 2)

    # Put attempt label under the scoreboard area to avoid collisions
    _draw_text_panel(
        frame,
        [f"ATTEMPT #{attempt_count}", (evt.details or "")[:50]],
        x=20,
        y_top=450,
        font_scale=0.7,
        line_h=26,
        bg_alpha=0.55,
    )

    return frame


def draw_scoreboard(frame, attempts: int, made: int, miss: int, unknown: int):
    """Always-on overlay (NO airball)."""
    lines = [
        f"Attempts: {attempts}",
        f"Made: {made}",
        f"Miss: {miss}",
        f"Unknown: {unknown}",
    ]
    _draw_text_panel(
        frame,
        lines,
        x=20,
        y_top=15,
        font_scale=0.75,
        thickness=2,
        line_h=28,
        pad=10,
        bg_alpha=0.65,
    )
    return frame


def draw_result_flash(frame, evt: Optional[MadeEvent]):
    """Big label that appears only when a decision is made."""
    if evt is None:
        return frame

    label = str(evt.outcome).upper()

    if evt.outcome == "made":
        color = (0, 255, 0)
    elif evt.outcome == "miss":
        color = (0, 255, 255)
    else:
        color = (255, 255, 255)

    cv2.putText(frame, label, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5, cv2.LINE_AA)
    return frame


def draw_attempt_gating_debug(frame, attempt_detector, org=(20, 140)):
    """
    Visual overlay for AttemptDetector gating.
    - Does NOT overlap scoreboard (default org y=140).
    - Draws background rectangle BEFORE text.
    Reads attempt_detector.last_debug (dict).
    """
    if attempt_detector is None:
        return frame

    dbg = getattr(attempt_detector, "last_debug", None)
    if not isinstance(dbg, dict) or not dbg:
        return frame

    def _safe_float(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    gate = str(dbg.get("gate_reason", ""))
    armed = bool(dbg.get("armed", False))
    shoot_now = bool(dbg.get("shoot_now", False))
    shoot_rise = bool(dbg.get("shoot_rise", False))
    shoot_streak = int(dbg.get("shoot_streak", 0))

    p_in = bool(dbg.get("person_in_shoot", False))
    b_in = bool(dbg.get("ball_in_shoot", False))
    left = bool(dbg.get("left_shoot", False))
    sep_ok = bool(dbg.get("sep_ok", False))

    streak = int(dbg.get("release_streak", 0))
    deb = int(dbg.get("release_debounce", 0))

    shoot_conf = float(dbg.get("shoot_conf", 0.0))
    d_bp = float(dbg.get("d_ball_person", 1e9))
    d_bp_max = float(dbg.get("ball_person_max_dist_px", 0.0))
    
    p_overlap_r = dbg.get("person_overlap_ratio", None)
    p_overlap_str = "-" if p_overlap_r is None else f"{_safe_float(p_overlap_r):.2f}"
    
    iou_ps = dbg.get("iou_person_shoot", None)
    iou_str = "-" if iou_ps is None else f"{_safe_float(iou_ps):.3f}"

    lines = [
        f"Attempt gate: {gate}",
        f"armed={armed} shoot_now={shoot_now} rise={shoot_rise} streak={shoot_streak} conf={shoot_conf:.2f}",
        f"person_in_shoot={p_in} ball_in_shoot={b_in}",
        f"d_ball_person={d_bp:.1f}/{d_bp_max:.0f}",
        f"release: left={left} sep_ok={sep_ok} streak={streak}/{deb}",
        f"person_overlap_ratio={p_overlap_str} iou={iou_str}",
    ]


    x, y = org
    _draw_text_panel(
        frame,
        lines,
        x=x,
        y_top=y,
        font_scale=0.55,
        thickness=2,
        line_h=20,
        pad=10,
        bg_alpha=0.60,
    )

    # Optional: draw line ball->person
    ball_xy = dbg.get("ball_xy", None)
    person_bbox = dbg.get("person_bbox", None)
    if ball_xy is not None and person_bbox is not None:
        bx, by = ball_xy
        x1, y1, x2, y2 = person_bbox
        pcx = 0.5 * (x1 + x2)
        pcy = 0.5 * (y1 + y2)
        cv2.line(frame, (int(bx), int(by)), (int(pcx), int(pcy)), (255, 255, 255), 2)

    return frame
