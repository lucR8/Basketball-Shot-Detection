from __future__ import annotations

import cv2
from typing import List, Tuple, Optional

from src.events.attempt import AttemptEvent
from src.events.made.made import MadeEvent


def draw_boxes(frame, detections, show_label: bool = True):
    """Draws bounding boxes for a list of YOLO-like detections (debug utility)."""
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
    """
    Draw the recent ball trajectory.

    Rationale:
    - Ball tracking is a temporal component; a short trace makes motion continuity
      (and potential tracking errors) visually obvious.
    """
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
    Draw a semi-transparent panel with multiple text lines.

    This keeps overlays readable without hiding the video entirely.
    Returns the panel bounding box (x1, y1, x2, y2).
    """
    if not lines:
        return (x, y_top, x, y_top)

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

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0, frame)

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
    """
    Visualize an AttemptEvent.

    Shows:
    - Rim center, ball point at trigger time, and their distance
    - A compact panel with the attempt id and short details
    """
    if evt is None:
        return frame

    rim_pt = (int(evt.rim_cx), int(evt.rim_cy))
    ball_pt = (int(evt.ball_cx), int(evt.ball_cy))

    cv2.circle(frame, rim_pt, int(radius_px), (0, 255, 255), 2)
    cv2.circle(frame, ball_pt, 6, (0, 0, 255), -1)
    cv2.line(frame, ball_pt, rim_pt, (255, 255, 0), 2)

    detail = (evt.details or "")
    detail_lines = []
    if detail:
        detail_lines = [detail[i:i + 60] for i in range(0, min(len(detail), 120), 60)]

    _draw_text_panel(
        frame,
        [f"ATTEMPT #{attempt_count}", *detail_lines],
        x=20,
        y_top=450,
        font_scale=0.7,
        line_h=26,
        bg_alpha=0.55,
    )
    return frame


def draw_scoreboard(frame, attempts: int, made: int, miss: int, unknown: int):
    """
    Always-on summary of final statistics.

    This is intentionally independent from the FSM internals:
    it reports only finalized outcomes (Made/Miss/Unknown).
    """
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
    """
    Display the outcome label at decision time (one-frame event).

    Engineering intent:
    - Provide immediate feedback when the outcome FSM commits.
    - Optionally display short decision details for debugging/justification.
    """
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

    details = (evt.details or "").strip()
    if details:
        wrapped = [details[i:i + 70] for i in range(0, min(len(details), 280), 70)]
        _draw_text_panel(
            frame,
            ["details:"] + wrapped,
            x=30,
            y_top=200,
            font_scale=0.55,
            thickness=2,
            line_h=20,
            pad=10,
            bg_alpha=0.55,
        )

    return frame


def draw_attempt_gating_debug(frame, attempt_detector, org=(20, 140)):
    """
    Inspect AttemptDetector gating decisions in real time.

    Reads attempt_detector.last_debug (a dict populated by the FSM).
    This is meant for analysis and threshold tuning, not for end users.
    """
    if attempt_detector is None:
        return frame

    dbg = getattr(attempt_detector, "last_debug", None)
    if not isinstance(dbg, dict) or not dbg:
        return frame

    def _safe_float(v, default=0.0):
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _safe_int(v, default=0):
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _safe_bool(v, default=False):
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in ("true", "1", "yes", "y"):
                return True
            if vv in ("false", "0", "no", "n"):
                return False
        return bool(v)

    gate = str(dbg.get("gate_reason", ""))

    shoot_now = _safe_bool(dbg.get("shoot_now", False))
    shoot_rise = _safe_bool(dbg.get("shoot_rise", False))
    shoot_streak = _safe_int(dbg.get("shoot_streak", 0))
    shoot_conf = _safe_float(dbg.get("shoot_conf", 0.0), 0.0)
    shoot_from_memory = _safe_bool(dbg.get("shoot_from_memory", False))

    fsm_state = str(dbg.get("fsm_state", dbg.get("state", "")))
    streak = _safe_int(dbg.get("release_streak", 0))
    deb = _safe_int(dbg.get("release_debounce", 0))

    p_in = _safe_bool(dbg.get("person_in_shoot", False))
    b_in = _safe_bool(dbg.get("ball_in_shoot", False))
    left = _safe_bool(dbg.get("left_shoot", False))
    left_raw = _safe_bool(dbg.get("left_shoot_raw", False))
    sep_ok = _safe_bool(dbg.get("sep_ok", False))
    raw_sep_ok = _safe_bool(dbg.get("raw_sep_ok", False))

    d_bp = _safe_float(dbg.get("d_ball_person", None), 1e9)
    d_bp_max = _safe_float(dbg.get("ball_person_max_dist_px", dbg.get("dist_thr", None)), 0.0)
    sep_thr = _safe_float(dbg.get("sep_thr", None), 0.0)

    ball_rel_y = _safe_float(dbg.get("ball_rel_y", None), -1.0)
    armed_ball_rel_y = _safe_float(dbg.get("armed_ball_rel_y", None), -1.0)
    rel_y_rise = _safe_float(dbg.get("rel_y_rise", None), 0.0)

    rise_arm_ok = _safe_bool(dbg.get("rise_arm_ok", False))
    streak_arm_ok = _safe_bool(dbg.get("streak_arm_ok", False))

    p_overlap_r = dbg.get("person_overlap_ratio", None)
    p_overlap_str = "-" if p_overlap_r is None else f"{_safe_float(p_overlap_r):.2f}"

    iou_ps = dbg.get("iou_person_shoot", None)
    iou_str = "-" if iou_ps is None else f"{_safe_float(iou_ps):.3f}"

    ball_src = dbg.get("ball_src", dbg.get("ball_source", None))
    ball_src_str = "-" if ball_src is None else str(ball_src)

    scale = _safe_float(dbg.get("scale", None), 1.0)

    armed = (fsm_state == "ARMED") or _safe_bool(dbg.get("armed", False))

    lines = [
        f"Attempt gate: {gate}",
        f"fsm={fsm_state} armed={armed}",
        f"shoot_now={shoot_now} rise={shoot_rise} streak={shoot_streak} conf={shoot_conf:.2f} mem={shoot_from_memory}",
        f"arm_ok: rise={rise_arm_ok} streak={streak_arm_ok}",
        f"person_in_shoot={p_in} ball_in_shoot={b_in} src={ball_src_str} scale={scale:.2f}",
        f"d_ball_person={d_bp:.1f}/{d_bp_max:.0f} sep_thr={sep_thr:.1f}",
        f"release: left={left} (raw={left_raw}) sep_ok={sep_ok} (raw={raw_sep_ok}) streak={streak}/{deb}",
        f"rel_y={ball_rel_y:.2f} armed_rel_y={armed_ball_rel_y:.2f} rise={rel_y_rise:.2f}",
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

    # Optional visualization: ball-to-person segment (helps see association issues).
    ball_xy = dbg.get("ball_xy", None)
    person_bbox = dbg.get("person_bbox", None)
    if (
        isinstance(ball_xy, (tuple, list))
        and len(ball_xy) >= 2
        and isinstance(person_bbox, (tuple, list))
        and len(person_bbox) == 4
    ):
        bx, by = float(ball_xy[0]), float(ball_xy[1])
        x1, y1, x2, y2 = map(float, person_bbox)
        pcx = 0.5 * (x1 + x2)
        pcy = 0.5 * (y1 + y2)
        cv2.line(frame, (int(bx), int(by)), (int(pcx), int(pcy)), (255, 255, 255), 2)

    return frame


def draw_made_debug(frame, made_detector):
    """
    Visualize MadeDetector internal geometry:
    - rim center
    - rim plane (y_line)
    - gates used for decision (center band / below rim line)
    """
    if not getattr(made_detector, "active", False):
        return frame

    cx, cy = int(made_detector._rim_cx), int(made_detector._rim_cy)
    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

    x_thr = made_detector._center_x_gate_thr()
    if x_thr is not None:
        x1 = int(cx - x_thr)
        x2 = int(cx + x_thr)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (x1, 0), (x2, h), (0, 255, 0), 1)

    y_line = int(made_detector._y_line)
    cv2.line(frame, (0, y_line), (frame.shape[1], y_line), (255, 0, 0), 1)

    y_below = made_detector._compute_below_rim_line()
    if y_below is not None:
        cv2.line(frame, (0, int(y_below)), (frame.shape[1], int(y_below)), (0, 255, 255), 1)

    if made_detector._dbg_plane_cross_pt is not None:
        x, y = made_detector._dbg_plane_cross_pt
        cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)

    for (x, y) in made_detector._dbg_center_hits_pts[-10:]:
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    for (x, y) in made_detector._dbg_below_hits_pts[-10:]:
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

    return frame
