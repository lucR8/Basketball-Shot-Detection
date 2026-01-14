from __future__ import annotations
import cv2
from typing import List, Tuple
from typing import Optional
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
            label = f'{d.get("name","cls")} {d["conf"]:.2f}'
            cv2.putText(frame, label, (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def draw_ball_trace(
    frame,
    trace: List[Tuple[float, float]],
    max_length: int = 30,
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 3,
):
    """
    Draw the recent trajectory of the ball on the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (OpenCV frame).
    trace : list of (cx, cy)
        History of ball center positions.
    max_length : int
        Number of recent points to draw.
    color : (B, G, R)
        Color of the trajectory.
    radius : int
        Radius of each point.
    """
    if not trace:
        return frame

    # Keep only the last N points
    pts = trace[-max_length:]

    # Draw points
    for i, (x, y) in enumerate(pts):
        alpha = (i + 1) / len(pts)  # fading effect
        r = max(1, int(radius * alpha))
        cv2.circle(frame, (int(x), int(y)), r, color, -1)

    # Draw connecting lines
    for i in range(1, len(pts)):
        p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p2 = (int(pts[i][0]), int(pts[i][1]))
        cv2.line(frame, p1, p2, color, 2)

    return frame

def draw_attempt_debug(
    frame,
    evt: Optional[AttemptEvent],
    attempt_count: int,
    radius_px: int = 85,
):
    """
    Draw visual debug for a detected shot attempt.
    """
    if evt is None:
        return frame

    rim_pt = (int(evt.rim_cx), int(evt.rim_cy))
    ball_pt = (int(evt.ball_cx), int(evt.ball_cy))

    # Rim ROI
    cv2.circle(frame, rim_pt, radius_px, (0, 255, 255), 2)

    # Ball position
    cv2.circle(frame, ball_pt, 6, (0, 0, 255), -1)

    # Line ball -> rim
    cv2.line(frame, ball_pt, rim_pt, (255, 255, 0), 2)

    # Text
    cv2.putText(
        frame,
        f"ATTEMPT #{attempt_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    return frame

def draw_scoreboard(frame, attempts: int, made: int, miss: int, airball: int, unknown: int):
    """
    Always-on overlay in the corner.
    """
    lines = [
        f"Attempts: {attempts}",
        f"Made: {made}",
        f"Miss: {miss}",
        f"Airball: {airball}",
        f"Unknown: {unknown}",
    ]

    x, y0 = 20, 30
    dy = 28

    # small background box for readability
    box_w, box_h = 260, dy * len(lines) + 15
    cv2.rectangle(frame, (x - 10, y0 - 22), (x - 10 + box_w, y0 - 22 + box_h), (0, 0, 0), -1)

    for i, t in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(frame, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame


def draw_result_flash(frame, evt: Optional[MadeEvent]):
    """
    Big label that appears only when a decision is made.
    """
    if evt is None:
        return frame

    label = evt.outcome.upper()
    # simple color mapping
    if evt.outcome == "made":
        color = (0, 255, 0)
    elif evt.outcome == "airball":
        color = (0, 0, 255)
    elif evt.outcome == "miss":
        color = (0, 255, 255)
    else:
        color = (255, 255, 255)

    cv2.putText(frame, label, (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)
    return frame
