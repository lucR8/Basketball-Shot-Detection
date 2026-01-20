from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from src.events.attempt.attempt_context import AttemptContext
from src.events.attempt_fsm import AttemptFSM


@dataclass(frozen=True)
class TrackerMatchParams:
    tracker_match_max_dx_px: float
    tracker_match_max_dy_px: float
    tracker_gate_min_ball_rel_y: float


def gate_ball_close_person(ctx: AttemptContext) -> bool:
    """
    Soft rule:
      - If ball is inside shoot bbox, ignore ball/person distance.
      - Else require ball close enough to person bbox.
    """
    if ctx.ball_in_shoot:
        return True
    return ctx.d_ball_person <= ctx.dist_thr


def gate_ball_too_low_before_armed(
    ctx: AttemptContext,
    *,
    ball_too_low_rel_y: float,
    fsm_state: str,
) -> bool:
    """
    Reject "ball too low" only BEFORE ARMED and only when ball is NOT in shoot bbox.
    """
    if fsm_state == AttemptFSM.ARMED:
        return True
    if ctx.ball_in_shoot:
        return True
    return ctx.ball_rel_y < float(ball_too_low_rel_y)


def gate_tracker_match_if_low(
    ctx: AttemptContext,
    *,
    ball_state,
    params: TrackerMatchParams,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Ghost-ball suppressor:
      - Only if ball is low in person bbox (ctx.ball_rel_y >= min),
      - Only if src != tracker,
      - Compare YOLO ball position to tracker position.

    Returns:
      (ok, dx, dy)
    """
    if ball_state is None:
        return True, None, None

    if ctx.ball_src == "tracker":
        return True, None, None

    if ctx.ball_rel_y < float(params.tracker_gate_min_ball_rel_y):
        return True, None, None

    dx = abs(float(ball_state.cx) - float(ctx.ball_xy[0]))
    dy = abs(float(ball_state.cy) - float(ctx.ball_xy[1]))

    ok = (dx <= float(params.tracker_match_max_dx_px)) and (dy <= float(params.tracker_match_max_dy_px))
    return ok, float(dx), float(dy)
