from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from src.events.attempt.attempt_context import AttemptContext
from src.events.attempt_fsm import AttemptFSM


@dataclass(frozen=True)
class TrackerMatchParams:
    """
    Parameters for the ghost-ball gate:
    compare YOLO ball point vs tracker point when the ball is low near the player.
    """
    tracker_match_max_dx_px: float
    tracker_match_max_dy_px: float
    tracker_gate_min_ball_rel_y: float


def gate_ball_close_person(ctx: AttemptContext) -> bool:
    """
    Ball/person proximity gate.

    Rationale:
    - Many false "ball" detections occur far from the shooter.
    - However, if the ball is inside the shoot bbox, we trust that configuration
      even if the ball/person distance is large (bbox geometry can be imperfect).
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
    "Ball too low" gate (pre-arming only).

    Engineering intent:
    - Before we commit to ARMED, we want to avoid triggering shots when the ball is
      at a dribble / carry position (low in the player's bbox).
    - Once ARMED, we do not apply this gate anymore (otherwise we'd cancel valid shots).

    Also:
    - If the ball is inside the shoot bbox, we accept it even if it's low,
      because shoot bbox is the strongest cue.
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
    Ghost-ball suppressor.

    Motivation:
    - Low-confidence YOLO "ball" detections may jump to hands/heads/jerseys.
    - The tracker provides continuity; when the ball is low near the player, we expect
      YOLO and tracker points to agree approximately.

    This gate is applied only when:
    - A tracker point exists (ball_state is not None)
    - The current ball source is not the tracker (i.e., YOLO is driving the point)
    - The ball is low in the person bbox (ball_rel_y >= threshold)
      (heuristic region where false positives are common)

    Returns:
    - (ok, dx, dy) where dx/dy are absolute pixel differences for debugging.
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
