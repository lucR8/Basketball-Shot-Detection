from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

from src.events.attempt.attempt_context import AttemptContext
from src.events.attempt_fsm import AttemptFSM


@dataclass(frozen=True)
class ReleaseInfo:
    left_shoot_raw: bool
    left_shoot: bool
    raw_sep_ok: bool
    sep_ok: bool
    release_by_motion: bool
    vy: Optional[float]
    vy_thr: Optional[float]
    release_signal: bool


def compute_release(
    ctx: AttemptContext,
    *,
    fsm_state: str,
    armed_dist_bp: Optional[float],
    min_rel_y_rise_for_sep: float,

    # sep_thr is optional: fallback to ctx.sep_thr
    sep_thr: Optional[float] = None,

    # Motion release is OFF by default (safe)
    enable_motion_release: bool = False,
    armed_clean: bool = True,

    # canonical param name (internal)
    motion_ball_rel_y_max: float = 0.55,

    # direct velocity params (optional)
    vy: Optional[float] = None,
    vy_thr: Optional[float] = None,

    # detector may pass this
    ball_state: Optional[Any] = None,

    # Backward-compatible names
    motion_release_vy: Optional[float] = None,
    motion_release_vy_thr: Optional[float] = None,
    motion_release_max_ball_rel_y: Optional[float] = None,
) -> ReleaseInfo:
    """
    Compute release signal.

    Compatible with multiple detector call signatures.
    - sep_thr defaults to ctx.sep_thr if not provided.
    - accepts ball_state, motion_* aliases.
    """

    # Resolve sep_thr
    if sep_thr is None:
        sep_thr = float(getattr(ctx, "sep_thr", 0.0))

    # Resolve aliases
    if vy is None and motion_release_vy is not None:
        vy = motion_release_vy
    if vy_thr is None and motion_release_vy_thr is not None:
        vy_thr = motion_release_vy_thr
    if motion_release_max_ball_rel_y is not None:
        motion_ball_rel_y_max = float(motion_release_max_ball_rel_y)

    # If vy not provided, try reading from ball_state
    if vy is None and ball_state is not None:
        try:
            vy = float(getattr(ball_state, "vy"))
        except Exception:
            vy = None

    # left_shoot
    left_raw = not bool(ctx.ball_in_shoot)
    left_ok = bool(left_raw and (ctx.rel_y_rise >= float(min_rel_y_rise_for_sep)))

    # sep_ok
    raw_sep = False
    if fsm_state == AttemptFSM.ARMED and armed_dist_bp is not None:
        raw_sep = bool((ctx.d_ball_person - float(armed_dist_bp)) >= float(sep_thr))
    sep_ok = bool(raw_sep and (ctx.rel_y_rise >= float(min_rel_y_rise_for_sep)))

    # optional motion release
    rel_by_motion = False
    if enable_motion_release and (fsm_state == AttemptFSM.ARMED):
        if armed_clean and (ctx.ball_rel_y <= float(motion_ball_rel_y_max)):
            if (ctx.rel_y_rise >= float(min_rel_y_rise_for_sep)) and (vy is not None) and (vy_thr is not None):
                rel_by_motion = bool(float(vy) <= -float(vy_thr))

    release_signal = bool(left_ok or sep_ok or rel_by_motion)

    return ReleaseInfo(
        left_shoot_raw=bool(left_raw),
        left_shoot=bool(left_ok),
        raw_sep_ok=bool(raw_sep),
        sep_ok=bool(sep_ok),
        release_by_motion=bool(rel_by_motion),
        vy=float(vy) if isinstance(vy, (int, float)) else None,
        vy_thr=float(vy_thr) if isinstance(vy_thr, (int, float)) else None,
        release_signal=bool(release_signal),
    )
