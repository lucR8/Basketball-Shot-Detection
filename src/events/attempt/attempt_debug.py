from __future__ import annotations

from typing import Any, Dict, Optional

from src.events.attempt.attempt_context import AttemptContext
from src.events.attempt.attempt_release import ReleaseInfo
from src.events.attempt_fsm import AttemptFSM


def make_last_debug(
    *,
    frame_idx: int,
    gate_reason: str,
    fsm_state: str,
    shoot_info: Optional[Dict[str, Any]] = None,
    ctx: Optional[AttemptContext] = None,
    release: Optional[ReleaseInfo] = None,
    armed_ball_rel_y: Optional[float] = None,
    armed_frame: Optional[int] = None,
    armed_dist_bp: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a flat debug dict consumed by overlay code.

    Design goal:
    - Keep keys stable across refactors so the visualization remains compatible.
    - Use only JSON-like primitive values (float/int/bool/None/tuples) to make logs readable.

    This function intentionally does not compute decisions: it only reports
    the intermediate signals used by gates + FSM.
    """
    shoot_info = shoot_info or {}

    d: Dict[str, Any] = {
        "frame": int(frame_idx),
        "gate_reason": str(gate_reason),

        # FSM state (and derived "armed" boolean used by overlays)
        "fsm_state": str(fsm_state),
        "armed": bool(fsm_state == AttemptFSM.ARMED),

        # Filled later by attach_fsm_counters()
        "release_streak": 0,
        "release_debounce": 0,
    }

    # Shoot signal summary
    d.update({
        "shoot_now": bool(shoot_info.get("shoot_now", False)),
        "shoot_rise": bool(shoot_info.get("shoot_rise", False)),
        "shoot_streak": int(shoot_info.get("shoot_streak", 0)),
        "shoot_conf": float(shoot_info.get("shoot_conf", 0.0)),
        "shoot_from_memory": bool(shoot_info.get("shoot_from_memory", False)),
        "shoot_bbox": shoot_info.get("shoot_bbox", None),
    })

    # Spatial context (ball / person / shoot relations)
    if ctx is not None:
        d.update({
            "person_bbox": ctx.person_bbox,
            "ball_xy": ctx.ball_xy,
            "ball_src": ctx.ball_src,

            "person_overlap_ratio": float(ctx.cover),
            "person_in_shoot": bool(ctx.person_in_shoot),

            "ball_in_shoot": bool(ctx.ball_in_shoot),
            "d_ball_person": float(ctx.d_ball_person),
            "ball_person_max_dist_px": float(ctx.dist_thr),

            "ball_rel_y": float(ctx.ball_rel_y),
            "rel_y_rise": float(ctx.rel_y_rise),
            "scale": float(ctx.scale),

            # Legacy alias used by some overlays
            "dist_thr": float(ctx.dist_thr),
        })

    # Release diagnostic signals (left_shoot / sep_ok / motion release)
    if release is not None:
        d.update({
            "left_shoot_raw": bool(release.left_shoot_raw),
            "left_shoot": bool(release.left_shoot),
            "raw_sep_ok": bool(release.raw_sep_ok),
            "sep_ok": bool(release.sep_ok),

            "release_by_motion": bool(release.release_by_motion),
            "vy": release.vy,
            "vy_thr": release.vy_thr,
        })

    # Arming bookkeeping (useful to interpret rel_y_rise / sep baseline)
    d.update({
        "armed_ball_rel_y": float(armed_ball_rel_y) if armed_ball_rel_y is not None else None,
        "armed_frame": int(armed_frame) if armed_frame is not None else None,
        "armed_dist_bp": float(armed_dist_bp) if armed_dist_bp is not None else None,
    })

    if extra:
        d.update(extra)

    return d


def attach_fsm_counters(dbg: Dict[str, Any], *, fsm) -> Dict[str, Any]:
    """
    Attach low-level FSM counters for debugging.

    These attributes are internal implementation details of AttemptFSM,
    so we access them defensively with getattr().
    """
    dbg["release_streak"] = int(getattr(fsm, "_release_streak", 0))
    dbg["release_debounce"] = int(getattr(fsm, "debounce", 0))
    return dbg
