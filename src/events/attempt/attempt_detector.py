from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

from src.events.shoot_signal import ShootSignalTracker
from src.events.ball_point import BallPointResolver
from src.events.attempt_fsm import AttemptFSM

from src.events.attempt.attempt_context import (
    BBox,
    build_context,
    pick_best,
    det_bbox,
    det_center,
)

from src.events.attempt.attempt_gates import (
    gate_ball_close_person,
    gate_ball_too_low_before_armed,
    gate_tracker_match_if_low,
    TrackerMatchParams,
)

from src.events.attempt.attempt_release import compute_release, ReleaseInfo
from src.events.attempt.attempt_debug import make_last_debug, attach_fsm_counters


@dataclass
class AttemptEvent:
    frame_idx: int
    ball_cx: float
    ball_cy: float
    rim_cx: float
    rim_cy: float
    distance_px: float
    details: str = ""


class AttemptDetector:
    """
    Attempt detector orchestrator (FSM + gates + signals).
    All "heavy" logic is split across:
      - attempt_context.py
      - attempt_gates.py
      - attempt_release.py
      - attempt_debug.py

    Patches intégrés :
      - shoot bbox memory en ARMED
      - person bbox memory en ARMED (anti trous 1-2 frames)
      - ARMED peut déclencher un release via sep_ok même si shoot bbox manque (anti FN)
      - anti faux positifs conservés (rel_y_rise gate, tracker gate seulement quand balle basse, etc.)

    Anti-faux positifs (NEW) :
      - ne jamais stocker _armed_dist_bp si invalide (sentinelle 1e9, inf, None)
      - (optionnel mais recommandé) en LOCKED: pas de release (clock FSM only)

    Debug patch :
      - “release_started_*” : snapshot du moment où release_signal devient True en ARMED
        (raison + métriques), pour éviter l’illusion “tout est à False au moment du trigger”.
    """

    def __init__(
        self,
        enter_radius_px: float = 85.0,

        # time memory
        rim_recent_frames: int = 15,
        ball_recent_frames: int = 25,  # kept for compat, not used here
        shoot_conf_min: float = 0.18,

        # arming / release
        shoot_arm_window: int = 25,
        release_debounce_frames: int = 2,
        shot_window_frames: int = 20,

        # gates
        person_in_shoot_min_cover: float = 0.55,
        person_in_shoot_pad_px: float = 20.0,
        ball_in_shoot_pad_px: float = 20.0,
        ball_person_max_dist_px: float = 95.0,
        release_sep_increase_px: float = 35.0,

        # ball filter
        enable_ball_size_filter: bool = True,
        ball_area_min_px2: float = 180.0,
        ball_area_max_px2: float = 12000.0,

        # rearm lock
        require_ball_below_rim_to_rearm: bool = True,
        below_margin_px: float = 45.0,
        below_confirm_frames: int = 3,

        # shoot memory
        shoot_memory_frames: int = 8,

        # local memories
        shoot_bbox_recent_frames: int = 6,
        person_bbox_recent_frames: int = 6,

        # rim scaling
        enable_rim_scaling: bool = False,
        rim_ref_width_px: float | None = None,
        rim_ref_w: float | None = None,
        rim_scale_min: float = 0.6,
        rim_scale_max: float = 1.8,

        # compat main.py (accepted but not used inside BallPointResolver here)
        ball_point_memory_frames: int = 6,
        allow_oversize_ball_when_armed: bool = False,
        oversize_ball_conf_min: float = 0.20,
        oversize_ball_dist_boost: float = 1.35,

        # "ball too low" gate
        ball_too_low_rel_y: float = 0.50,

        # ghost ball gate
        enable_tracker_ball_match_gate: bool = True,
        tracker_match_max_dx_px: float = 120.0,
        tracker_match_max_dy_px: float = 120.0,
        tracker_gate_min_ball_rel_y: float = 0.62,

        # arm / release patch
        streak_arm_max_ball_rel_y: float = 0.62,
        min_rel_y_rise_for_sep: float = 0.08,

        # OPTIONAL: motion release (default OFF to avoid new false positives)
        enable_motion_release: bool = False,
        motion_release_vy_thr: float = 2.2,
        motion_release_max_ball_rel_y: float = 0.55,

        debug: bool = False,
    ):
        # compat rim_ref_w
        if rim_ref_width_px is None and rim_ref_w is not None:
            rim_ref_width_px = rim_ref_w
        if rim_ref_width_px is None:
            rim_ref_width_px = 140.0

        self.enter_radius_px = float(enter_radius_px)

        self.rim_recent_frames = int(rim_recent_frames)
        self.ball_recent_frames = int(ball_recent_frames)

        self.shoot_conf_min = float(shoot_conf_min)
        self.shoot_arm_window = int(shoot_arm_window)

        self.person_in_shoot_min_cover = float(person_in_shoot_min_cover)
        self.person_in_shoot_pad_px = float(person_in_shoot_pad_px)
        self.ball_in_shoot_pad_px = float(ball_in_shoot_pad_px)

        self.ball_person_max_dist_px = float(ball_person_max_dist_px)
        self.release_sep_increase_px = float(release_sep_increase_px)

        self.enable_ball_size_filter = bool(enable_ball_size_filter)
        self.ball_area_min_px2 = float(ball_area_min_px2)
        self.ball_area_max_px2 = float(ball_area_max_px2)

        self.require_ball_below_rim_to_rearm = bool(require_ball_below_rim_to_rearm)
        self.below_margin_px = float(below_margin_px)
        self.below_confirm_frames = int(below_confirm_frames)

        self.shoot_bbox_recent_frames = int(shoot_bbox_recent_frames)
        self.person_bbox_recent_frames = int(person_bbox_recent_frames)

        self.enable_rim_scaling = bool(enable_rim_scaling)
        self.rim_ref_width_px = float(rim_ref_width_px)
        self.rim_scale_min = float(rim_scale_min)
        self.rim_scale_max = float(rim_scale_max)

        self.ball_too_low_rel_y = float(ball_too_low_rel_y)

        self.enable_tracker_ball_match_gate = bool(enable_tracker_ball_match_gate)
        self.tracker_params = TrackerMatchParams(
            tracker_match_max_dx_px=float(tracker_match_max_dx_px),
            tracker_match_max_dy_px=float(tracker_match_max_dy_px),
            tracker_gate_min_ball_rel_y=float(tracker_gate_min_ball_rel_y),
        )

        self.streak_arm_max_ball_rel_y = float(streak_arm_max_ball_rel_y)
        self.min_rel_y_rise_for_sep = float(min_rel_y_rise_for_sep)

        self.enable_motion_release = bool(enable_motion_release)
        self.motion_release_vy_thr = float(motion_release_vy_thr)
        self.motion_release_max_ball_rel_y = float(motion_release_max_ball_rel_y)

        self.debug = bool(debug)

        self.shoot = ShootSignalTracker(self.shoot_conf_min, memory_frames=int(shoot_memory_frames))

        self.ball = BallPointResolver(
            memory_frames=int(ball_point_memory_frames),
            enable_size_filter=self.enable_ball_size_filter,
            area_min_px2=self.ball_area_min_px2,
            area_max_px2=self.ball_area_max_px2,
        )

        self.fsm = AttemptFSM(
            debounce=int(release_debounce_frames),
            shot_window=int(shot_window_frames),
        )

        # rim memory
        self.last_rim_xy: Optional[Tuple[float, float]] = None
        self.last_rim_bbox: Optional[BBox] = None
        self.last_rim_frame: int = -10**9

        # local shoot bbox memory for ARMED continuity
        self._last_shoot_bbox: Optional[BBox] = None
        self._last_shoot_frame: int = -10**9

        # local person bbox memory for ARMED continuity
        self._last_person_bbox: Optional[BBox] = None
        self._last_person_frame: int = -10**9

        # rearm lock
        self._waiting_below = False
        self._below_streak = 0

        # arming bookkeeping
        self._armed_frame: int = -10**9
        self._armed_dist_bp: Optional[float] = None
        self._armed_ball_rel_y: Optional[float] = None
        self._armed_via_rise: bool = False

        # release debug bookkeeping
        self._release_started_frame: int = -10**9
        self._release_started_reason: Optional[str] = None
        self._release_started_snapshot: Optional[Dict[str, Any]] = None

        # debug
        self.last_debug: Dict[str, Any] = {}

    # -----------------
    # utilities
    # -----------------
    def _scale(self) -> float:
        if (not self.enable_rim_scaling) or (self.last_rim_bbox is None):
            return 1.0
        x1, _, x2, _ = self.last_rim_bbox
        w = max(1.0, float(x2 - x1))
        s = w / max(1.0, self.rim_ref_width_px)
        return max(self.rim_scale_min, min(self.rim_scale_max, s))

    def _clock_fsm_no_release(self, frame_idx: int):
        self.fsm.update(frame_idx, arm_signal=False, release_signal=False)

    def _choose_shoot_bbox(self, frame_idx: int, shoot_info: Dict[str, Any]) -> Optional[BBox]:
        shoot_bbox = shoot_info.get("shoot_bbox")
        if shoot_bbox is not None:
            self._last_shoot_bbox = shoot_bbox
            self._last_shoot_frame = int(frame_idx)
            return shoot_bbox

        if self._last_shoot_bbox is not None and (frame_idx - self._last_shoot_frame) <= self.shoot_bbox_recent_frames:
            return self._last_shoot_bbox

        return None

    def _choose_person_bbox(self, frame_idx: int, person_det: Optional[Dict[str, Any]]) -> Optional[BBox]:
        if person_det is not None:
            pb = det_bbox(person_det)
            self._last_person_bbox = pb
            self._last_person_frame = int(frame_idx)
            return pb

        if self._last_person_bbox is not None and (frame_idx - self._last_person_frame) <= self.person_bbox_recent_frames:
            return self._last_person_bbox

        return None

    def _reset_release_started(self):
        self._release_started_frame = -10**9
        self._release_started_reason = None
        self._release_started_snapshot = None

    def _maybe_mark_release_started(self, frame_idx: int, *, shoot_info: Dict[str, Any], ctx, release: ReleaseInfo):
        if self.fsm.state != AttemptFSM.ARMED:
            return
        if not bool(getattr(release, "release_signal", False)):
            return
        if self._release_started_frame > -10**8:
            return

        reason = "unknown"
        if bool(getattr(release, "left_shoot", False)):
            reason = "left_shoot"
        elif bool(getattr(release, "sep_ok", False)):
            reason = "sep_ok"
        elif bool(getattr(release, "motion_ok", False)):
            reason = "motion"
        elif bool(getattr(release, "motion_release", False)):
            reason = "motion"

        self._release_started_frame = int(frame_idx)
        self._release_started_reason = reason

        snap: Dict[str, Any] = {
            "shoot_now": bool(shoot_info.get("shoot_now", False)),
            "shoot_rise": bool(shoot_info.get("shoot_rise", False)),
            "shoot_streak": int(shoot_info.get("shoot_streak", 0)),
            "shoot_conf": float(shoot_info.get("shoot_conf", 0.0)),
            "shoot_from_memory": bool(shoot_info.get("shoot_from_memory", False)),
            "ball_in_shoot": getattr(ctx, "ball_in_shoot", None),
            "ball_rel_y": getattr(ctx, "ball_rel_y", None),
            "rel_y_rise": getattr(ctx, "rel_y_rise", None),
            "d_ball_person": getattr(ctx, "d_ball_person", None),
            "sep_thr": getattr(ctx, "sep_thr", None),
            "left_shoot": bool(getattr(release, "left_shoot", False)),
            "sep_ok": bool(getattr(release, "sep_ok", False)),
            "raw_sep_ok": bool(getattr(release, "raw_sep_ok", False)),
        }
        self._release_started_snapshot = snap

    def _is_valid_dist(self, d: Optional[float]) -> bool:
        if d is None:
            return False
        try:
            d = float(d)
        except (TypeError, ValueError):
            return False
        # 1e8+ => sentinelle/invalide (ex: 1e9)
        return math.isfinite(d) and (0.0 <= d < 1e8)

    def _safe_dist(self, d: Optional[float]) -> Optional[float]:
        return float(d) if self._is_valid_dist(d) else None

    # -----------------
    # main update
    # -----------------
    def update(self, frame_idx: int, detections: List[Dict[str, Any]], ball_state) -> Optional[AttemptEvent]:
        # rim memory
        rim_det = pick_best(detections, "rim")
        if rim_det is not None:
            self.last_rim_xy = det_center(rim_det)
            self.last_rim_bbox = det_bbox(rim_det)
            self.last_rim_frame = int(frame_idx)

        # below-rim lock
        if self._waiting_below and self.last_rim_xy is not None:
            _, rim_cy = self.last_rim_xy
            below_now = False
            if ball_state is not None:
                below_now = float(ball_state.cy) >= (rim_cy + self.below_margin_px)

            self._below_streak = (self._below_streak + 1) if below_now else 0
            if self._below_streak < self.below_confirm_frames:
                dbg = make_last_debug(
                    frame_idx=frame_idx,
                    gate_reason="blocked_waiting_below_rim",
                    fsm_state=self.fsm.state,
                    extra={"below_streak": int(self._below_streak)},
                )
                self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
                return None

            self._waiting_below = False
            self._below_streak = 0

        shoot_info = self.shoot.update(frame_idx, detections)
        shoot_now = bool(shoot_info.get("shoot_now", False))
        shoot_bbox = self._choose_shoot_bbox(frame_idx, shoot_info)

        person_det = pick_best(detections, "person")
        ball_det = pick_best(detections, "ball")
        person_bbox = self._choose_person_bbox(frame_idx, person_det)

        # (RECO) LOCKED: ne pas recalculer release / gates -> clock only
        if self.fsm.state == AttemptFSM.LOCKED:
            self._clock_fsm_no_release(frame_idx)
            dbg = make_last_debug(
                frame_idx=frame_idx,
                gate_reason="LOCKED",
                fsm_state=self.fsm.state,
                shoot_info=shoot_info,
            )
            self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
            return None

        # IDLE strict
        if self.fsm.state == AttemptFSM.IDLE:
            self._reset_release_started()

            if (not shoot_now) or (person_bbox is None) or (shoot_bbox is None):
                dbg = make_last_debug(
                    frame_idx=frame_idx,
                    gate_reason="missing_shoot_or_person" if ((not shoot_now) or (person_bbox is None)) else "missing_shoot_bbox",
                    fsm_state=self.fsm.state,
                    shoot_info=shoot_info,
                    extra={
                        "shoot_bbox_mem_age": int(frame_idx - self._last_shoot_frame) if self._last_shoot_frame > -10**8 else None,
                        "person_bbox_mem_age": int(frame_idx - self._last_person_frame) if self._last_person_frame > -10**8 else None,
                    },
                )
                self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
                return None

        # ARMED: if shoot bbox missing -> allow sep_ok-only continuation (FN fix)
        if self.fsm.state == AttemptFSM.ARMED and shoot_bbox is None:
            if person_bbox is None:
                self._clock_fsm_no_release(frame_idx)
                dbg = make_last_debug(
                    frame_idx=frame_idx,
                    gate_reason="armed_missing_person",
                    fsm_state=self.fsm.state,
                    shoot_info=shoot_info,
                    extra={
                        "shoot_bbox_mem_age": int(frame_idx - self._last_shoot_frame) if self._last_shoot_frame > -10**8 else None,
                        "person_bbox_mem_age": int(frame_idx - self._last_person_frame) if self._last_person_frame > -10**8 else None,
                    },
                )
                self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
                return None

            ball_point_tmp = self.ball.update(frame_idx, ball_state, ball_det, person_bbox)
            if ball_point_tmp is None:
                self._clock_fsm_no_release(frame_idx)
                dbg = make_last_debug(
                    frame_idx=frame_idx,
                    gate_reason="armed_missing_shoot_bbox",
                    fsm_state=self.fsm.state,
                    shoot_info=shoot_info,
                    extra={
                        "shoot_bbox_mem_age": int(frame_idx - self._last_shoot_frame) if self._last_shoot_frame > -10**8 else None,
                        "person_bbox": person_bbox,
                    },
                )
                self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
                return None

            bx, by, src = float(ball_point_tmp[0]), float(ball_point_tmp[1]), str(ball_point_tmp[2])

            from src.events.spatial_gates import dist_point_to_bbox
            d_bp = float(dist_point_to_bbox(bx, by, person_bbox))

            x1, y1, x2, y2 = person_bbox
            h = max(1.0, (y2 - y1))
            ball_rel_y = float((by - y1) / h)

            rel_y_rise = 0.0
            if self._armed_ball_rel_y is not None:
                rel_y_rise = float(self._armed_ball_rel_y - ball_rel_y)

            s = self._scale()
            sep_thr = float(self.release_sep_increase_px * s)

            d_bp_safe = self._safe_dist(d_bp)
            armed_bp_safe = self._safe_dist(self._armed_dist_bp)

            raw_sep_ok = False
            if (d_bp_safe is not None) and (armed_bp_safe is not None):
                raw_sep_ok = (d_bp_safe - armed_bp_safe) >= sep_thr

            sep_ok = bool(raw_sep_ok and (rel_y_rise >= self.min_rel_y_rise_for_sep))

            if sep_ok and self._release_started_frame <= -10**8:
                self._release_started_frame = int(frame_idx)
                self._release_started_reason = "sep_ok"
                self._release_started_snapshot = {
                    "mode": "ARMED_NO_SHOOT_BBOX",
                    "ball_src": src,
                    "d_ball_person": float(d_bp_safe) if d_bp_safe is not None else None,
                    "armed_dist_bp": float(armed_bp_safe) if armed_bp_safe is not None else None,
                    "sep_thr": float(sep_thr),
                    "raw_sep_ok": bool(raw_sep_ok),
                    "sep_ok": bool(sep_ok),
                    "ball_rel_y": float(ball_rel_y),
                    "rel_y_rise": float(rel_y_rise),
                    "shoot_now": bool(shoot_now),
                    "shoot_streak": int(shoot_info.get("shoot_streak", 0)),
                    "shoot_conf": float(shoot_info.get("shoot_conf", 0.0)),
                }

            evt2 = self.fsm.update(frame_idx, arm_signal=False, release_signal=sep_ok)

            if evt2 is not None and getattr(evt2, "name", None) == "ATTEMPT":
                if self.require_ball_below_rim_to_rearm:
                    self._waiting_below = True
                    self._below_streak = 0

                if self.last_rim_xy is None or (frame_idx - self.last_rim_frame) > self.rim_recent_frames:
                    dbg = make_last_debug(
                        frame_idx=frame_idx,
                        gate_reason="attempt_but_rim_stale",
                        fsm_state=self.fsm.state,
                        shoot_info=shoot_info,
                        extra={"armed_missing_shoot_bbox_release": True},
                    )
                    self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
                    return None

                rim_cx, rim_cy = self.last_rim_xy
                dist_br = math.hypot(bx - rim_cx, by - rim_cy)

                dbg = make_last_debug(
                    frame_idx=frame_idx,
                    gate_reason="ATTEMPT_TRIGGERED",
                    fsm_state=self.fsm.state,
                    shoot_info=shoot_info,
                    extra={
                        "ball_xy": (bx, by),
                        "ball_src": src,
                        "person_bbox": person_bbox,
                        "sep_ok": bool(sep_ok),
                        "raw_sep_ok": bool(raw_sep_ok),
                        "rel_y_rise": float(rel_y_rise),
                        "ball_rel_y": float(ball_rel_y),
                        "scale": float(s),
                        "sep_thr": float(sep_thr),
                        "armed_missing_shoot_bbox_release": True,
                        "release_started_frame": int(self._release_started_frame) if self._release_started_frame > -10**8 else None,
                        "release_started_reason": self._release_started_reason,
                        "release_started_age": int(frame_idx - self._release_started_frame) if self._release_started_frame > -10**8 else None,
                        "release_started_snapshot": self._release_started_snapshot,
                    },
                )
                self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)

                return AttemptEvent(
                    frame_idx=int(frame_idx),
                    ball_cx=float(bx),
                    ball_cy=float(by),
                    rim_cx=float(rim_cx),
                    rim_cy=float(rim_cy),
                    distance_px=float(dist_br),
                    details=f"release(ARMED_NO_SHOOT_BBOX, ball_src={src}, sep={sep_ok}, rel_y_rise={rel_y_rise:.2f}, scale={s:.2f})",
                )

            dbg = make_last_debug(
                frame_idx=frame_idx,
                gate_reason="armed_missing_shoot_bbox",
                fsm_state=self.fsm.state,
                shoot_info=shoot_info,
                extra={
                    "ball_xy": (bx, by),
                    "ball_src": src,
                    "person_bbox": person_bbox,
                    "sep_ok": bool(sep_ok),
                    "raw_sep_ok": bool(raw_sep_ok),
                    "rel_y_rise": float(rel_y_rise),
                    "ball_rel_y": float(ball_rel_y),
                    "scale": float(s),
                    "sep_thr": float(sep_thr),
                    "armed_dist_bp": self._armed_dist_bp,
                    "d_ball_person": float(d_bp) if d_bp is not None else None,
                    "release_started_frame": int(self._release_started_frame) if self._release_started_frame > -10**8 else None,
                    "release_started_reason": self._release_started_reason,
                    "release_started_age": int(frame_idx - self._release_started_frame) if self._release_started_frame > -10**8 else None,
                },
            )
            self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
            return None

        # From here, we need shoot bbox + person bbox
        if person_bbox is None or shoot_bbox is None:
            if self.fsm.state == AttemptFSM.ARMED:
                self._clock_fsm_no_release(frame_idx)
            dbg = make_last_debug(
                frame_idx=frame_idx,
                gate_reason="missing_person_or_shoot_bbox",
                fsm_state=self.fsm.state,
                shoot_info=shoot_info,
                extra={
                    "shoot_bbox_mem_age": int(frame_idx - self._last_shoot_frame) if self._last_shoot_frame > -10**8 else None,
                    "person_bbox_mem_age": int(frame_idx - self._last_person_frame) if self._last_person_frame > -10**8 else None,
                },
            )
            self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
            return None

        # ball point resolution
        ball_point = self.ball.update(frame_idx, ball_state, ball_det, person_bbox)
        if ball_point is None:
            if self.fsm.state == AttemptFSM.ARMED:
                self._clock_fsm_no_release(frame_idx)
            dbg = make_last_debug(
                frame_idx=frame_idx,
                gate_reason="no_ball_point",
                fsm_state=self.fsm.state,
                shoot_info=shoot_info,
                extra={"person_bbox": person_bbox, "shoot_bbox": shoot_bbox},
            )
            self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
            return None

        s = self._scale()

        ctx = build_context(
            frame_idx=frame_idx,
            shoot_info=shoot_info,
            shoot_bbox=shoot_bbox,
            person_bbox=person_bbox,
            ball_point=ball_point,
            scale=s,
            person_in_shoot_min_cover=self.person_in_shoot_min_cover,
            person_in_shoot_pad_px=self.person_in_shoot_pad_px,
            ball_in_shoot_pad_px=self.ball_in_shoot_pad_px,
            ball_person_max_dist_px=self.ball_person_max_dist_px,
            release_sep_increase_px=self.release_sep_increase_px,
            armed_ball_rel_y=self._armed_ball_rel_y,
        )

        # Gate: person in shoot
        if not ctx.person_in_shoot:
            if self.fsm.state == AttemptFSM.ARMED:
                self._clock_fsm_no_release(frame_idx)
                dbg = make_last_debug(
                    frame_idx=frame_idx,
                    gate_reason="armed_fail_person_in_shoot",
                    fsm_state=self.fsm.state,
                    shoot_info=shoot_info,
                    ctx=ctx,
                    armed_ball_rel_y=self._armed_ball_rel_y,
                    armed_frame=self._armed_frame,
                    armed_dist_bp=self._armed_dist_bp,
                    extra={"cover": float(ctx.cover)},
                )
                self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
                return None

            dbg = make_last_debug(
                frame_idx=frame_idx,
                gate_reason="fail_person_in_shoot",
                fsm_state=self.fsm.state,
                shoot_info=shoot_info,
                ctx=ctx,
                extra={"cover": float(ctx.cover)},
            )
            self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
            return None

        # Gate: ghost ball (only when ball low)
        if self.enable_tracker_ball_match_gate:
            ok, dx, dy = gate_tracker_match_if_low(ctx, ball_state=ball_state, params=self.tracker_params)
            if not ok:
                if self.fsm.state == AttemptFSM.ARMED:
                    self._clock_fsm_no_release(frame_idx)
                dbg = make_last_debug(
                    frame_idx=frame_idx,
                    gate_reason="ball_not_from_tracker",
                    fsm_state=self.fsm.state,
                    shoot_info=shoot_info,
                    ctx=ctx,
                    armed_ball_rel_y=self._armed_ball_rel_y,
                    armed_frame=self._armed_frame,
                    armed_dist_bp=self._armed_dist_bp,
                    extra={
                        "dx": dx,
                        "dy": dy,
                        "max_dx": float(self.tracker_params.tracker_match_max_dx_px),
                        "max_dy": float(self.tracker_params.tracker_match_max_dy_px),
                        "min_rel_y": float(self.tracker_params.tracker_gate_min_ball_rel_y),
                    },
                )
                self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
                return None

        # Gate: ball close to person unless ball_in_shoot
        if not gate_ball_close_person(ctx):
            if self.fsm.state == AttemptFSM.ARMED:
                self._clock_fsm_no_release(frame_idx)
            dbg = make_last_debug(
                frame_idx=frame_idx,
                gate_reason="fail_ball_close_person",
                fsm_state=self.fsm.state,
                shoot_info=shoot_info,
                ctx=ctx,
            )
            self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
            return None

        # Gate: ball too low (pre-armed only)
        if not gate_ball_too_low_before_armed(ctx, ball_too_low_rel_y=self.ball_too_low_rel_y, fsm_state=self.fsm.state):
            dbg = make_last_debug(
                frame_idx=frame_idx,
                gate_reason="ball_too_low_for_shot",
                fsm_state=self.fsm.state,
                shoot_info=shoot_info,
                ctx=ctx,
                extra={"thr": float(self.ball_too_low_rel_y)},
            )
            self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
            return None

        # ARM SIGNAL (patched)
        shoot_rise = bool(shoot_info.get("shoot_rise", False))
        shoot_streak = int(shoot_info.get("shoot_streak", 0))
        shoot_from_memory = bool(shoot_info.get("shoot_from_memory", False))

        streak_arm_ok = bool(
            (shoot_streak >= 2)
            and ctx.ball_in_shoot
            and (ctx.ball_rel_y <= self.streak_arm_max_ball_rel_y)
        )
        rise_arm_ok = bool(shoot_rise and ctx.ball_in_shoot and (not shoot_from_memory))
        arm_signal = bool(rise_arm_ok or streak_arm_ok)

        if (self.fsm.state == AttemptFSM.IDLE) and (not arm_signal):
            dbg = make_last_debug(
                frame_idx=frame_idx,
                gate_reason="not_armed_waiting_shoot_rise",
                fsm_state=self.fsm.state,
                shoot_info=shoot_info,
                ctx=ctx,
                extra={
                    "rise_arm_ok": bool(rise_arm_ok),
                    "streak_arm_ok": bool(streak_arm_ok),
                },
            )
            self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
            return None

        # arming timeout
        if self.fsm.state == AttemptFSM.ARMED and self._armed_frame > -10**8:
            if (frame_idx - self._armed_frame) > self.shoot_arm_window:
                self.fsm.reset()
                self._armed_frame = -10**9
                self._armed_dist_bp = None
                self._armed_ball_rel_y = None
                self._armed_via_rise = False
                self._reset_release_started()

                dbg = make_last_debug(
                    frame_idx=frame_idx,
                    gate_reason="arming_window_expired",
                    fsm_state=self.fsm.state,
                    shoot_info=shoot_info,
                    ctx=ctx,
                )
                self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
                return None

                # RELEASE
        armed_clean = bool(self._armed_via_rise or streak_arm_ok)

        # ---------------------------------------------------------
        # PATCH B (local, propre):
        # si d_ball_person (ctx) OU baseline armed_dist_bp sont invalides,
        # on interdit *sep_ok* => release ne peut venir que de left_shoot (ou motion si activé)
        # ---------------------------------------------------------
        armed_bp_valid = self._is_valid_dist(self._armed_dist_bp)
        ctx_bp_valid = self._is_valid_dist(getattr(ctx, "d_ball_person", None))
        allow_sep = bool(armed_bp_valid and ctx_bp_valid)

        # On calcule release normalement (pour garder left_shoot/motion diagnostics),
        # mais on NE FAIT PAS CONFIANCE à release.release_signal si allow_sep=False.
        release: ReleaseInfo = compute_release(
            ctx,
            fsm_state=self.fsm.state,
            armed_dist_bp=(self._armed_dist_bp if allow_sep else None),
            min_rel_y_rise_for_sep=self.min_rel_y_rise_for_sep,
            enable_motion_release=self.enable_motion_release,
            motion_release_vy_thr=self.motion_release_vy_thr,
            motion_release_max_ball_rel_y=self.motion_release_max_ball_rel_y,
            armed_clean=armed_clean,
            ball_state=ball_state,
        )

        # release_signal SAFE:
        # - si allow_sep=True : on respecte release.release_signal (left_shoot OR sep_ok OR motion...)
        # - si allow_sep=False: on ignore totalement sep_ok => on ne garde que left_shoot/motion
        if allow_sep:
            release_signal_safe = bool(getattr(release, "release_signal", False))
        else:
            release_signal_safe = bool(getattr(release, "left_shoot", False)) or bool(getattr(release, "motion_ok", False)) or bool(
                getattr(release, "motion_release", False)
            )

        # debug snapshot (reste OK, mais attention: si allow_sep=False on n'autorisera pas sep en FSM)
        self._maybe_mark_release_started(frame_idx, shoot_info=shoot_info, ctx=ctx, release=release)

        prev_state = self.fsm.state
        evt = self.fsm.update(frame_idx, arm_signal=arm_signal, release_signal=release_signal_safe)


        # baseline distance + rel_y at arming (IDLE -> ARMED)
        if prev_state == AttemptFSM.IDLE and self.fsm.state == AttemptFSM.ARMED:
            self._armed_frame = int(frame_idx)

            # >>> PATCH A: ne jamais stocker une distance invalide
            armed_bp = self._safe_dist(getattr(ctx, "d_ball_person", None))
            self._armed_dist_bp = armed_bp

            self._armed_ball_rel_y = float(ctx.ball_rel_y)
            self._armed_via_rise = bool(rise_arm_ok)
            self._reset_release_started()

        if prev_state != AttemptFSM.IDLE and self.fsm.state == AttemptFSM.IDLE:
            self._reset_release_started()

        # ATTEMPT triggered
        if evt is not None and getattr(evt, "name", None) == "ATTEMPT":
            if self.require_ball_below_rim_to_rearm:
                self._waiting_below = True
                self._below_streak = 0

            if self.last_rim_xy is None or (frame_idx - self.last_rim_frame) > self.rim_recent_frames:
                dbg = make_last_debug(
                    frame_idx=frame_idx,
                    gate_reason="attempt_but_rim_stale",
                    fsm_state=self.fsm.state,
                    shoot_info=shoot_info,
                    ctx=ctx,
                    release=release,
                    armed_ball_rel_y=self._armed_ball_rel_y,
                    armed_frame=self._armed_frame,
                    armed_dist_bp=self._armed_dist_bp,
                    extra={
                        "release_started_frame": int(self._release_started_frame) if self._release_started_frame > -10**8 else None,
                        "release_started_reason": self._release_started_reason,
                        "release_started_age": int(frame_idx - self._release_started_frame) if self._release_started_frame > -10**8 else None,
                        "release_started_snapshot": self._release_started_snapshot,
                        "armed_dist_bp_valid": self._is_valid_dist(self._armed_dist_bp),
                    },
                )
                self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
                return None

            rim_cx, rim_cy = self.last_rim_xy
            bx, by = ctx.ball_xy
            dist_br = math.hypot(bx - rim_cx, by - rim_cy)

            dbg = make_last_debug(
                frame_idx=frame_idx,
                gate_reason="ATTEMPT_TRIGGERED",
                fsm_state=self.fsm.state,
                shoot_info=shoot_info,
                ctx=ctx,
                release=release,
                armed_ball_rel_y=self._armed_ball_rel_y,
                armed_frame=self._armed_frame,
                armed_dist_bp=self._armed_dist_bp,
                extra={
                    "rise_arm_ok": bool(rise_arm_ok),
                    "streak_arm_ok": bool(streak_arm_ok),
                    "sep_thr": float(ctx.sep_thr),
                    "shoot_bbox_mem_age": int(frame_idx - self._last_shoot_frame) if self._last_shoot_frame > -10**8 else None,
                    "person_bbox_mem_age": int(frame_idx - self._last_person_frame) if self._last_person_frame > -10**8 else None,
                    "release_started_frame": int(self._release_started_frame) if self._release_started_frame > -10**8 else None,
                    "release_started_reason": self._release_started_reason,
                    "release_started_age": int(frame_idx - self._release_started_frame) if self._release_started_frame > -10**8 else None,
                    "release_started_snapshot": self._release_started_snapshot,
                    "armed_dist_bp_valid": self._is_valid_dist(self._armed_dist_bp),
                },
            )
            self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)

            return AttemptEvent(
                frame_idx=int(frame_idx),
                ball_cx=float(bx),
                ball_cy=float(by),
                rim_cx=float(rim_cx),
                rim_cy=float(rim_cy),
                distance_px=float(dist_br),
                details=(
                    f"release(ball_src={ctx.ball_src}, left={release.left_shoot}, sep={release.sep_ok}, "
                    f"rel_y_rise={ctx.rel_y_rise:.2f}, scale={ctx.scale:.2f}, shoot_mem={shoot_from_memory}, "
                    f"release_started={self._release_started_reason}@{self._release_started_frame})"
                ),
            )

        dbg = make_last_debug(
            frame_idx=frame_idx,
            gate_reason=("armed_waiting_release" if self.fsm.state == AttemptFSM.ARMED else str(self.fsm.state)),
            fsm_state=self.fsm.state,
            shoot_info=shoot_info,
            ctx=ctx,
            release=release,
            armed_ball_rel_y=self._armed_ball_rel_y,
            armed_frame=self._armed_frame,
            armed_dist_bp=self._armed_dist_bp,
            extra={
                "rise_arm_ok": bool(rise_arm_ok),
                "streak_arm_ok": bool(streak_arm_ok),
                "sep_thr": float(ctx.sep_thr),
                "shoot_bbox_mem_age": int(frame_idx - self._last_shoot_frame) if self._last_shoot_frame > -10**8 else None,
                "person_bbox_mem_age": int(frame_idx - self._last_person_frame) if self._last_person_frame > -10**8 else None,
                "release_started_frame": int(self._release_started_frame) if self._release_started_frame > -10**8 else None,
                "release_started_reason": self._release_started_reason,
                "release_started_age": int(frame_idx - self._release_started_frame) if self._release_started_frame > -10**8 else None,
                "armed_dist_bp_valid": self._is_valid_dist(self._armed_dist_bp),
                "allow_sep": allow_sep,
                "armed_bp_valid": armed_bp_valid,
                "ctx_bp_valid": ctx_bp_valid,
                "release_signal_safe": bool(release_signal_safe),
            },
        )
        self.last_debug = attach_fsm_counters(dbg, fsm=self.fsm)
        return None
