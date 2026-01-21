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
    """
    Shot attempt event emitted by AttemptDetector.

    This is the bridge between:
    - Attempt detection (temporal logic + gates), and
    - Outcome classification (MadeDetector), which needs rim + ball geometry.
    """
    frame_idx: int
    ball_cx: float
    ball_cy: float
    rim_cx: float
    rim_cy: float
    distance_px: float
    details: str = ""


class AttemptDetector:
    """
    Orchestrator for shot attempt detection.

    Responsibility (engineering separation):
    - Consume frame-level detections (YOLO) and ball_state (tracker).
    - Apply *gates* (spatial plausibility constraints) to reduce false positives.
    - Run a small FSM to convert continuous signals into a discrete ATTEMPT event.

    Internal structure:
    - ShootSignalTracker: turn YOLO "shoot" into temporal signals (+ short bbox memory).
    - BallPointResolver: choose one ball point per frame (tracker / yolo / memory).
    - build_context(): compute geometric features at this frame (ball-in-shoot, overlap, distances, scaling).
    - attempt_gates.py: pure boolean gating functions.
    - attempt_release.py: compute release cues (left_shoot / sep_ok / optional motion).
    - AttemptFSM: debounced event trigger + lock window.

    Design assumptions:
    - One relevant shooter and one relevant ball at a time (single-attempt reasoning).
    - Rim is approximately static over a short window; we keep a recent rim memory.
    - YOLO detections can drop for a few frames (especially person/shoot at long distance);
      ARMED state contains controlled fallbacks to remain robust to short holes.
    """

    def __init__(
        self,
        enter_radius_px: float = 85.0,

        # Rim memory horizon (frames)
        rim_recent_frames: int = 15,
        ball_recent_frames: int = 25,  # kept for compatibility; not used directly here
        shoot_conf_min: float = 0.18,

        # FSM timing
        shoot_arm_window: int = 25,
        release_debounce_frames: int = 2,
        shot_window_frames: int = 20,

        # Spatial gates
        person_in_shoot_min_cover: float = 0.55,
        person_in_shoot_pad_px: float = 20.0,
        ball_in_shoot_pad_px: float = 20.0,
        ball_person_max_dist_px: float = 95.0,
        release_sep_increase_px: float = 35.0,

        # Ball validity (anti false positives)
        enable_ball_size_filter: bool = True,
        ball_area_min_px2: float = 180.0,
        ball_area_max_px2: float = 12000.0,

        # Re-arming lock
        require_ball_below_rim_to_rearm: bool = True,
        below_margin_px: float = 45.0,
        below_confirm_frames: int = 3,

        # Shoot bbox memory inside ShootSignalTracker
        shoot_memory_frames: int = 8,

        # Local memory inside AttemptDetector (ARMED continuity)
        shoot_bbox_recent_frames: int = 6,
        person_bbox_recent_frames: int = 6,

        # Optional rim scaling (normalize pixel thresholds by rim size)
        enable_rim_scaling: bool = False,
        rim_ref_width_px: float | None = None,
        rim_ref_w: float | None = None,  # backwards-compatible alias
        rim_scale_min: float = 0.6,
        rim_scale_max: float = 1.8,

        # Compatibility (BallPointResolver memory)
        ball_point_memory_frames: int = 6,
        allow_oversize_ball_when_armed: bool = False,  # accepted for compatibility; not used here
        oversize_ball_conf_min: float = 0.20,
        oversize_ball_dist_boost: float = 1.35,

        # Pre-armed "ball too low" heuristic
        ball_too_low_rel_y: float = 0.50,

        # Ghost-ball gate (compare YOLO vs tracker when ball is low near player)
        enable_tracker_ball_match_gate: bool = True,
        tracker_match_max_dx_px: float = 120.0,
        tracker_match_max_dy_px: float = 120.0,
        tracker_gate_min_ball_rel_y: float = 0.62,

        # Arming / release robustness
        streak_arm_max_ball_rel_y: float = 0.62,
        min_rel_y_rise_for_sep: float = 0.08,

        # Optional motion-based release (OFF by default to avoid new false positives)
        enable_motion_release: bool = False,
        motion_release_vy_thr: float = 2.2,
        motion_release_max_ball_rel_y: float = 0.55,

        debug: bool = False,
    ):
        # Backward compatibility: rim_ref_w alias
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

        # Perception-side helpers
        self.shoot = ShootSignalTracker(self.shoot_conf_min, memory_frames=int(shoot_memory_frames))
        self.ball = BallPointResolver(
            memory_frames=int(ball_point_memory_frames),
            enable_size_filter=self.enable_ball_size_filter,
            area_min_px2=self.ball_area_min_px2,
            area_max_px2=self.ball_area_max_px2,
        )

        # Event FSM
        self.fsm = AttemptFSM(
            debounce=int(release_debounce_frames),
            shot_window=int(shot_window_frames),
        )

        # Rim memory (used for AttemptEvent geometry and “rim stale” rejection)
        self.last_rim_xy: Optional[Tuple[float, float]] = None
        self.last_rim_bbox: Optional[BBox] = None
        self.last_rim_frame: int = -10**9

        # Local memories to bridge short dropouts in ARMED state
        self._last_shoot_bbox: Optional[BBox] = None
        self._last_shoot_frame: int = -10**9

        self._last_person_bbox: Optional[BBox] = None
        self._last_person_frame: int = -10**9

        # Rearm lock: enforce ball below rim before allowing a new attempt
        self._waiting_below = False
        self._below_streak = 0

        # Arming baseline (stored at IDLE -> ARMED)
        self._armed_frame: int = -10**9
        self._armed_dist_bp: Optional[float] = None
        self._armed_ball_rel_y: Optional[float] = None
        self._armed_via_rise: bool = False

        # Release debug snapshot: stores the first moment release becomes true
        # so overlays reflect the actual “start” of release, not only the trigger frame.
        self._release_started_frame: int = -10**9
        self._release_started_reason: Optional[str] = None
        self._release_started_snapshot: Optional[Dict[str, Any]] = None

        self.last_debug: Dict[str, Any] = {}

    # -----------------
    # small utilities (pure helpers)
    # -----------------
    def _scale(self) -> float:
        """
        Optional scaling factor derived from rim width.

        Motivation:
        - Pixel thresholds (distance pads, sep thresholds) should adapt to camera distance.
        - When enabled, we scale gate thresholds by (current_rim_width / reference_rim_width).
        """
        if (not self.enable_rim_scaling) or (self.last_rim_bbox is None):
            return 1.0
        x1, _, x2, _ = self.last_rim_bbox
        w = max(1.0, float(x2 - x1))
        s = w / max(1.0, self.rim_ref_width_px)
        return max(self.rim_scale_min, min(self.rim_scale_max, s))

    def _clock_fsm_no_release(self, frame_idx: int):
        """
        Advance the FSM timing without allowing release transitions.

        Used to keep LOCKED state time-based and to avoid accidental triggers
        when required inputs are missing.
        """
        self.fsm.update(frame_idx, arm_signal=False, release_signal=False)

    def _choose_shoot_bbox(self, frame_idx: int, shoot_info: Dict[str, Any]) -> Optional[BBox]:
        """
        Choose a shoot bbox for this frame.

        Policy:
        - Prefer current shoot bbox from ShootSignalTracker.
        - Otherwise allow a short local memory fallback (ARMED continuity),
          because shoot detections can disappear briefly at long distance.
        """
        shoot_bbox = shoot_info.get("shoot_bbox")
        if shoot_bbox is not None:
            self._last_shoot_bbox = shoot_bbox
            self._last_shoot_frame = int(frame_idx)
            return shoot_bbox

        if self._last_shoot_bbox is not None and (frame_idx - self._last_shoot_frame) <= self.shoot_bbox_recent_frames:
            return self._last_shoot_bbox

        return None

    def _choose_person_bbox(self, frame_idx: int, person_det: Optional[Dict[str, Any]]) -> Optional[BBox]:
        """
        Choose a person bbox for this frame with short memory fallback.

        Rationale:
        - Person detection can be missed intermittently.
        - When ARMED, a brief hole should not cancel the attempt logic immediately.
        """
        if person_det is not None:
            pb = det_bbox(person_det)
            self._last_person_bbox = pb
            self._last_person_frame = int(frame_idx)
            return pb

        if self._last_person_bbox is not None and (frame_idx - self._last_person_frame) <= self.person_bbox_recent_frames:
            return self._last_person_bbox

        return None

    def _reset_release_started(self):
        """Clear release snapshot bookkeeping (called on re-arming / reset)."""
        self._release_started_frame = -10**9
        self._release_started_reason = None
        self._release_started_snapshot = None

    def _maybe_mark_release_started(self, frame_idx: int, *, shoot_info: Dict[str, Any], ctx, release: ReleaseInfo):
        """
        Record the first frame where release_signal becomes True in ARMED.

        This is for debugging only:
        without this snapshot, the overlay may show “all false” at the trigger frame
        because the decisive signal happened a few frames earlier (debounce effect).
        """
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

        self._release_started_snapshot = {
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

    def _is_valid_dist(self, d: Optional[float]) -> bool:
        """
        Distance sanity check used for sep_ok baseline.

        We explicitly reject sentinel values (very large numbers) and non-finite values.
        This prevents sep_ok from triggering due to invalid bookkeeping.
        """
        if d is None:
            return False
        try:
            d = float(d)
        except (TypeError, ValueError):
            return False
        return math.isfinite(d) and (0.0 <= d < 1e8)

    def _safe_dist(self, d: Optional[float]) -> Optional[float]:
        """Return float(d) if valid, otherwise None."""
        return float(d) if self._is_valid_dist(d) else None

    # -----------------
    # main update
    # -----------------
    def update(self, frame_idx: int, detections: List[Dict[str, Any]], ball_state) -> Optional[AttemptEvent]:
        """
        Main per-frame update.

        Returns:
        - AttemptEvent exactly once when a shot attempt is triggered, else None.

        The typical flow is:
        - Maintain rim memory
        - Apply rearm lock (optional)
        - Build shoot/person/ball inputs
        - If LOCKED: clock-only
        - Else: build context, apply gates, compute arm/release signals, update FSM
        """

        # -------------------------
        # 1) Rim memory (used at trigger time to attach rim coordinates)
        # -------------------------
        rim_det = pick_best(detections, "rim")
        if rim_det is not None:
            self.last_rim_xy = det_center(rim_det)
            self.last_rim_bbox = det_bbox(rim_det)
            self.last_rim_frame = int(frame_idx)

        # -------------------------
        # 2) Optional re-arm lock: require ball below rim before accepting a new attempt
        # -------------------------
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

        # -------------------------
        # 3) Perception-side signals (shoot, person, ball)
        # -------------------------
        shoot_info = self.shoot.update(frame_idx, detections)
        shoot_now = bool(shoot_info.get("shoot_now", False))
        shoot_bbox = self._choose_shoot_bbox(frame_idx, shoot_info)

        person_det = pick_best(detections, "person")
        ball_det = pick_best(detections, "ball")
        person_bbox = self._choose_person_bbox(frame_idx, person_det)

        # -------------------------
        # 4) LOCKED: never recompute gates/release; only advance the timer.
        # -------------------------
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

        # -------------------------
        # 5) IDLE prerequisites: we require shoot_now + person bbox + shoot bbox
        # -------------------------
        if self.fsm.state == AttemptFSM.IDLE:
            self._reset_release_started()

            # If we cannot even define the "shoot configuration", we do not arm.
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

        # -------------------------
        # 6) ARMED fallback path: allow release by separation even if shoot bbox disappears
        #
        # Motivation:
        # - At long distance, 'shoot' bbox can be missed intermittently.
        # - Once ARMED, we prefer to keep continuity and avoid false negatives.
        # - This fallback only triggers release via sep_ok (with additional safeguards).
        # -------------------------
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

            # Resolve ball point (tracker/yolo/memory). Needed for sep_ok computation.
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

            # Compute ball_rel_y and rel_y_rise in the same convention as build_context()
            x1, y1, x2, y2 = person_bbox
            h = max(1.0, (y2 - y1))
            ball_rel_y = float((by - y1) / h)

            rel_y_rise = 0.0
            if self._armed_ball_rel_y is not None:
                rel_y_rise = float(self._armed_ball_rel_y - ball_rel_y)

            s = self._scale()
            sep_thr = float(self.release_sep_increase_px * s)

            # sep_ok relies on a baseline distance at arming; both must be valid.
            d_bp_safe = self._safe_dist(d_bp)
            armed_bp_safe = self._safe_dist(self._armed_dist_bp)

            raw_sep_ok = False
            if (d_bp_safe is not None) and (armed_bp_safe is not None):
                raw_sep_ok = (d_bp_safe - armed_bp_safe) >= sep_thr

            # Additional guard: require a minimal rise to avoid bbox jitter triggers.
            sep_ok = bool(raw_sep_ok and (rel_y_rise >= self.min_rel_y_rise_for_sep))

            # Record the moment release became true (debug transparency).
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

                # At trigger time, we require a recent rim reference to attach geometry.
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

        # From this point, the "standard" path requires both person and shoot bbox.
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

        # -------------------------
        # 7) Resolve ball point (tracker/yolo/memory) and build the per-frame context
        # -------------------------
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

        # -------------------------
        # 8) Gates (false-positive control)
        # -------------------------
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

        # -------------------------
        # 9) Arm signal (IDLE -> ARMED)
        # -------------------------
        shoot_rise = bool(shoot_info.get("shoot_rise", False))
        shoot_streak = int(shoot_info.get("shoot_streak", 0))
        shoot_from_memory = bool(shoot_info.get("shoot_from_memory", False))

        # Two arming modes:
        # - rise_arm_ok: prefer a true "rising edge" from a fresh detection
        # - streak_arm_ok: tolerate missing rise by using a short shoot streak (robustness)
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

        # If ARMED and we don't get a release within a bounded window, reset to IDLE.
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

        # -------------------------
        # 10) Release signal (ARMED -> ATTEMPT)
        # -------------------------
        armed_clean = bool(self._armed_via_rise or streak_arm_ok)

        # sep_ok requires a valid baseline distance at arming AND a valid current distance.
        # If either is invalid, we forbid sep-based release to avoid spurious triggers.
        armed_bp_valid = self._is_valid_dist(self._armed_dist_bp)
        ctx_bp_valid = self._is_valid_dist(getattr(ctx, "d_ball_person", None))
        allow_sep = bool(armed_bp_valid and ctx_bp_valid)

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

        # If sep is not allowed, keep only left_shoot (and optional motion) for FSM.
        if allow_sep:
            release_signal_safe = bool(getattr(release, "release_signal", False))
        else:
            release_signal_safe = (
                bool(getattr(release, "left_shoot", False))
                or bool(getattr(release, "motion_ok", False))
                or bool(getattr(release, "motion_release", False))
            )

        self._maybe_mark_release_started(frame_idx, shoot_info=shoot_info, ctx=ctx, release=release)

        prev_state = self.fsm.state
        evt = self.fsm.update(frame_idx, arm_signal=arm_signal, release_signal=release_signal_safe)

        # When entering ARMED, store baselines used by rel_y_rise and sep_ok.
        if prev_state == AttemptFSM.IDLE and self.fsm.state == AttemptFSM.ARMED:
            self._armed_frame = int(frame_idx)
            self._armed_dist_bp = self._safe_dist(getattr(ctx, "d_ball_person", None))  # only store if valid
            self._armed_ball_rel_y = float(ctx.ball_rel_y)
            self._armed_via_rise = bool(rise_arm_ok)
            self._reset_release_started()

        # If we fell back to IDLE, clear release snapshot.
        if prev_state != AttemptFSM.IDLE and self.fsm.state == AttemptFSM.IDLE:
            self._reset_release_started()

        # -------------------------
        # 11) ATTEMPT: emit event once, attach rim + ball geometry
        # -------------------------
        if evt is not None and getattr(evt, "name", None) == "ATTEMPT":
            if self.require_ball_below_rim_to_rearm:
                self._waiting_below = True
                self._below_streak = 0

            # Do not emit an attempt if rim reference is too old:
            # outcome classification needs consistent rim geometry.
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

        # Default path: no event yet, report current status for overlays.
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
