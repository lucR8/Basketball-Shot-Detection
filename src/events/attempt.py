from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math

from src.events.shoot_signal import ShootSignalTracker
from src.events.ball_point import BallPointResolver
from src.events.spatial_gates import (
    point_in_bbox,
    dist_point_to_bbox,
    person_cover_ratio,
)
from src.events.attempt_fsm import AttemptFSM


@dataclass
class AttemptEvent:
    frame_idx: int
    ball_cx: float
    ball_cy: float
    rim_cx: float
    rim_cy: float
    distance_px: float
    details: str = ""


def _pick_best(detections: List[Dict[str, Any]], cls_name: str) -> Optional[Dict[str, Any]]:
    cls = cls_name.lower()
    cand = [d for d in detections if str(d.get("name", "")).lower() == cls]
    if not cand:
        return None
    return max(cand, key=lambda d: float(d.get("conf", 0.0)))


def _center(det: Dict[str, Any]) -> Tuple[float, float]:
    return (
        (float(det["x1"]) + float(det["x2"])) / 2.0,
        (float(det["y1"]) + float(det["y2"])) / 2.0,
    )


def _bbox(det: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(det["x1"]),
        float(det["y1"]),
        float(det["x2"]),
        float(det["y2"]),
    )


class AttemptDetector:
    """
    Attempt detector (events/).

    - shoot signal + memory (ShootSignalTracker)
    - ball point resolver (BallPointResolver)
    - gates spatiaux
    - arming/release via AttemptFSM
    - rim scaling optionnel

    Patch anti faux positif "balle au sol":
      - interdit d'armer juste sur shoot_streak/memory si ball_in_shoot == False
        => évite ARMED + left_shoot=True en boucle => attempt fantôme
    """

    def __init__(
        self,
        enter_radius_px: float = 85.0,

        # time memory
        rim_recent_frames: int = 15,
        ball_recent_frames: int = 25,
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

        # rim scaling (compat main.py)
        enable_rim_scaling: bool = False,
        rim_ref_width_px: float | None = None,
        rim_ref_w: float | None = None,
        rim_scale_min: float = 0.6,
        rim_scale_max: float = 1.8,

        # compat main.py (acceptés mais non utilisés ici)
        ball_point_memory_frames: int = 6,
        allow_oversize_ball_when_armed: bool = False,
        oversize_ball_conf_min: float = 0.20,
        oversize_ball_dist_boost: float = 1.35,

        # "ball too low" gate simple (reste utile)
        ball_too_low_rel_y: float = 0.50,

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

        self.enable_rim_scaling = bool(enable_rim_scaling)
        self.rim_ref_width_px = float(rim_ref_width_px)
        self.rim_scale_min = float(rim_scale_min)
        self.rim_scale_max = float(rim_scale_max)

        self.ball_too_low_rel_y = float(ball_too_low_rel_y)

        self.debug = bool(debug)

        self.shoot = ShootSignalTracker(self.shoot_conf_min, memory_frames=int(shoot_memory_frames))

        # IMPORTANT: pas de kwargs "oversize" ici (BallPointResolver ne les supporte pas)
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

        self.last_rim_xy: Optional[Tuple[float, float]] = None
        self.last_rim_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_rim_frame: int = -10**9

        self._waiting_below = False
        self._below_streak = 0

        self._armed_frame: int = -10**9
        self._armed_dist_bp: Optional[float] = None

        self.last_debug: Dict[str, Any] = {}

    def _scale(self) -> float:
        if (not self.enable_rim_scaling) or (self.last_rim_bbox is None):
            return 1.0
        x1, _, x2, _ = self.last_rim_bbox
        w = max(1.0, float(x2 - x1))
        s = w / max(1.0, self.rim_ref_width_px)
        return max(self.rim_scale_min, min(self.rim_scale_max, s))

    def update(self, frame_idx: int, detections: List[Dict[str, Any]], ball_state) -> Optional[AttemptEvent]:
        # rim memory
        rim = _pick_best(detections, "rim")
        if rim is not None:
            self.last_rim_xy = _center(rim)
            self.last_rim_bbox = _bbox(rim)
            self.last_rim_frame = frame_idx

        # below-rim lock
        if self._waiting_below and self.last_rim_xy is not None:
            _, rim_cy = self.last_rim_xy
            below_now = False
            if ball_state is not None:
                below_now = float(ball_state.cy) >= (rim_cy + self.below_margin_px)

            self._below_streak = self._below_streak + 1 if below_now else 0
            if self._below_streak < self.below_confirm_frames:
                self.last_debug = {"frame": frame_idx, "gate_reason": "blocked_waiting_below_rim"}
                return None
            self._waiting_below = False
            self._below_streak = 0

        shoot_info = self.shoot.update(frame_idx, detections)
        person = _pick_best(detections, "person")
        ball_det = _pick_best(detections, "ball")

        if (not shoot_info["shoot_now"]) or (person is None):
            self.last_debug = {"frame": frame_idx, "gate_reason": "missing_shoot_or_person"}
            return None

        shoot_bbox = shoot_info.get("shoot_bbox")
        if shoot_bbox is None:
            self.last_debug = {"frame": frame_idx, "gate_reason": "missing_shoot_bbox"}
            return None

        person_bbox = _bbox(person)

        ball_point = self.ball.update(frame_idx, ball_state, ball_det, person_bbox)
        if ball_point is None:
            self.last_debug = {"frame": frame_idx, "gate_reason": "no_ball_point"}
            return None

        bx, by, src = ball_point

        s = self._scale()
        pad_person = self.person_in_shoot_pad_px * s
        pad_ball = self.ball_in_shoot_pad_px * s
        dist_thr = self.ball_person_max_dist_px * s
        sep_thr = self.release_sep_increase_px * s

        # person in shoot
        cover = person_cover_ratio(person_bbox, shoot_bbox, pad=pad_person)
        if cover < self.person_in_shoot_min_cover:
            self.last_debug = {"frame": frame_idx, "gate_reason": "fail_person_in_shoot", "cover": float(cover)}
            return None

        # ball in shoot + dist ball-person
        ball_in_shoot = point_in_bbox(bx, by, shoot_bbox, pad=pad_ball)
        d_bp = dist_point_to_bbox(bx, by, person_bbox)

        # soft: si balle dans shoot bbox -> on ne la pénalise pas sur close-person
        if (not ball_in_shoot) and (d_bp > dist_thr):
            self.last_debug = {
                "frame": frame_idx,
                "gate_reason": "fail_ball_close_person",
                "d_ball_person": float(d_bp),
                "dist_thr": float(dist_thr),
                "ball_src": src,
            }
            return None

        # ball too low (simple)
        x1, y1, x2, y2 = person_bbox
        h = max(1.0, (y2 - y1))
        ball_rel_y = float((by - y1) / h)  # 0=haut, 1=bas
        if (self.fsm.state != AttemptFSM.ARMED) and (not ball_in_shoot) and (ball_rel_y >= self.ball_too_low_rel_y):
            self.last_debug = {
                "frame": frame_idx,
                "gate_reason": "ball_too_low_for_shot",
                "ball_rel_y": float(ball_rel_y),
                "thr": float(self.ball_too_low_rel_y),
            }
            return None

        # -------------------------------------------------
        # ✅ PATCH IMPORTANT : arm_signal "streak" autorisé seulement si ball_in_shoot
        # -------------------------------------------------
        shoot_rise = bool(shoot_info.get("shoot_rise", False))
        shoot_streak = int(shoot_info.get("shoot_streak", 0))

        # ancien: arm_signal = shoot_rise or shoot_streak >= 2
        # nouveau:
        # - shoot_rise reste suffisant (signal fort)
        # - streak>=2 ne suffit que si la balle est effectivement "dans la zone shoot"
        arm_signal = shoot_rise or (shoot_streak >= 2 and ball_in_shoot)

        if not arm_signal and self.fsm.state == AttemptFSM.IDLE:
            # debug explicite (utile pour vérifier que le faux positif saute bien)
            self.last_debug = {
                "frame": frame_idx,
                "gate_reason": "not_armed_waiting_shoot_rise",
                "shoot_rise": bool(shoot_rise),
                "shoot_streak": int(shoot_streak),
                "ball_in_shoot": bool(ball_in_shoot),
            }
            return None
        # -------------------------------------------------

        # arming timeout
        if self.fsm.state == AttemptFSM.ARMED and self._armed_frame > -10**8:
            if (frame_idx - self._armed_frame) > self.shoot_arm_window:
                self.fsm.reset()
                self._armed_frame = -10**9
                self._armed_dist_bp = None
                self.last_debug = {"frame": frame_idx, "gate_reason": "arming_window_expired"}
                return None

        # release
        left_shoot = not ball_in_shoot
        sep_ok = (
            self.fsm.state == AttemptFSM.ARMED
            and self._armed_dist_bp is not None
            and (d_bp - self._armed_dist_bp) >= sep_thr
        )

        prev_state = self.fsm.state
        evt = self.fsm.update(frame_idx, arm_signal=arm_signal, release_signal=(left_shoot or sep_ok))

        # baseline distance at arming
        if prev_state == AttemptFSM.IDLE and self.fsm.state == AttemptFSM.ARMED:
            self._armed_frame = frame_idx
            self._armed_dist_bp = float(d_bp)

        if evt is not None and evt.name == "ATTEMPT":
            if self.require_ball_below_rim_to_rearm:
                self._waiting_below = True
                self._below_streak = 0

            if self.last_rim_xy is None or (frame_idx - self.last_rim_frame) > self.rim_recent_frames:
                self.last_debug = {"frame": frame_idx, "gate_reason": "attempt_but_rim_stale"}
                return None

            rim_cx, rim_cy = self.last_rim_xy
            dist_br = math.hypot(bx - rim_cx, by - rim_cy)

            self.last_debug = {
                "frame": frame_idx,
                "gate_reason": "ATTEMPT_TRIGGERED",
                "ball_src": src,
                "shoot_rise": bool(shoot_rise),
                "shoot_streak": int(shoot_streak),
                "ball_in_shoot": bool(ball_in_shoot),
                "left": bool(left_shoot),
                "sep_ok": bool(sep_ok),
                "ball_rel_y": float(ball_rel_y),
                "scale": float(s),
            }

            return AttemptEvent(
                frame_idx=frame_idx,
                ball_cx=bx,
                ball_cy=by,
                rim_cx=rim_cx,
                rim_cy=rim_cy,
                distance_px=float(dist_br),
                details=f"release(ball_src={src}, left={left_shoot}, sep={sep_ok}, scale={s:.2f})",
            )

        self.last_debug = {
            "frame": frame_idx,
            "gate_reason": "armed_waiting_release" if self.fsm.state == AttemptFSM.ARMED else self.fsm.state,
            "shoot_rise": bool(shoot_rise),
            "shoot_streak": int(shoot_streak),
            "ball_in_shoot": bool(ball_in_shoot),
            "ball_rel_y": float(ball_rel_y),
        }
        return None
