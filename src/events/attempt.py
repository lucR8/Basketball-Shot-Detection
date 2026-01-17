from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math

from src.track.ball_tracker import BallState


@dataclass
class AttemptEvent:
    """
    Event emitted when a shot attempt is detected.
    We store ball/rim positions (at trigger time) so downstream logic can use them
    even if detections disappear in the next frames.
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
    Attempt detection with HARD gating on:
      - shoot bbox present
      - person inside shoot bbox (robust overlap / coverage test)
      - ball inside shoot bbox
      - ball close to person (Option A: distance to PERSON BBOX, not person center)
    Then attempt triggers on RELEASE:
      - ball exits shoot bbox (with pad) OR
      - ball-person distance increases enough (separation)
      - with debounce (N consecutive frames)

    Notes:
      - person_in_shoot "overlap" mode is PERSON COVERAGE: inter(person, shoot) / area(person)
        (more intuitive than IoU when shoot box is much larger than person box).
      - "ball close to person" uses distance from ball point to the PERSON BBOX (0 if inside).
      - thresholds are in pixels. If you resize to 1080p, keep them around these defaults.
        If you keep true 4K, consider scaling them (or compute a scale factor upstream).
    """

    def __init__(
        self,
        # --- legacy / compatibility ---
        enter_radius_px: float = 85.0,
        vy_min: float = 0.2,
        cooldown_frames: int = 12,
        require_approach: bool = True,
        approach_window: int = 6,

        # --- shoot logic ---
        shoot_conf_min: float = 0.18,
        shoot_conf_strong: float = 0.40,      # kept for compatibility
        ball_recent_frames: int = 25,         # kept for compatibility
        rim_recent_frames: int = 15,

        # --- arming / release window ---
        shoot_arm_window: int = 25,
        # If "shoot" is already active when gates become valid, we can still arm
        # after a short stable streak (helps when shoot bbox is intermittent).
        shoot_arm_stable_frames: int = 2,
        # Pad used ONLY for the "ball must be inside shoot" gate.
        # Keep this a bit larger than release_exit_pad_px to increase recall.
        ball_in_shoot_pad_px: float = 25.0,
        release_exit_pad_px: float = 8.0,
        release_sep_increase_px: float = 35.0,
        release_debounce_frames: int = 2,

        # --- HARD constraints ---
        # Option A (recommended): distance to PERSON BBOX in px (not center)
        ball_person_max_dist_px: float = 90.0,
        require_person_in_shoot: bool = True,
        require_ball_in_shoot: bool = True,

        # --- person in shoot ---
        # "overlap" means PERSON COVERAGE (intersection/person_area), not IoU.
        person_in_shoot_mode: str = "overlap",   # "center" or "overlap"
        person_in_shoot_min_iou: float = 0.02,   # kept for logging only
        person_in_shoot_pad_px: float = 20.0,
        person_in_shoot_min_cover: float = 0.55,

        # --- anti false ball (hand) ---
        enable_ball_size_filter: bool = True,
        ball_area_min_px2: float = 150.0,
        ball_area_max_px2: float = 20000.0,

        # --- anti rebound / double attempts ---
        require_ball_below_rim_to_rearm: bool = True,
        below_margin_px: float = 45.0,
        below_confirm_frames: int = 3,

        # --- shot window ---
        shot_window_frames: int = 35,

        # --- optional compatibility knobs ---
        post_attempt_block_until_ball_far: bool = False,
        post_attempt_far_factor: float = 1.6,
        post_attempt_min_frames: int = 10,

        # --- OPTIONAL ---
        allow_ball_based: bool = False,

        # --- debug ---
        debug: bool = False,
        debug_every: int = 15,
    ):
        self.enter_radius_px = float(enter_radius_px)
        self.vy_min = float(vy_min)
        self.cooldown_frames = int(cooldown_frames)
        self.require_approach = bool(require_approach)
        self.approach_window = int(approach_window)

        self.shoot_conf_min = float(shoot_conf_min)
        self.shoot_conf_strong = float(shoot_conf_strong)
        self.ball_recent_frames = int(ball_recent_frames)
        self.rim_recent_frames = int(rim_recent_frames)

        self.shoot_arm_window = int(shoot_arm_window)
        self.shoot_arm_stable_frames = max(1, int(shoot_arm_stable_frames))
        self.ball_in_shoot_pad_px = float(ball_in_shoot_pad_px)
        self.release_exit_pad_px = float(release_exit_pad_px)
        self.release_sep_increase_px = float(release_sep_increase_px)
        self.release_debounce_frames = max(1, int(release_debounce_frames))

        self.ball_person_max_dist_px = float(ball_person_max_dist_px)
        self.require_person_in_shoot = bool(require_person_in_shoot)
        self.require_ball_in_shoot = bool(require_ball_in_shoot)

        self.person_in_shoot_mode = str(person_in_shoot_mode).lower().strip()
        if self.person_in_shoot_mode not in ("center", "overlap"):
            self.person_in_shoot_mode = "overlap"
        self.person_in_shoot_min_iou = float(person_in_shoot_min_iou)
        self.person_in_shoot_pad_px = float(person_in_shoot_pad_px)
        self.person_in_shoot_min_cover = float(person_in_shoot_min_cover)

        self.enable_ball_size_filter = bool(enable_ball_size_filter)
        self.ball_area_min_px2 = float(ball_area_min_px2)
        self.ball_area_max_px2 = float(ball_area_max_px2)

        self.require_ball_below_rim_to_rearm = bool(require_ball_below_rim_to_rearm)
        self.below_margin_px = float(below_margin_px)
        self.below_confirm_frames = int(below_confirm_frames)

        self.shot_window_frames = int(shot_window_frames)

        self.post_attempt_block_until_ball_far = bool(post_attempt_block_until_ball_far)
        self.post_attempt_far_factor = float(post_attempt_far_factor)
        self.post_attempt_min_frames = int(post_attempt_min_frames)

        self.allow_ball_based = bool(allow_ball_based)

        self.debug = bool(debug)
        self.debug_every = max(1, int(debug_every))

        # state
        self._cooldown = 0

        self._last_rim_frame = -10**9
        self._last_rim_pos: Optional[Tuple[float, float]] = None

        self._shoot_active = False
        self._shoot_streak = 0
        self._armed_shoot_frame = -10**9

        self._armed_ball_pos: Optional[Tuple[float, float]] = None
        self._armed_person_pos: Optional[Tuple[float, float]] = None
        self._armed_ball_person_dist: Optional[float] = None
        self._armed_shoot_conf: float = 0.0

        self._release_streak = 0

        self._waiting_ball_below_rim = False
        self._below_rim_streak = 0

        self._shot_window_active = False
        self._shot_window_end = -10**9
        self._shot_window_attempt_emitted = False

        self._post_attempt_block = 0
        self._waiting_ball_far = False

        self._dist_hist: List[Tuple[int, float]] = []

        self.last_debug: Dict[str, Any] = {}

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _center(det: Dict[str, Any]) -> Tuple[float, float]:
        return ((float(det["x1"]) + float(det["x2"])) / 2.0,
                (float(det["y1"]) + float(det["y2"])) / 2.0)

    @staticmethod
    def _bbox(det: Dict[str, Any]) -> Tuple[float, float, float, float]:
        return float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _clip_bbox(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = b
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    @staticmethod
    def _point_in_bbox(x: float, y: float, b: Tuple[float, float, float, float], pad: float = 0.0) -> bool:
        x1, y1, x2, y2 = AttemptDetector._clip_bbox(b)
        return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)

    @staticmethod
    def _area(det: Dict[str, Any]) -> float:
        w = max(0.0, float(det["x2"]) - float(det["x1"]))
        h = max(0.0, float(det["y2"]) - float(det["y1"]))
        return w * h

    @staticmethod
    def _bbox_intersection(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = AttemptDetector._clip_bbox(a)
        bx1, by1, bx2, by2 = AttemptDetector._clip_bbox(b)
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        return iw * ih

    @staticmethod
    def _bbox_area(b: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = AttemptDetector._clip_bbox(b)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _bbox_iou(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        inter = self._bbox_intersection(a, b)
        if inter <= 0.0:
            return 0.0
        ua = self._bbox_area(a)
        ub = self._bbox_area(b)
        denom = ua + ub - inter
        return float(inter / denom) if denom > 1e-9 else 0.0

    def _expand_bbox(self, b: Tuple[float, float, float, float], pad: float) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self._clip_bbox(b)
        return (x1 - pad, y1 - pad, x2 + pad, y2 + pad)

    def _pick_best(self, detections: List[Dict[str, Any]], cls_name: str) -> Optional[Dict[str, Any]]:
        cls = cls_name.lower()
        cand = [d for d in detections if str(d.get("name", "")).lower() == cls]
        if not cand:
            return None
        return max(cand, key=lambda d: float(d.get("conf", 0.0)))

    @staticmethod
    def _dist_point_to_bbox(px: float, py: float, b: Tuple[float, float, float, float]) -> float:
        """
        Option A:
          distance from point to bbox (0 if inside).
        """
        x1, y1, x2, y2 = AttemptDetector._clip_bbox(b)
        dx = 0.0
        if px < x1:
            dx = x1 - px
        elif px > x2:
            dx = px - x2

        dy = 0.0
        if py < y1:
            dy = y1 - py
        elif py > y2:
            dy = py - y2

        return math.hypot(dx, dy)

    def _shot_window_update(self, frame_idx: int) -> None:
        if self._shot_window_active and frame_idx > self._shot_window_end:
            self._shot_window_active = False
            self._shot_window_attempt_emitted = False

    def _open_shot_window(self, frame_idx: int) -> None:
        self._shot_window_active = True
        self._shot_window_end = frame_idx + self.shot_window_frames
        self._shot_window_attempt_emitted = False

    def _disarm(self) -> None:
        self._armed_shoot_frame = -10**9
        self._armed_ball_pos = None
        self._armed_person_pos = None
        self._armed_ball_person_dist = None
        self._armed_shoot_conf = 0.0
        self._release_streak = 0

    def _start_post_attempt_locks(self) -> None:
        if self.require_ball_below_rim_to_rearm:
            self._waiting_ball_below_rim = True
            self._below_rim_streak = 0

        if self.post_attempt_block_until_ball_far:
            self._post_attempt_block = max(self._post_attempt_block, self.post_attempt_min_frames)
            self._waiting_ball_far = True

    def _update_last_debug(
        self,
        frame_idx: int,
        shoot_det: Optional[Dict[str, Any]],
        person_det: Optional[Dict[str, Any]],
        ball_det: Optional[Dict[str, Any]],
        ball_state: Optional[BallState],
        shoot_now: bool,
        shoot_rise: bool,
        person_in_shoot: bool,
        ball_in_shoot: bool,
        d_ball_person: float,
        left_shoot: bool,
        sep_ok: bool,
        release_signal: bool,
        gate_reason: str,
        *,
        person_overlap_ratio: Optional[float] = None,
        iou_person_shoot: Optional[float] = None,
        ball_source: str = "tracker",
    ) -> None:
        self.last_debug = {
            "frame": int(frame_idx),
            "gate_reason": gate_reason,
            "armed": bool(self._armed_shoot_frame > -10**8),
            "armed_dt": int(frame_idx - self._armed_shoot_frame) if self._armed_shoot_frame > -10**8 else None,
            "release_streak": int(self._release_streak),
            "release_debounce": int(self.release_debounce_frames),
            "shoot_now": bool(shoot_now),
            "shoot_rise": bool(shoot_rise),
            "shoot_streak": int(getattr(self, "_shoot_streak", 0)),
            "shoot_conf": float(shoot_det["conf"]) if shoot_det is not None else 0.0,
            "person_conf": float(person_det["conf"]) if person_det is not None else 0.0,
            "ball_conf": float(ball_det["conf"]) if ball_det is not None else 0.0,
            "person_in_shoot": bool(person_in_shoot),
            "ball_in_shoot": bool(ball_in_shoot),
            "d_ball_person": float(d_ball_person),
            "ball_person_max_dist_px": float(self.ball_person_max_dist_px),
            "left_shoot": bool(left_shoot),
            "sep_ok": bool(sep_ok),
            "release_signal": bool(release_signal),
            "person_overlap_ratio": float(person_overlap_ratio) if person_overlap_ratio is not None else None,
            "iou_person_shoot": float(iou_person_shoot) if iou_person_shoot is not None else None,
            "ball_source": str(ball_source),
            "shoot_bbox": self._clip_bbox(self._bbox(shoot_det)) if shoot_det is not None else None,
            "person_bbox": self._clip_bbox(self._bbox(person_det)) if person_det is not None else None,
            "ball_bbox": self._clip_bbox(self._bbox(ball_det)) if ball_det is not None else None,
            "ball_xy": (float(ball_state.cx), float(ball_state.cy)) if ball_state is not None else None,
            "rim_xy": self._last_rim_pos,
        }

    # -----------------------------
    # Main
    # -----------------------------
    def update(
        self,
        frame_idx: int,
        detections: List[Dict[str, Any]],
        ball_state: Optional[BallState],
    ) -> Optional[AttemptEvent]:

        # cooldown
        if self._cooldown > 0:
            self._cooldown -= 1
        if self._post_attempt_block > 0:
            self._post_attempt_block -= 1

        self._shot_window_update(frame_idx)

        # rim memory
        rim_det = self._pick_best(detections, "rim")
        if rim_det is not None:
            self._last_rim_pos = self._center(rim_det)
            self._last_rim_frame = frame_idx

        if self._shot_window_active and self._shot_window_attempt_emitted:
            self._disarm()
            self._update_last_debug(
                frame_idx, None, None, None, ball_state,
                shoot_now=False, shoot_rise=False,
                person_in_shoot=False, ball_in_shoot=False,
                d_ball_person=float("inf"),
                left_shoot=False, sep_ok=False, release_signal=False,
                gate_reason="blocked_shot_window",
            )
            return None

        # below-rim lock
        if self._waiting_ball_below_rim and self._last_rim_pos is not None:
            rim_cx, rim_cy = self._last_rim_pos
            below_now = False
            if ball_state is not None:
                below_now = float(ball_state.cy) >= (rim_cy + self.below_margin_px)

            if below_now:
                self._below_rim_streak += 1
            else:
                self._below_rim_streak = 0

            if self._below_rim_streak >= self.below_confirm_frames:
                self._waiting_ball_below_rim = False
                self._below_rim_streak = 0
            else:
                self._disarm()
                self._update_last_debug(
                    frame_idx, None, None, None, ball_state,
                    shoot_now=False, shoot_rise=False,
                    person_in_shoot=False, ball_in_shoot=False,
                    d_ball_person=float("inf"),
                    left_shoot=False, sep_ok=False, release_signal=False,
                    gate_reason="blocked_waiting_below_rim",
                )
                return None

        # optional distance lock
        if self.post_attempt_block_until_ball_far and self._waiting_ball_far and self._last_rim_pos is not None:
            rim_cx, rim_cy = self._last_rim_pos
            if ball_state is not None:
                dist_ball_rim = self._dist((float(ball_state.cx), float(ball_state.cy)), (rim_cx, rim_cy))
                far_thr = self.enter_radius_px * self.post_attempt_far_factor
                if dist_ball_rim >= far_thr:
                    self._waiting_ball_far = False

            if self._waiting_ball_far and self._post_attempt_block > 0:
                self._disarm()
                self._update_last_debug(
                    frame_idx, None, None, None, ball_state,
                    shoot_now=False, shoot_rise=False,
                    person_in_shoot=False, ball_in_shoot=False,
                    d_ball_person=float("inf"),
                    left_shoot=False, sep_ok=False, release_signal=False,
                    gate_reason="blocked_waiting_far",
                )
                return None

        if self._cooldown != 0:
            self._update_last_debug(
                frame_idx, None, None, None, ball_state,
                shoot_now=False, shoot_rise=False,
                person_in_shoot=False, ball_in_shoot=False,
                d_ball_person=float("inf"),
                left_shoot=False, sep_ok=False, release_signal=False,
                gate_reason="blocked_cooldown",
            )
            return None

        # detections
        shoot_det = self._pick_best(detections, "shoot")
        person_det = self._pick_best(detections, "person")
        ball_det = self._pick_best(detections, "ball")

        shoot_conf = float(shoot_det["conf"]) if shoot_det is not None else 0.0
        shoot_now = (shoot_det is not None) and (shoot_conf >= self.shoot_conf_min)
        shoot_rise = shoot_now and not self._shoot_active
        self._shoot_active = shoot_now
        # Track how stable the shoot signal is. This helps arming when the first
        # valid gates happen AFTER the initial shoot_rise frame.
        self._shoot_streak = (self._shoot_streak + 1) if shoot_now else 0

        # ------------------------------------------------------------
        # Choose ball point used for gating (tracker + YOLO fallback)
        # ------------------------------------------------------------
        # IMPORTANT:
        # - Previously, if ball_state was None (tracker lost), we returned early ("no_ball_state").
        #   This kills recall because YOLO may still detect the ball while the tracker drops.
        # - Patch A: if tracker is lost but YOLO provides a valid ball detection, proceed using YOLO only.

        shoot_det = self._pick_best(detections, "shoot")
        person_det = self._pick_best(detections, "person")
        ball_det = self._pick_best(detections, "ball")

        shoot_conf = float(shoot_det["conf"]) if shoot_det is not None else 0.0
        shoot_now = (shoot_det is not None) and (shoot_conf >= self.shoot_conf_min)
        shoot_rise = shoot_now and not self._shoot_active
        self._shoot_active = shoot_now

        # Track stability of shoot signal (helps arming when shoot_rise is missed)
        self._shoot_streak = (self._shoot_streak + 1) if shoot_now else 0

        # ---- ball point selection ----
        ball_source = "none"
        bx = None
        by = None

        # 1) Start from tracker if available
        if ball_state is not None:
            bx, by = float(ball_state.cx), float(ball_state.cy)
            ball_source = "tracker"

        # 2) Prefer YOLO ball if present and passes size filter
        if ball_det is not None:
            if self.enable_ball_size_filter:
                a = self._area(ball_det)
                if self.ball_area_min_px2 <= a <= self.ball_area_max_px2:
                    bx, by = self._center(ball_det)
                    ball_source = "yolo" if ball_state is not None else "yolo_only"
            else:
                bx, by = self._center(ball_det)
                ball_source = "yolo" if ball_state is not None else "yolo_only"

        # 3) If still no ball point, we truly cannot proceed
        if bx is None or by is None:
            self._update_last_debug(
                frame_idx, shoot_det, person_det, ball_det, ball_state,
                shoot_now=shoot_now, shoot_rise=shoot_rise,
                person_in_shoot=False, ball_in_shoot=False,
                d_ball_person=float("inf"),
                left_shoot=False, sep_ok=False, release_signal=False,
                gate_reason="no_ball_point",
                ball_source="none",
            )
            return None

        # must have shoot + person (keep current design)
        if shoot_det is None or person_det is None:
            self._update_last_debug(
                frame_idx, shoot_det, person_det, ball_det, ball_state,
                shoot_now=shoot_now, shoot_rise=shoot_rise,
                person_in_shoot=False, ball_in_shoot=False,
                d_ball_person=float("inf"),
                left_shoot=False, sep_ok=False, release_signal=False,
                gate_reason="missing_shoot_or_person",
                ball_source=ball_source,
            )
            return None

        shoot_bbox = self._clip_bbox(self._bbox(shoot_det))
        person_bbox = self._clip_bbox(self._bbox(person_det))

        pcx, pcy = self._center(person_det)

        # person in shoot (coverage)
        person_in_shoot = False
        person_overlap_ratio = 0.0
        iou_ps = 0.0

        if self.person_in_shoot_mode == "center":
            person_in_shoot = self._point_in_bbox(pcx, pcy, shoot_bbox, pad=self.person_in_shoot_pad_px)
        else:
            shoot_exp = self._expand_bbox(shoot_bbox, self.person_in_shoot_pad_px)

            inter = self._bbox_intersection(person_bbox, shoot_exp)
            p_area = max(1e-9, self._bbox_area(person_bbox))
            person_overlap_ratio = inter / p_area  # 1.0 = person fully covered by shoot_exp

            iou_ps = self._bbox_iou(person_bbox, shoot_exp)  # debug only

            person_in_shoot = person_overlap_ratio >= self.person_in_shoot_min_cover

        # ball in shoot (use chosen bx/by)
        # NOTE: gate pad is intentionally a bit larger than the exit pad to
        # improve recall without making the release trigger too early.
        ball_in_shoot = self._point_in_bbox(bx, by, shoot_bbox, pad=self.ball_in_shoot_pad_px)

        # HARD gate
        if self.require_person_in_shoot and not person_in_shoot:
            self._update_last_debug(
                frame_idx, shoot_det, person_det, ball_det, ball_state,
                shoot_now=shoot_now, shoot_rise=shoot_rise,
                person_in_shoot=person_in_shoot, ball_in_shoot=ball_in_shoot,
                d_ball_person=float("inf"),
                left_shoot=False, sep_ok=False, release_signal=False,
                gate_reason="fail_person_in_shoot",
                person_overlap_ratio=person_overlap_ratio,
                iou_person_shoot=iou_ps,
                ball_source=ball_source,
            )
            return None

        if self.require_ball_in_shoot and not ball_in_shoot:
            self._update_last_debug(
                frame_idx, shoot_det, person_det, ball_det, ball_state,
                shoot_now=shoot_now, shoot_rise=shoot_rise,
                person_in_shoot=person_in_shoot, ball_in_shoot=ball_in_shoot,
                d_ball_person=float("inf"),
                left_shoot=False, sep_ok=False, release_signal=False,
                gate_reason="fail_ball_in_shoot",
                person_overlap_ratio=person_overlap_ratio,
                iou_person_shoot=iou_ps,
                ball_source=ball_source,
            )
            return None

        # Option A: ball close to PERSON BBOX (not person center)
        d_ball_person = self._dist_point_to_bbox(bx, by, person_bbox)
        if d_ball_person > self.ball_person_max_dist_px:
            self._update_last_debug(
                frame_idx, shoot_det, person_det, ball_det, ball_state,
                shoot_now=shoot_now, shoot_rise=shoot_rise,
                person_in_shoot=person_in_shoot, ball_in_shoot=ball_in_shoot,
                d_ball_person=d_ball_person,
                left_shoot=False, sep_ok=False, release_signal=False,
                gate_reason="fail_ball_close_person",
                person_overlap_ratio=person_overlap_ratio,
                iou_person_shoot=iou_ps,
                ball_source=ball_source,
            )
            return None

        # ARM
        # - Primary: shoot_rise
        # - Fallback: shoot_now stable for N frames (shoot_arm_stable_frames)
        #   (prevents missing arming when shoot is already active as gates become valid)
        arm_condition = shoot_rise or (shoot_now and self._shoot_streak >= self.shoot_arm_stable_frames)
        if arm_condition and self._armed_shoot_frame < -10**8:
            self._armed_shoot_frame = frame_idx
            self._armed_ball_pos = (bx, by)
            self._armed_person_pos = (pcx, pcy)
            self._armed_ball_person_dist = d_ball_person
            self._armed_shoot_conf = shoot_conf
            self._release_streak = 0

        if self._armed_shoot_frame < -10**8:
            self._update_last_debug(
                frame_idx, shoot_det, person_det, ball_det, ball_state,
                shoot_now=shoot_now, shoot_rise=shoot_rise,
                person_in_shoot=person_in_shoot, ball_in_shoot=ball_in_shoot,
                d_ball_person=d_ball_person,
                left_shoot=False, sep_ok=False, release_signal=False,
                gate_reason="not_armed_waiting_shoot_rise",
                person_overlap_ratio=person_overlap_ratio,
                iou_person_shoot=iou_ps,
                ball_source=ball_source,
            )
            return None

        # arming window expired
        if (frame_idx - self._armed_shoot_frame) > self.shoot_arm_window:
            self._disarm()
            self._update_last_debug(
                frame_idx, shoot_det, person_det, ball_det, ball_state,
                shoot_now=shoot_now, shoot_rise=shoot_rise,
                person_in_shoot=person_in_shoot, ball_in_shoot=ball_in_shoot,
                d_ball_person=d_ball_person,
                left_shoot=False, sep_ok=False, release_signal=False,
                gate_reason="arming_window_expired",
                person_overlap_ratio=person_overlap_ratio,
                iou_person_shoot=iou_ps,
                ball_source=ball_source,
            )
            return None

        # RELEASE
        left_shoot = not self._point_in_bbox(bx, by, shoot_bbox, pad=self.release_exit_pad_px)

        sep_ok = False
        if self._armed_ball_person_dist is not None:
            sep_ok = (d_ball_person - self._armed_ball_person_dist) >= self.release_sep_increase_px

        release_signal = left_shoot or sep_ok
        if release_signal:
            self._release_streak += 1
        else:
            self._release_streak = 0

        self._update_last_debug(
            frame_idx, shoot_det, person_det, ball_det, ball_state,
            shoot_now=shoot_now, shoot_rise=shoot_rise,
            person_in_shoot=person_in_shoot, ball_in_shoot=ball_in_shoot,
            d_ball_person=d_ball_person,
            left_shoot=left_shoot, sep_ok=sep_ok, release_signal=release_signal,
            gate_reason="armed_waiting_release",
            person_overlap_ratio=person_overlap_ratio,
            iou_person_shoot=iou_ps,
            ball_source=ball_source,
        )

        if self._release_streak < self.release_debounce_frames:
            return None

        # rim required
        if self._last_rim_pos is None:
            self._disarm()
            self._update_last_debug(
                frame_idx, shoot_det, person_det, ball_det, ball_state,
                shoot_now=shoot_now, shoot_rise=shoot_rise,
                person_in_shoot=person_in_shoot, ball_in_shoot=ball_in_shoot,
                d_ball_person=d_ball_person,
                left_shoot=left_shoot, sep_ok=sep_ok, release_signal=release_signal,
                gate_reason="release_but_no_rim",
                person_overlap_ratio=person_overlap_ratio,
                iou_person_shoot=iou_ps,
                ball_source=ball_source,
            )
            return None

        if (frame_idx - self._last_rim_frame) > self.rim_recent_frames:
            self._disarm()
            self._update_last_debug(
                frame_idx, shoot_det, person_det, ball_det, ball_state,
                shoot_now=shoot_now, shoot_rise=shoot_rise,
                person_in_shoot=person_in_shoot, ball_in_shoot=ball_in_shoot,
                d_ball_person=d_ball_person,
                left_shoot=left_shoot, sep_ok=sep_ok, release_signal=release_signal,
                gate_reason="release_but_rim_stale",
                person_overlap_ratio=person_overlap_ratio,
                iou_person_shoot=iou_ps,
                ball_source=ball_source,
            )
            return None

        rim_cx, rim_cy = self._last_rim_pos
        dist_ball_rim = self._dist((bx, by), (rim_cx, rim_cy))

        # TRIGGER attempt
        self._cooldown = self.cooldown_frames
        self._open_shot_window(frame_idx)
        self._shot_window_attempt_emitted = True

        self._disarm()
        self._start_post_attempt_locks()

        self._update_last_debug(
            frame_idx, shoot_det, person_det, ball_det, ball_state,
            shoot_now=shoot_now, shoot_rise=shoot_rise,
            person_in_shoot=person_in_shoot, ball_in_shoot=ball_in_shoot,
            d_ball_person=d_ball_person,
            left_shoot=left_shoot, sep_ok=sep_ok, release_signal=release_signal,
            gate_reason="ATTEMPT_TRIGGERED",
            person_overlap_ratio=person_overlap_ratio,
            iou_person_shoot=iou_ps,
            ball_source=ball_source,
        )

        if self.debug and (frame_idx % self.debug_every == 0):
            dbg = self.last_debug
            print(
                f"[AttemptDBG f={frame_idx}] gate={dbg.get('gate_reason')} "
                f"shoot_now={dbg.get('shoot_now')} conf={dbg.get('shoot_conf'):.2f} "
                f"p_in={dbg.get('person_in_shoot')} cover={dbg.get('person_overlap_ratio')} "
                f"b_in={dbg.get('ball_in_shoot')} src={dbg.get('ball_source')} "
                f"d_bp={dbg.get('d_ball_person'):.1f}/{dbg.get('ball_person_max_dist_px'):.0f} "
                f"armed={dbg.get('armed')} streak={dbg.get('release_streak')}/{dbg.get('release_debounce')} "
                f"left={dbg.get('left_shoot')} sep={dbg.get('sep_ok')}"
            )

        return AttemptEvent(
            frame_idx=frame_idx,
            ball_cx=bx,
            ball_cy=by,
            rim_cx=rim_cx,
            rim_cy=rim_cy,
            distance_px=dist_ball_rim,
            details=(
                f"shoot_release_strict("
                f"shoot_conf={shoot_conf:.2f}, "
                f"person_in_shoot={person_in_shoot}, ball_in_shoot={ball_in_shoot}, "
                f"d_ball_person_bbox={d_ball_person:.1f}, "
                f"left_shoot={left_shoot}, sep_ok={sep_ok}, "
                f"ball_src={ball_source}, "
                f"debounce={self._release_streak}/{self.release_debounce_frames})"
            ),
        )
