from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math

from src.track.ball_tracker import BallState


@dataclass
class AttemptEvent:
    """
    Event emitted when a shot attempt is detected.
    We store ball/rim positions (at trigger time) so downstream logic (made/miss/airball)
    can use them even if detections disappear in the next frames.
    """
    frame_idx: int
    ball_cx: float
    ball_cy: float
    rim_cx: float
    rim_cy: float
    distance_px: float
    details: str = ""  # debug reason: "ball_near_rim", "shoot_release", etc.


class AttemptDetector:
    """
    Detect shot attempts.

    Two complementary triggers:

    1) Normal trigger (ball-based):
       - rim detected recently
       - ball tracked (visible)
       - ball enters radius around rim (enter_radius_px)
       - ball is descending (vy >= vy_min)
       - optional: ball is approaching the rim (distance decreasing)

    2) Fallback trigger (shoot-based) to handle ball disappearing:
       - detect class "shoot" (pose/gesture) with decent confidence
       - use FRONT MONTANT ("shoot_rise") to ARM once per shot
       - then wait for a "release" signal (ball moved / vy / approach) before emitting AttemptEvent
       - rim must still be recent (avoid using stale rim)

    Why not trigger immediately on "shoot_rise"?
      Because "shoot" can start while the ball is still in the hands.
      We arm on shoot_rise, and trigger only when the ball actually leaves the hand.

    PATCH (shot window):
      - When we detect a valid "shoot_release", we open a shot window (N frames).
      - During this window, we allow at most ONE AttemptEvent (prevents rebound double-attempts).
      - We also disallow ball-based attempts when the window is active BUT not "open" yet (no release).
    """

    def __init__(
        self,
        enter_radius_px: float = 85.0,
        vy_min: float = 0.2,
        cooldown_frames: int = 12,          # shorter is better when using shoot arming
        require_approach: bool = True,
        approach_window: int = 6,
        # --- shoot fallback ---
        shoot_conf_min: float = 0.18,       # "shoot is present" threshold
        shoot_conf_strong: float = 0.40,    # if shoot_conf >= this, arm even if ball not recent
        ball_recent_frames: int = 25,       # how long ball memory is valid for shoot arming
        rim_recent_frames: int = 15,        # avoid using stale rim
        # --- shoot arming / release gate ---
        shoot_arm_window: int = 20,         # frames to wait after shoot_rise for "release"
        shoot_arm_min_ball_move: float = 18.0,  # px: ball must move from arming position
        shoot_arm_min_vy: float = -0.2,     # allow slight upward right after release (vy can be negative)
        # --- anti rebound / anti double-attempt near rim ---
        ignore_shoot_if_ball_near_rim_factor: float = 1.1,  # ignore shoot arming if ball already near rim
        reset_arming_if_ball_near_rim: bool = True,         # disarm if ball gets near rim (rebounds etc.)
        # --- NEW: block new attempts until ball goes BELOW rim (anti-rebound) ---
        require_ball_below_rim_to_rearm: bool = True,
        below_margin_px: float = 35.0,      # ball must be below rim_cy + margin to allow next attempt
        below_confirm_frames: int = 3,      # NEW: must be below for N consecutive frames (debounce)
        # --- NEW: shot window (1 attempt per release) ---
        shot_window_frames: int = 120,      # ~4s @30fps. Prevents rebound re-attempt on same shot.
        # --- (optional) compatibility knobs if your main passes them ---
        post_attempt_block_until_ball_far: bool = False,
        post_attempt_far_factor: float = 1.6,
        post_attempt_min_frames: int = 10,
    ):
        # Normal (ball/rim) params
        self.enter_radius_px = float(enter_radius_px)
        self.vy_min = float(vy_min)
        self.cooldown_frames = int(cooldown_frames)
        self.require_approach = bool(require_approach)
        self.approach_window = int(approach_window)

        # Shoot fallback params
        self.shoot_conf_min = float(shoot_conf_min)
        self.shoot_conf_strong = float(shoot_conf_strong)
        self.ball_recent_frames = int(ball_recent_frames)
        self.rim_recent_frames = int(rim_recent_frames)

        # Shoot arming / release gate params
        self.shoot_arm_window = int(shoot_arm_window)
        self.shoot_arm_min_ball_move = float(shoot_arm_min_ball_move)
        self.shoot_arm_min_vy = float(shoot_arm_min_vy)

        # Anti-rebound params (prevents arming during rim rebounds)
        self.ignore_shoot_if_ball_near_rim_factor = float(ignore_shoot_if_ball_near_rim_factor)
        self.reset_arming_if_ball_near_rim = bool(reset_arming_if_ball_near_rim)

        # NEW: require ball below rim to rearm attempts
        self.require_ball_below_rim_to_rearm = bool(require_ball_below_rim_to_rearm)
        self.below_margin_px = float(below_margin_px)
        self.below_confirm_frames = int(below_confirm_frames)

        # NEW: shot window (one attempt per release)
        self.shot_window_frames = int(shot_window_frames)

        # Optional compatibility knobs (not strictly needed if you use below-rim lock)
        self.post_attempt_block_until_ball_far = bool(post_attempt_block_until_ball_far)
        self.post_attempt_far_factor = float(post_attempt_far_factor)
        self.post_attempt_min_frames = int(post_attempt_min_frames)

        # Cooldown state (prevents double-counting)
        self._cooldown = 0
        self._last_attempt_frame = -10**9

        # Distance history to validate "approach toward rim"
        self._dist_hist: List[Tuple[int, float]] = []  # (frame_idx, distance_px)

        # --- Memories (for robustness when detections vanish) ---
        self._last_ball_frame: int = -10**9
        self._last_ball_pos: Optional[Tuple[float, float]] = None

        self._last_rim_frame: int = -10**9
        self._last_rim_pos: Optional[Tuple[float, float]] = None

        # Shoot state for "front montant" detection
        self._shoot_active: bool = False
        self._last_shoot_frame: int = -10**9
        self._last_shoot_conf: float = 0.0

        # Shoot arming state (do NOT emit attempt immediately on shoot_rise)
        self._armed_shoot_frame: int = -10**9
        self._armed_shoot_ball_pos: Optional[Tuple[float, float]] = None
        self._armed_shoot_conf: float = 0.0

        # NEW: lock until ball is BELOW rim (anti-rebound)
        self._waiting_ball_below_rim: bool = False
        self._below_rim_streak: int = 0  # NEW: consecutive frames below rim to unlock

        # Optional: post-attempt block by distance (kept for compatibility)
        self._post_attempt_block = 0
        self._waiting_ball_far = False

        # NEW: shot window state
        self._shot_window_end: int = -10**9
        self._shot_window_active: bool = False
        self._shot_window_attempt_emitted: bool = False

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _center(det: Dict[str, Any]) -> Tuple[float, float]:
        """BBox center."""
        cx = (float(det["x1"]) + float(det["x2"])) / 2.0
        cy = (float(det["y1"]) + float(det["y2"])) / 2.0
        return cx, cy

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Euclidean distance in pixels."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _pick_rim(self, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Pick best rim detection (highest conf)."""
        rims = [d for d in detections if str(d.get("name", "")).lower() == "rim"]
        if not rims:
            return None
        return max(rims, key=lambda d: float(d.get("conf", 0.0)))

    def _max_conf(self, detections: List[Dict[str, Any]], cls_name: str) -> float:
        """Return maximum confidence for a given class name in detections."""
        cls_name = cls_name.lower()
        confs = [
            float(d.get("conf", 0.0))
            for d in detections
            if str(d.get("name", "")).lower() == cls_name
        ]
        return max(confs) if confs else 0.0

    def _is_approaching(self) -> bool:
        """
        True if distance has been decreasing overall in the last approach_window points.
        This filters some false attempts when the ball is drifting away.
        """
        if len(self._dist_hist) < self.approach_window:
            return True  # not enough history -> don't block

        last = self._dist_hist[-self.approach_window:]
        d0 = last[0][1]
        d1 = last[-1][1]
        return d1 < (d0 - 2.0)  # allow a bit of noise

    def _disarm(self) -> None:
        """Reset shoot arming state."""
        self._armed_shoot_frame = -10**9
        self._armed_shoot_ball_pos = None
        self._armed_shoot_conf = 0.0

    def _start_post_attempt_locks(self) -> None:
        """
        After an attempt:
          - lock until ball goes BELOW rim (best anti-rebound)
          - optional additional lock until ball is FAR (compat)
        """
        if self.require_ball_below_rim_to_rearm:
            self._waiting_ball_below_rim = True
            self._below_rim_streak = 0  # NEW: must rebuild streak from 0

        if self.post_attempt_block_until_ball_far:
            self._post_attempt_block = max(self._post_attempt_block, self.post_attempt_min_frames)
            self._waiting_ball_far = True

    def _open_shot_window(self, frame_idx: int) -> None:
        """
        Open a time window after a valid shoot_release during which we allow ONLY ONE attempt.
        This prevents rebounds near the rim from being counted as extra attempts.
        """
        self._shot_window_active = True
        self._shot_window_end = frame_idx + self.shot_window_frames
        self._shot_window_attempt_emitted = False

    def _shot_window_update(self, frame_idx: int) -> None:
        """Expire shot window when time is up."""
        if self._shot_window_active and frame_idx > self._shot_window_end:
            self._shot_window_active = False
            self._shot_window_attempt_emitted = False

    # -----------------------------
    # Main API
    # -----------------------------
    def update(
        self,
        frame_idx: int,
        detections: List[Dict[str, Any]],
        ball_state: Optional[BallState],
    ) -> Optional[AttemptEvent]:
        """
        Call once per frame.

        Returns:
            AttemptEvent if a new attempt is detected, otherwise None.
        """

        # ----------------------------------------------------------
        # 0) Cooldown update
        # ----------------------------------------------------------
        if self._cooldown > 0:
            self._cooldown -= 1
        in_cooldown = (self._cooldown != 0)

        # Optional: post-attempt distance lock countdown (compat)
        if self._post_attempt_block > 0:
            self._post_attempt_block -= 1

        # NEW: update/expire shot window
        self._shot_window_update(frame_idx)

        # ----------------------------------------------------------
        # 1) Update memories: rim, ball, shoot
        # ----------------------------------------------------------
        rim_det = self._pick_rim(detections)
        if rim_det is not None:
            self._last_rim_pos = self._center(rim_det)
            self._last_rim_frame = frame_idx

        if ball_state is not None:
            self._last_ball_frame = frame_idx
            self._last_ball_pos = (ball_state.cx, ball_state.cy)

        shoot_conf = self._max_conf(detections, "shoot")
        shoot_now = shoot_conf >= self.shoot_conf_min
        shoot_rise = shoot_now and not self._shoot_active
        self._shoot_active = shoot_now

        if shoot_now:
            self._last_shoot_frame = frame_idx
            self._last_shoot_conf = shoot_conf

        # ----------------------------------------------------------
        # 2) Need a recent rim for any attempt (avoid stale triggers)
        # ----------------------------------------------------------
        if self._last_rim_pos is None:
            return None
        if (frame_idx - self._last_rim_frame) > self.rim_recent_frames:
            return None

        rim_cx, rim_cy = self._last_rim_pos
        rim_pos = (rim_cx, rim_cy)

        # ----------------------------------------------------------
        # 3) Compute ball near rim + manage post-attempt locks
        # ----------------------------------------------------------
        # NOTE:
        #   - ball_near_rim uses last_ball_pos (memory) because it is useful even when ball_state disappears briefly.
        #   - BUT the BELOW-RIM UNLOCK uses *ball_state only* (stronger signal, avoids memory glitches).
        dist_ball_rim = float("inf")
        ball_near_rim = False
        if self._last_ball_pos is not None:
            bx, by = self._last_ball_pos
            dist_ball_rim = self._dist((bx, by), rim_pos)
            ball_near_rim = dist_ball_rim <= (self.enter_radius_px * self.ignore_shoot_if_ball_near_rim_factor)

        # NEW: hard lock — do NOT allow any new attempt until ball is confirmed below rim
        if self._waiting_ball_below_rim:
            below_now = False
            if ball_state is not None:
                # "below rim" in image coordinates means y is greater (downwards)
                below_now = ball_state.cy >= (rim_cy + self.below_margin_px)

            if below_now:
                self._below_rim_streak += 1
            else:
                self._below_rim_streak = 0

            # Unlock only if we have N consecutive frames below rim (debounce)
            if self._below_rim_streak >= self.below_confirm_frames:
                self._waiting_ball_below_rim = False
                self._below_rim_streak = 0
            else:
                # Keep updating arming state in background if you want,
                # but NEVER emit attempts while we are waiting ball below rim.
                # Also disarm to avoid “release” happening during rebounds.
                self._disarm()
                return None

        # Optional: distance lock (compat)
        if self.post_attempt_block_until_ball_far and self._waiting_ball_far:
            far_thr = self.enter_radius_px * self.post_attempt_far_factor

            if (self._last_ball_pos is not None) and (dist_ball_rim >= far_thr):
                self._waiting_ball_far = False

            if self._waiting_ball_far and (self._post_attempt_block > 0 or ball_near_rim):
                self._disarm()
                return None

        # ----------------------------------------------------------
        # Anti-rebond: si la balle est déjà près du rim, ignorer shoot_rise
        # ----------------------------------------------------------
        # (This prevents "shoot" detections on rebounds from arming a new shot.)
        if ball_near_rim:
            shoot_rise = False
            if self.reset_arming_if_ball_near_rim:
                self._disarm()

        # ----------------------------------------------------------
        # 4) If in cooldown, we do not trigger new attempts
        # ----------------------------------------------------------
        if in_cooldown:
            return None

        # ----------------------------------------------------------
        # NEW: shot window rule — if we already emitted an attempt for this shot, block any new one
        # ----------------------------------------------------------
        if self._shot_window_active and self._shot_window_attempt_emitted:
            # No second attempt on the same shot (rebounds, rim bounces, tracker glitches)
            self._disarm()
            return None

        # ==========================================================
        # A) SHOOT-BASED PATH (ARM THEN RELEASE)
        #    - Arm on shoot_rise
        #    - Emit AttemptEvent only when the ball "releases" (moves/vy/approach) within a window
        # ==========================================================
        if shoot_rise:
            ball_recent = (frame_idx - self._last_ball_frame) <= self.ball_recent_frames
            strong_shoot = shoot_conf >= self.shoot_conf_strong

            if ball_recent or strong_shoot:
                # Arm the shot (do NOT trigger attempt immediately)
                self._armed_shoot_frame = frame_idx
                self._armed_shoot_ball_pos = self._last_ball_pos  # may be None
                self._armed_shoot_conf = shoot_conf
                # Reset dist history to avoid "approach" using old trajectory
                self._dist_hist.clear()

        # If a shot is armed, wait for a "release" signal within shoot_arm_window
        armed_dt = frame_idx - self._armed_shoot_frame
        if 0 <= armed_dt <= self.shoot_arm_window:
            if ball_state is not None and self._armed_shoot_ball_pos is not None:
                bx0, by0 = self._armed_shoot_ball_pos
                moved = self._dist((ball_state.cx, ball_state.cy), (bx0, by0))
                dist_now = self._dist((ball_state.cx, ball_state.cy), rim_pos)

                # Keep a short dist history so "approach" works right after release
                self._dist_hist.append((frame_idx, dist_now))
                if len(self._dist_hist) > 50:
                    self._dist_hist = self._dist_hist[-50:]

                approaching = self._is_approaching() if self.require_approach else True

                # Release heuristics:
                # - ball moved enough from the arming position
                # - OR ball vertical speed suggests release (vy threshold is permissive)
                release_by_motion = moved >= self.shoot_arm_min_ball_move
                release_by_vy = ball_state.vy >= self.shoot_arm_min_vy

                # We do NOT require "approaching" here because airballs can go away from rim.
                if release_by_motion or release_by_vy:
                    self._cooldown = self.cooldown_frames
                    self._last_attempt_frame = frame_idx

                    # NEW: open shot window (so rebounds won't create a new attempt)
                    self._open_shot_window(frame_idx)
                    self._shot_window_attempt_emitted = True

                    # Disarm after triggering + start post-attempt locks (anti-rebound)
                    self._disarm()
                    self._start_post_attempt_locks()

                    return AttemptEvent(
                        frame_idx=frame_idx,
                        ball_cx=ball_state.cx,
                        ball_cy=ball_state.cy,
                        rim_cx=rim_cx,
                        rim_cy=rim_cy,
                        distance_px=dist_now,
                        details=(
                            f"shoot_release (armed_conf={self._armed_shoot_conf:.2f}, moved={moved:.1f}, "
                            f"vy={ball_state.vy:.2f}, approaching={approaching})"
                        ),
                    )

        # If arming window expired, disarm (avoid stale armed shots)
        if armed_dt > self.shoot_arm_window:
            self._disarm()

        # ==========================================================
        # B) NORMAL PATH: attempt by ball entering rim zone
        # ==========================================================
        # IMPORTANT with shot window:
        #   - If a shot window is active, we only allow at most ONE attempt in that window.
        #   - If you rely on shoot_release for counting attempts, you can also choose to block this path
        #     when shot window is active. Here we allow it ONLY if window is active and no attempt emitted yet,
        #     OR if there is no shot window (no shoot class).
        allow_ball_based = True
        if self._shot_window_active and self._shot_window_attempt_emitted:
            allow_ball_based = False

        if allow_ball_based and ball_state is not None:
            ball_pos = (ball_state.cx, ball_state.cy)
            dist = self._dist(ball_pos, rim_pos)

            # Update history for approach check
            self._dist_hist.append((frame_idx, dist))
            if len(self._dist_hist) > 50:
                self._dist_hist = self._dist_hist[-50:]

            near_rim = dist <= self.enter_radius_px
            descending = ball_state.vy >= self.vy_min  # +y is down in image coordinates
            approaching = self._is_approaching() if self.require_approach else True

            if near_rim and descending and approaching:
                self._cooldown = self.cooldown_frames
                self._last_attempt_frame = frame_idx

                # NEW: if we didn't see a shoot_release, open a window anyway (ball-based shots)
                if not self._shot_window_active:
                    self._open_shot_window(frame_idx)
                self._shot_window_attempt_emitted = True

                # Disarm if we were armed (ball-based attempt is enough)
                self._disarm()

                # Start post-attempt locks (anti-rebound)
                self._start_post_attempt_locks()

                return AttemptEvent(
                    frame_idx=frame_idx,
                    ball_cx=ball_state.cx,
                    ball_cy=ball_state.cy,
                    rim_cx=rim_cx,
                    rim_cy=rim_cy,
                    distance_px=dist,
                    details="ball_near_rim (descending+approach)",
                )

        return None
