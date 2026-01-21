from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math

from src.track.ball_tracker import BallState


@dataclass
class MadeEvent:
    """Outcome decision produced by MadeDetector."""
    frame_idx: int
    outcome: str
    details: str = ""


class MadeDetector:
    """
    Outcome classifier (Made / Miss / Unknown) triggered after an AttemptEvent.

    High-level idea:
    - A shot outcome cannot be inferred reliably from a single frame.
    - We accumulate ball evidence over a bounded temporal window after the attempt.

    Main signals:
    1) Rim-plane crossing: ball trajectory passes from above to below a rim reference line.
    2) Center-trajectory evidence: the ball x-position comes close to rim center at least once.
       This is robust to missing detections at the exact crossing frame.
    3) Below-rim confirmation: multiple consecutive ball points clearly below the rim.

    Engineering assumptions:
    - Rim is approximately stable; we optionally consume a stabilized rim bbox/center.
    - Ball state is provided by BallTracker (smoothing + occlusion handling).
    """

    def __init__(
        self,
        window_frames: int = 90,
        max_window_frames: int = 210,
        near_rim_grace_frames: int = 18,

        below_confirm_frames: int = 3,
        below_rim_rel_y: float = 0.86,

        y_epsilon_px: float = 3.0,
        rim_line_rel_y: float = 0.28,

        plane_pass_segments: int = 18,
        max_cross_gap_frames: int = 12,

        gate_rx_rel: float = 0.46,
        gate_ry_rel: float = 0.18,
        gate_y_slack_px: float = 2.0,

        require_near_rim_for_plane_pass: bool = False,
        near_rim_dist_px: float = 155.0,
        rim_expand_factor: float = 1.20,

        min_points_for_decision: int = 4,
        fallback_extra_y_epsilon_px: float = 6.0,

        below_gate_confirm_frames: int = 2,
        below_gate_window: int = 10,
        below_gate_radius_rel: float = 0.55,
        below_gate_min_px: float = 22.0,

        allow_postgap_plane: bool = True,

        center_gate_radius_rel: float = 0.35,
        center_gate_min_px: float = 18.0,
        center_gate_window: int = 6,
        center_gate_min_hits: int = 1,

        far_rim_dist_px: float = 260.0,
        far_rim_confirm_frames: int = 6,
    ):
        # Windowing / timeouts
        self.window_frames = int(window_frames)
        self.max_window_frames = int(max_window_frames)
        self.near_rim_grace_frames = int(near_rim_grace_frames)

        # Confirmation thresholds
        self.below_confirm_frames = int(below_confirm_frames)
        self.below_rim_rel_y = float(below_rim_rel_y)

        # Rim-plane definition (y_line) and crossing tolerance
        self.y_epsilon_px = float(y_epsilon_px)
        self.rim_line_rel_y = float(rim_line_rel_y)

        # Parameters kept for compatibility with earlier iterations
        self.plane_pass_segments = int(plane_pass_segments)
        self.max_cross_gap_frames = int(max_cross_gap_frames)

        # Rim gate for validating the crossing x-position
        self.gate_rx_rel = float(gate_rx_rel)
        self.gate_ry_rel = float(gate_ry_rel)
        self.gate_y_slack_px = float(gate_y_slack_px)

        # "Near rim" notion used for grace windows and optional gating
        self.require_near_rim_for_plane_pass = bool(require_near_rim_for_plane_pass)
        self.near_rim_dist_px = float(near_rim_dist_px)
        self.rim_expand_factor = float(rim_expand_factor)

        self.min_points_for_decision = int(min_points_for_decision)
        self.fallback_extra_y_epsilon_px = float(fallback_extra_y_epsilon_px)

        # Additional “swish-like” evidence under rim (multiple points)
        self.below_gate_confirm_frames = int(below_gate_confirm_frames)
        self.below_gate_window = int(below_gate_window)
        self.below_gate_radius_rel = float(below_gate_radius_rel)
        self.below_gate_min_px = float(below_gate_min_px)

        self.allow_postgap_plane = bool(allow_postgap_plane)

        # Center trajectory evidence (robust to crossing-frame dropouts)
        self.center_gate_radius_rel = float(center_gate_radius_rel)
        self.center_gate_min_px = float(center_gate_min_px)
        self.center_gate_window = int(center_gate_window)
        self.center_gate_min_hits = int(center_gate_min_hits)

        # Early-miss heuristic: ball goes far away after approaching the rim
        self.far_rim_dist_px = float(far_rim_dist_px)
        self.far_rim_confirm_frames = int(far_rim_confirm_frames)

        # Internal state (one active attempt at a time)
        self.active: bool = False
        self._start_frame: int = -10**9

        self._rim_cx: float = 0.0
        self._rim_cy: float = 0.0
        self._rim_bbox: Optional[Tuple[float, float, float, float]] = None

        self._y_line: float = 0.0
        self._y_line_from_bbox: bool = False

        # Stored ball points: (frame_idx, x, y)
        self._pts: List[Tuple[int, float, float]] = []

        self._came_close_to_rim: bool = False

        self._passed_rim_plane: bool = False
        self._pass_details: str = ""
        self._pass_frame: int = -10**9

        self._last_ball_frame: int = -10**9
        self._last_near_rim_frame: int = -10**9

        self._below_rim_confirm: int = 0
        self._min_dist_to_rim: float = float("inf")

        # Center evidence is “ever seen” during the attempt, not only at crossing.
        self._center_pass_hits: int = 0
        self._ever_center_evidence: bool = False

        # Direction check: we require the ball to have been above rim center at least once.
        self._ever_above_rim_center: bool = False

        # Early-miss bookkeeping
        self._far_rim_count = 0

        # Descent detection (image y increases downward): used to avoid deciding too early.
        self._ever_descended = False
        self._last_cy: Optional[float] = None
        self._has_descended = False

        # DEBUG visualization points (used only by draw_made_debug)
        self._dbg_center_hits_pts: List[Tuple[float, float]] = []
        self._dbg_below_hits_pts: List[Tuple[float, float]] = []
        self._dbg_plane_cross_pt: Optional[Tuple[float, float]] = None

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _pick_rim(detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        rims = [d for d in detections if str(d.get("name", "")).lower() == "rim"]
        if not rims:
            return None
        return max(rims, key=lambda d: float(d.get("conf", 0.0)))

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _expanded_rim_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Slightly enlarge rim bbox to be more tolerant to jitter/box tightness."""
        if self._rim_bbox is None:
            return None
        x1, y1, x2, y2 = self._rim_bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, (x2 - x1)) * self.rim_expand_factor
        h = max(1.0, (y2 - y1)) * self.rim_expand_factor
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    @staticmethod
    def _point_in_bbox(x: float, y: float, bbox: Tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = bbox
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    def _compute_y_line(self) -> Tuple[float, bool]:
        """
        Compute the rim plane y-coordinate.

        If a rim bbox is available, y_line is placed at a relative vertical position
        inside the bbox. Otherwise we fall back to rim_cy and increase tolerance later.
        """
        if self._rim_bbox is None:
            return self._rim_cy, False
        x1, y1, x2, y2 = self._rim_bbox
        h = max(1.0, (y2 - y1))
        rel = min(0.60, max(0.05, self.rim_line_rel_y))
        return (y1 + rel * h), True

    def _compute_below_rim_line(self) -> Optional[float]:
        """Secondary line used to confirm the ball is clearly below the rim."""
        if self._rim_bbox is None:
            return None
        x1, y1, x2, y2 = self._rim_bbox
        h = max(1.0, (y2 - y1))
        rel = min(0.98, max(0.40, self.below_rim_rel_y))
        return (y1 + rel * h)

    def _rim_size(self) -> Optional[Tuple[float, float]]:
        if self._rim_bbox is None:
            return None
        x1, y1, x2, y2 = self._rim_bbox
        return (max(1.0, x2 - x1), max(1.0, y2 - y1))

    def _center_x_gate_thr(self) -> Optional[float]:
        """Horizontal distance threshold (in px) for center trajectory evidence."""
        size = self._rim_size()
        if size is None:
            return None
        rim_w, _ = size
        return max(self.center_gate_min_px, self.center_gate_radius_rel * rim_w)

    def _reset(self) -> None:
        """Reset the state at the end of an attempt (decision made or cancelled)."""
        self.active = False
        self._start_frame = -10**9
        self._pts.clear()
        self._center_pass_hits = 0
        self._ever_center_evidence = False
        self._dbg_center_hits_pts.clear()
        self._dbg_below_hits_pts.clear()
        self._dbg_plane_cross_pt = None
        self._ever_above_rim_center = False
        self._far_rim_count = 0
        self._ever_descended = False
        self._last_cy = None
        self._has_descended = False

    def _near_rim_now(self, cx: float, cy: float) -> bool:
        """Near-rim test used for grace window logic and early-miss heuristic."""
        d = self._dist((cx, cy), (self._rim_cx, self._rim_cy))
        if d <= self.near_rim_dist_px:
            return True
        exp = self._expanded_rim_bbox()
        if exp is not None and self._point_in_bbox(cx, cy, exp):
            return True
        return False

    def _rim_gate_radii(self) -> Optional[Tuple[float, float]]:
        """
        Elliptical gate radii around the rim center for validating plane-crossing x.
        This prevents declaring a “pass” far from the hoop.
        """
        if self._rim_bbox is None:
            return None
        x1, y1, x2, y2 = self._rim_bbox
        rim_w = max(1.0, x2 - x1)
        rim_h = max(1.0, y2 - y1)
        rx = max(6.0, self.gate_rx_rel * rim_w)
        ry = max(3.0, self.gate_ry_rel * rim_h)
        return rx, ry

    def _inside_rim_gate(self, x: float, y: float) -> bool:
        """Validate that a rim-plane crossing occurs close enough to the rim center."""
        radii = self._rim_gate_radii()
        if radii is None:
            return (abs(x - self._rim_cx) <= 65.0) and (abs(y - self._y_line) <= (self.y_epsilon_px + 3.0))
        rx, ry = radii
        dx = (x - self._rim_cx) / rx
        dy = (y - self._y_line) / ry
        return (dx * dx + dy * dy) <= 1.0

    # -----------------------------
    # Main API
    # -----------------------------
    def update(
        self,
        frame_idx: int,
        detections: List[Dict[str, Any]],
        ball_state: Optional[BallState],
        new_attempt=None,
        *,
        rim_stable_bbox: Optional[Tuple[float, float, float, float]] = None,
        rim_stable_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[MadeEvent]:
        """
        Update outcome state machine.

        Inputs:
        - new_attempt: starts a new outcome window (one attempt at a time)
        - ball_state: ball evidence each frame (can be None during occlusions)
        - rim_stable_bbox/center: optional stabilized rim reference (preferred)

        Output:
        - MadeEvent when the module commits to an outcome, else None.
        """

        # If a new attempt starts while one is still active, force-close the previous one.
        # This is a pragmatic rule to avoid overlapping windows.
        if new_attempt is not None and self.active:
            out = MadeEvent(frame_idx=frame_idx, outcome="miss", details="forced_miss_new_attempt")
            self._reset()
            return out

        # Start a new attempt window.
        if new_attempt is not None:
            self._reset()
            self.active = True
            self._start_frame = int(frame_idx)

            self._rim_cx = float(getattr(new_attempt, "rim_cx"))
            self._rim_cy = float(getattr(new_attempt, "rim_cy"))

            if rim_stable_center is not None:
                self._rim_cx, self._rim_cy = map(float, rim_stable_center)

            if rim_stable_bbox is not None:
                self._rim_bbox = tuple(map(float, rim_stable_bbox))
            else:
                rim_det = self._pick_rim(detections)
                if rim_det is not None:
                    self._rim_bbox = (
                        float(rim_det["x1"]),
                        float(rim_det["y1"]),
                        float(rim_det["x2"]),
                        float(rim_det["y2"]),
                    )

            self._y_line, self._y_line_from_bbox = self._compute_y_line()

        if not self.active:
            return None

        # Keep rim reference up to date when stabilized data is provided.
        if rim_stable_center is not None:
            self._rim_cx, self._rim_cy = map(float, rim_stable_center)

        if rim_stable_bbox is not None:
            self._rim_bbox = tuple(map(float, rim_stable_bbox))
            self._y_line, self._y_line_from_bbox = self._compute_y_line()

        # Accumulate ball evidence.
        if ball_state is not None:
            cx, cy = float(ball_state.cx), float(ball_state.cy)
            self._pts.append((frame_idx, cx, cy))
            self._last_ball_frame = frame_idx

            # Descent detection (used as “apex passed” guard).
            # Note: y increases downward in images.
            if self._last_cy is not None:
                if cy > self._last_cy + 1.5:
                    self._has_descended = True
            self._last_cy = cy

            # Additional descent bookkeeping (kept for compatibility with current behavior).
            if self._last_cy is not None:
                if cy > self._last_cy + 1.0:
                    self._ever_descended = True
            self._last_cy = cy

            # Center evidence: ball x near rim center at least once in the window.
            x_thr = self._center_x_gate_thr()
            if x_thr is not None and abs(cx - self._rim_cx) <= x_thr:
                self._ever_center_evidence = True
                self._dbg_center_hits_pts.append((cx, cy))

            # Below-rim evidence: require consecutive frames below a line.
            y_below = self._compute_below_rim_line()
            if y_below is not None:
                if cy >= y_below:
                    self._below_rim_confirm += 1
                    self._dbg_below_hits_pts.append((cx, cy))
                else:
                    self._below_rim_confirm = 0

            # Direction sanity: ball was above rim center at least once.
            if cy <= (self._rim_cy - 2.0):
                self._ever_above_rim_center = True

            # Track if ball ever came close to rim (used for grace windows).
            if self._near_rim_now(cx, cy):
                self._came_close_to_rim = True
                self._last_near_rim_frame = frame_idx

            # Early miss: if the ball went near the rim, then moved far away for long enough
            # (and we have passed the apex), declare miss. This prevents waiting for timeout
            # on obvious long rebounds.
            dist = self._dist((cx, cy), (self._rim_cx, self._rim_cy))
            if self._came_close_to_rim and not self._passed_rim_plane and dist >= self.far_rim_dist_px:
                self._far_rim_count += 1
            else:
                self._far_rim_count = 0

            if self._has_descended and self._far_rim_count >= self.far_rim_confirm_frames:
                if not self._ever_descended:
                    # If descent evidence is not strong, avoid committing.
                    return None
                out = MadeEvent(frame_idx=frame_idx, outcome="miss", details=f"early_miss_far_rim(dist={dist:.1f})")
                self._reset()
                return out

        # Detect rim-plane crossing once: above -> below y_line with a tolerance,
        # and validate the crossing x using the rim gate.
        if len(self._pts) >= 2 and not self._passed_rim_plane:
            y_line = self._y_line
            eps = self.y_epsilon_px + (0.0 if self._y_line_from_bbox else self.fallback_extra_y_epsilon_px)

            for (_, x0, y0), (_, x1, y1) in zip(self._pts[:-1], self._pts[1:]):
                if (y0 <= y_line - eps) and (y1 >= y_line + eps):
                    t = (y_line - y0) / max(1e-6, (y1 - y0))
                    x_cross = x0 + t * (x1 - x0)
                    if self._inside_rim_gate(x_cross, y_line):
                        self._passed_rim_plane = True
                        self._pass_frame = frame_idx
                        self._dbg_plane_cross_pt = (x_cross, y_line)
                        self._pass_details = f"pass_plane(x={x_cross:.1f})"
                        break

        # Final MADE rule: require descent, plane-cross, center evidence,
        # “was above rim” evidence, and multiple below-rim confirmations.
        if (
            self._has_descended
            and self._passed_rim_plane
            and self._ever_center_evidence
            and self._ever_above_rim_center
            and self._below_rim_confirm >= self.below_confirm_frames
        ):
            out = MadeEvent(
                frame_idx=frame_idx,
                outcome="made",
                details=f"made(center_ever={self._ever_center_evidence}, below={self._below_rim_confirm})",
            )
            self._reset()
            return out

        # Timeout -> MISS (only once the ball has started descending).
        # Grace windows avoid deciding too early when the ball is still near rim
        # or detections temporarily drop around the hoop.
        elapsed = frame_idx - self._start_frame
        if elapsed >= self.window_frames and self._has_descended:
            if not self._ever_descended:
                return None
            if elapsed < self.max_window_frames:
                if frame_idx - self._last_near_rim_frame <= self.near_rim_grace_frames:
                    return None
                if frame_idx - self._last_ball_frame <= self.near_rim_grace_frames:
                    return None

            out = MadeEvent(
                frame_idx=frame_idx,
                outcome="miss",
                details=f"timeout_miss(elapsed={elapsed}, center_ever={self._ever_center_evidence})",
            )
            self._reset()
            return out

        return None
