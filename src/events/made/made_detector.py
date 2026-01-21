from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math

from src.track.ball_tracker import BallState

from .made_types import MadeEvent
from .made_context import (
    BBox,
    build_context,
    center_x_gate_thr,
    rim_gate_radii,
)
from .made_gates import (
    near_rim_now,
    plane_crossing_point,
    inside_rim_gate,
)
from .made_debug import MadeDebug


class MadeDetector:
    """
    Outcome classifier triggered after an AttemptEvent.

    Engineering intent:
    - The detector does not infer outcomes from a single frame.
    - It accumulates ball evidence in a bounded window after the attempt start.

    Main evidence (conceptually):
    1) Rim-plane crossing (trajectory goes from above to below a reference rim line)
    2) Center evidence (ball x close to rim center at least once in the window)
    3) Below-rim confirmation (several consecutive frames clearly below the rim)
    plus guards:
    - descent/apex passed (avoid early decisions)
    - direction sanity (ball was above rim center at least once)
    - grace windows around the hoop (tolerate occlusions/noisy detections)
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

        # Rim-plane crossing tolerance
        self.y_epsilon_px = float(y_epsilon_px)
        self.rim_line_rel_y = float(rim_line_rel_y)
        self.fallback_extra_y_epsilon_px = float(fallback_extra_y_epsilon_px)

        # Compatibility parameters (kept to avoid breaking older configs)
        self.plane_pass_segments = int(plane_pass_segments)
        self.max_cross_gap_frames = int(max_cross_gap_frames)
        self.min_points_for_decision = int(min_points_for_decision)
        self.allow_postgap_plane = bool(allow_postgap_plane)

        # Crossing gate radii
        self.gate_rx_rel = float(gate_rx_rel)
        self.gate_ry_rel = float(gate_ry_rel)
        self.gate_y_slack_px = float(gate_y_slack_px)

        # Near-rim logic
        self.require_near_rim_for_plane_pass = bool(require_near_rim_for_plane_pass)
        self.near_rim_dist_px = float(near_rim_dist_px)
        self.rim_expand_factor = float(rim_expand_factor)

        # Extra “under rim” params (currently not used in your final decision rule,
        # but kept for compatibility / future experiments)
        self.below_gate_confirm_frames = int(below_gate_confirm_frames)
        self.below_gate_window = int(below_gate_window)
        self.below_gate_radius_rel = float(below_gate_radius_rel)
        self.below_gate_min_px = float(below_gate_min_px)

        # Center evidence params
        self.center_gate_radius_rel = float(center_gate_radius_rel)
        self.center_gate_min_px = float(center_gate_min_px)
        self.center_gate_window = int(center_gate_window)
        self.center_gate_min_hits = int(center_gate_min_hits)

        # Early miss heuristic
        self.far_rim_dist_px = float(far_rim_dist_px)
        self.far_rim_confirm_frames = int(far_rim_confirm_frames)

        # State: one active attempt at a time
        self.active: bool = False
        self._start_frame: int = -10**9

        self._rim_cx: float = 0.0
        self._rim_cy: float = 0.0
        self._rim_bbox: Optional[BBox] = None

        # Ball points history (frame_idx, x, y)
        self._pts: List[Tuple[int, float, float]] = []
        self._last_ball_frame: int = -10**9
        self._last_near_rim_frame: int = -10**9

        # Evidence flags / counters
        self._came_close_to_rim: bool = False
        self._passed_rim_plane: bool = False
        self._pass_details: str = ""
        self._pass_frame: int = -10**9

        self._below_rim_confirm: int = 0
        self._ever_center_evidence: bool = False
        self._center_pass_hits: int = 0

        self._ever_above_rim_center: bool = False

        self._far_rim_count: int = 0

        # Descent (apex passed) bookkeeping
        self._last_cy: Optional[float] = None
        self._has_descended: bool = False
        self._ever_descended: bool = False  # used as a "stronger" guard in your code

        # Debug container for overlays
        self.debug = MadeDebug()

        # Cached lines (recomputed when rim bbox updates)
        self._y_line: float = 0.0
        self._y_line_from_bbox: bool = False

    # -----------------------------
    # IO helpers
    # -----------------------------
    @staticmethod
    def _pick_rim(detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        rims = [d for d in detections if str(d.get("name", "")).lower() == "rim"]
        if not rims:
            return None
        return max(rims, key=lambda d: float(d.get("conf", 0.0)))

    def _reset(self) -> None:
        """Reset internal state at the end/cancel of an attempt window."""
        self.active = False
        self._start_frame = -10**9

        self._pts.clear()
        self._last_ball_frame = -10**9
        self._last_near_rim_frame = -10**9

        self._came_close_to_rim = False
        self._passed_rim_plane = False
        self._pass_details = ""
        self._pass_frame = -10**9

        self._below_rim_confirm = 0
        self._ever_center_evidence = False
        self._center_pass_hits = 0
        self._ever_above_rim_center = False

        self._far_rim_count = 0

        self._last_cy = None
        self._has_descended = False
        self._ever_descended = False

        self.debug.reset()

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
        rim_stable_bbox: Optional[BBox] = None,
        rim_stable_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[MadeEvent]:
        """
        Update outcome reasoning for the current frame.

        Inputs:
        - new_attempt starts a new outcome window (one attempt at a time).
        - ball_state is the temporally smoothed ball evidence (can be None during occlusions).
        - rim_stable_* provides a preferred rim reference (stabilized).

        Returns:
        - MadeEvent when a decision is committed, otherwise None.
        """

        # Overlapping attempts are force-closed as miss (pragmatic non-overlap rule).
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

            # Cache y_line definition at attempt start (and keep updating if stable bbox arrives).
            from .made_context import compute_y_line
            self._y_line, self._y_line_from_bbox = compute_y_line(self._rim_cy, self._rim_bbox, self.rim_line_rel_y)

        if not self.active:
            return None

        # Update rim reference when stabilized data is provided.
        if rim_stable_center is not None:
            self._rim_cx, self._rim_cy = map(float, rim_stable_center)

        if rim_stable_bbox is not None:
            self._rim_bbox = tuple(map(float, rim_stable_bbox))
            from .made_context import compute_y_line
            self._y_line, self._y_line_from_bbox = compute_y_line(self._rim_cy, self._rim_bbox, self.rim_line_rel_y)

        # If no ball this frame, we only evaluate timeout/grace logic later.
        if ball_state is not None:
            cx, cy = float(ball_state.cx), float(ball_state.cy)
            self._pts.append((int(frame_idx), cx, cy))
            self._last_ball_frame = int(frame_idx)

            # --------
            # Descent detection (FIXED):
            # We compare against the previous cy before updating _last_cy.
            # y increases downward in images; "descending" means cy increases.
            # --------
            prev_cy = self._last_cy
            if prev_cy is not None:
                if cy > prev_cy + 1.5:
                    self._has_descended = True
                if cy > prev_cy + 1.0:
                    self._ever_descended = True
            self._last_cy = cy

            # Build context (geometric measurements)
            ctx = build_context(
                frame_idx=frame_idx,
                ball_xy=(cx, cy),
                rim_xy=(self._rim_cx, self._rim_cy),
                rim_bbox=self._rim_bbox,
                rim_line_rel_y=self.rim_line_rel_y,
                below_rim_rel_y=self.below_rim_rel_y,
            )

            # Center evidence: ball x close to rim center at least once.
            x_thr = center_x_gate_thr(self._rim_bbox, self.center_gate_radius_rel, self.center_gate_min_px)
            if x_thr is not None and abs(cx - self._rim_cx) <= x_thr:
                self._ever_center_evidence = True
                self._center_pass_hits += 1
                self.debug.center_hits_pts.append((cx, cy))

            # Below-rim confirmation: consecutive frames below below_line.
            if ctx.below_line is not None:
                if cy >= ctx.below_line:
                    self._below_rim_confirm += 1
                    self.debug.below_hits_pts.append((cx, cy))
                else:
                    self._below_rim_confirm = 0

            # Direction sanity: ball above rim center at least once.
            if cy <= (self._rim_cy - 2.0):
                self._ever_above_rim_center = True

            # Near-rim evidence (used for grace + early miss)
            if near_rim_now(
                ball_xy=(cx, cy),
                rim_xy=(self._rim_cx, self._rim_cy),
                near_rim_dist_px=self.near_rim_dist_px,
                rim_bbox=self._rim_bbox,
                rim_expand_factor=self.rim_expand_factor,
            ):
                self._came_close_to_rim = True
                self._last_near_rim_frame = int(frame_idx)

            # Early miss heuristic:
            # If ball approached rim but then stays far away long enough after apex, commit miss.
            if self._came_close_to_rim and (not self._passed_rim_plane) and (ctx.dist_to_rim >= self.far_rim_dist_px):
                self._far_rim_count += 1
            else:
                self._far_rim_count = 0

            if self._has_descended and self._far_rim_count >= self.far_rim_confirm_frames:
                # Your original code added an extra guard using _ever_descended.
                # With the bug fixed, this now behaves as intended.
                if not self._ever_descended:
                    return None
                out = MadeEvent(frame_idx=frame_idx, outcome="miss", details=f"early_miss_far_rim(dist={ctx.dist_to_rim:.1f})")
                self._reset()
                return out

            # --------
            # Rim-plane crossing (computed once)
            # --------
            if len(self._pts) >= 2 and not self._passed_rim_plane:
                y_line = float(self._y_line)
                eps = float(self.y_epsilon_px + (0.0 if self._y_line_from_bbox else self.fallback_extra_y_epsilon_px))

                # Gate for validating crossing position.
                radii = rim_gate_radii(self._rim_bbox, self.gate_rx_rel, self.gate_ry_rel)

                # Optional requirement: only accept plane pass if we were near rim at some point.
                # (kept as parameter because it can reduce false positives in noisy settings)
                if (not self.require_near_rim_for_plane_pass) or self._came_close_to_rim:
                    for (_, x0, y0), (_, x1, y1) in zip(self._pts[:-1], self._pts[1:]):
                        cross = plane_crossing_point(prev_xy=(x0, y0), cur_xy=(x1, y1), y_line=y_line, eps=eps)
                        if cross is None:
                            continue

                        x_cross, y_cross = cross
                        if inside_rim_gate(
                            x=x_cross,
                            y=y_cross,
                            rim_cx=self._rim_cx,
                            y_line=y_line,
                            radii=radii,
                            fallback_x_px=65.0,
                            fallback_y_px=(self.y_epsilon_px + 3.0),
                        ):
                            self._passed_rim_plane = True
                            self._pass_frame = int(frame_idx)
                            self._pass_details = f"pass_plane(x={x_cross:.1f})"
                            self.debug.plane_cross_pt = (x_cross, y_cross)
                            break

            # --------
            # Final MADE decision rule (same semantics as your original rule)
            # --------
            if (
                self._has_descended
                and self._passed_rim_plane
                and self._ever_center_evidence
                and self._ever_above_rim_center
                and (self._below_rim_confirm >= self.below_confirm_frames)
            ):
                out = MadeEvent(
                    frame_idx=frame_idx,
                    outcome="made",
                    details=f"made(center_ever={self._ever_center_evidence}, below={self._below_rim_confirm})",
                )
                self._reset()
                return out

        # --------
        # Timeout -> MISS (after descent, with grace windows around rim / missing ball)
        # --------
        elapsed = int(frame_idx - self._start_frame)
        if elapsed >= self.window_frames and self._has_descended:
            if not self._ever_descended:
                return None

            if elapsed < self.max_window_frames:
                # Grace when still near rim or when ball evidence temporarily disappears.
                if (frame_idx - self._last_near_rim_frame) <= self.near_rim_grace_frames:
                    return None
                if (frame_idx - self._last_ball_frame) <= self.near_rim_grace_frames:
                    return None

            out = MadeEvent(
                frame_idx=frame_idx,
                outcome="miss",
                details=f"timeout_miss(elapsed={elapsed}, center_ever={self._ever_center_evidence})",
            )
            self._reset()
            return out

        return None
