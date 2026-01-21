from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple

from src.track.ball_tracker import BallState

from .made_types import MadeEvent
from .made_context import (
    BBox,
    build_context,
    center_x_gate_thr,
    rim_gate_radii,
    rim_scale_from_bbox,
    scale_px,
)
from .made_gates import (
    near_rim_now,
    find_plane_crossing,
    inside_rim_gate,
    below_gate_hit,
)


class MadeDetector:
    """
    Outcome classifier (Made / Miss / Unknown) triggered after an AttemptEvent.

    Engineering intent:
    - The detector does not infer outcomes from a single frame.
    - It accumulates ball evidence in a bounded window after the attempt start.

    Key principle:
    - Perception (YOLO + tracking) provides noisy measurements.
    - This module performs temporal/geometric reasoning on those measurements.
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

        enable_rim_scaling: bool = True,
        rim_ref_w: float = 110.0,
        rim_scale_min: float = 0.65,
        rim_scale_max: float = 1.80,

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

        # Rim-plane definition (y_line) and crossing tolerance (pixels)
        self.y_epsilon_px = float(y_epsilon_px)
        self.rim_line_rel_y = float(rim_line_rel_y)

        # Parameters kept for compatibility with earlier iterations
        self.plane_pass_segments = int(plane_pass_segments)
        self.max_cross_gap_frames = int(max_cross_gap_frames)

        # Rim gate for validating the crossing x-position
        self.gate_rx_rel = float(gate_rx_rel)
        self.gate_ry_rel = float(gate_ry_rel)
        self.gate_y_slack_px = float(gate_y_slack_px)

        # "Near rim" notion used for grace windows and early-miss heuristic (pixels)
        self.require_near_rim_for_plane_pass = bool(require_near_rim_for_plane_pass)
        self.near_rim_dist_px = float(near_rim_dist_px)
        self.rim_expand_factor = float(rim_expand_factor)

        # Rim-size scaling: convert pixel thresholds to be consistent across zoom levels.
        self.enable_rim_scaling = bool(enable_rim_scaling)
        self.rim_ref_w = float(rim_ref_w)
        self.rim_scale_min = float(rim_scale_min)
        self.rim_scale_max = float(rim_scale_max)

        self.min_points_for_decision = int(min_points_for_decision)
        self.fallback_extra_y_epsilon_px = float(fallback_extra_y_epsilon_px)

        # Additional “swish-like” evidence under rim (multiple points)
        self.below_gate_confirm_frames = int(below_gate_confirm_frames)
        self.below_gate_window = int(below_gate_window)
        self.below_gate_radius_rel = float(below_gate_radius_rel)
        self.below_gate_min_px = float(below_gate_min_px)

        self.allow_postgap_plane = bool(allow_postgap_plane)

        # Center trajectory evidence
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
        self._rim_bbox: Optional[BBox] = None

        self._y_line: float = 0.0
        self._y_line_from_bbox: bool = False

        # Current rim scale factor (1.0 when bbox missing or scaling disabled).
        self._scale: float = 1.0

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

        self._ever_center_evidence: bool = False
        self._ever_above_rim_center: bool = False

        # Early-miss bookkeeping
        self._far_rim_count = 0

        # Descent detection (image y increases downward)
        self._last_cy: Optional[float] = None
        self._has_descended = False

        # DEBUG visualization points
        self._dbg_center_hits_pts: List[Tuple[float, float]] = []
        self._dbg_below_hits_pts: List[Tuple[float, float]] = []
        self._dbg_plane_cross_pt: Optional[Tuple[float, float]] = None

    @staticmethod
    def _pick_rim(detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        rims = [d for d in detections if str(d.get("name", "")).lower() == "rim"]
        if not rims:
            return None
        return max(rims, key=lambda d: float(d.get("conf", 0.0)))

    def _compute_scale(self) -> float:
        if not self.enable_rim_scaling:
            return 1.0
        return rim_scale_from_bbox(
            self._rim_bbox,
            rim_ref_w=self.rim_ref_w,
            scale_min=self.rim_scale_min,
            scale_max=self.rim_scale_max,
        )

    def _reset(self) -> None:
        self.active = False
        self._start_frame = -10**9
        self._pts.clear()

        self._came_close_to_rim = False
        self._passed_rim_plane = False
        self._pass_details = ""
        self._pass_frame = -10**9

        self._last_ball_frame = -10**9
        self._last_near_rim_frame = -10**9

        self._below_rim_confirm = 0
        self._min_dist_to_rim = float("inf")

        self._ever_center_evidence = False
        self._ever_above_rim_center = False

        self._far_rim_count = 0

        self._last_cy = None
        self._has_descended = False

        self._dbg_center_hits_pts.clear()
        self._dbg_below_hits_pts.clear()
        self._dbg_plane_cross_pt = None

        self._scale = 1.0

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
        Update outcome state machine.

        Inputs:
        - new_attempt: starts a new outcome window (one attempt at a time)
        - ball_state: ball evidence each frame (can be None during occlusions)
        - rim_stable_bbox/center: optional stabilized rim reference (preferred)

        Output:
        - MadeEvent when the module commits to an outcome, else None.
        """

        # If a new attempt starts while one is still active, force-close the previous one.
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

            # y-line depends on bbox; scale depends on bbox width.
            ctx0 = build_context(
                frame_idx=frame_idx,
                ball_xy=(self._rim_cx, self._rim_cy),
                rim_xy=(self._rim_cx, self._rim_cy),
                rim_bbox=self._rim_bbox,
                rim_line_rel_y=self.rim_line_rel_y,
                below_rim_rel_y=self.below_rim_rel_y,
            )
            self._y_line = float(ctx0.y_line)
            self._y_line_from_bbox = bool(ctx0.y_line_from_bbox)
            self._scale = self._compute_scale()

        if not self.active:
            return None

        # Keep rim reference up to date when stabilized data is provided.
        if rim_stable_center is not None:
            self._rim_cx, self._rim_cy = map(float, rim_stable_center)

        if rim_stable_bbox is not None:
            self._rim_bbox = tuple(map(float, rim_stable_bbox))
            ctx0 = build_context(
                frame_idx=frame_idx,
                ball_xy=(self._rim_cx, self._rim_cy),
                rim_xy=(self._rim_cx, self._rim_cy),
                rim_bbox=self._rim_bbox,
                rim_line_rel_y=self.rim_line_rel_y,
                below_rim_rel_y=self.below_rim_rel_y,
            )
            self._y_line = float(ctx0.y_line)
            self._y_line_from_bbox = bool(ctx0.y_line_from_bbox)
            self._scale = self._compute_scale()

        # Accumulate ball evidence.
        if ball_state is not None:
            cx, cy = float(ball_state.cx), float(ball_state.cy)
            self._pts.append((frame_idx, cx, cy))
            self._last_ball_frame = frame_idx

            # Descent detection (y increases downward in images).
            if self._last_cy is not None and (cy > self._last_cy + 1.5):
                self._has_descended = True
            self._last_cy = cy

            ctx = build_context(
                frame_idx=frame_idx,
                ball_xy=(cx, cy),
                rim_xy=(self._rim_cx, self._rim_cy),
                rim_bbox=self._rim_bbox,
                rim_line_rel_y=self.rim_line_rel_y,
                below_rim_rel_y=self.below_rim_rel_y,
            )

            # --- Scaled thresholds (pixel-based -> rim-size-based) ---
            near_rim_dist = scale_px(self.near_rim_dist_px, self._scale)
            far_rim_dist = scale_px(self.far_rim_dist_px, self._scale)
            y_eps = scale_px(self.y_epsilon_px, self._scale)
            fb_eps = scale_px(self.fallback_extra_y_epsilon_px, self._scale)

            # Center evidence: x close to rim center at least once in the window.
            x_thr = center_x_gate_thr(
                self._rim_bbox,
                self.center_gate_radius_rel,
                scale_px(self.center_gate_min_px, self._scale),
            )
            if x_thr is not None and abs(cx - self._rim_cx) <= x_thr:
                self._ever_center_evidence = True
                self._dbg_center_hits_pts.append((cx, cy))

            # Below-rim evidence: require consecutive frames below a line.
            if ctx.below_line is not None:
                if cy >= float(ctx.below_line):
                    self._below_rim_confirm += 1
                    self._dbg_below_hits_pts.append((cx, cy))
                else:
                    self._below_rim_confirm = 0

            # Direction sanity: ball was above rim center at least once.
            if cy <= (self._rim_cy - 2.0):
                self._ever_above_rim_center = True

            # Near-rim evidence (used for grace + early miss).
            if near_rim_now(
                ball_xy=(cx, cy),
                rim_xy=(self._rim_cx, self._rim_cy),
                near_rim_dist_px=near_rim_dist,
                rim_bbox=self._rim_bbox,
                rim_expand_factor=self.rim_expand_factor,
            ):
                self._came_close_to_rim = True
                self._last_near_rim_frame = frame_idx

            # Early miss: ball approached and then moved far away after apex passed.
            dist_to_rim = float(ctx.dist_to_rim)
            if self._came_close_to_rim and (not self._passed_rim_plane) and (dist_to_rim >= far_rim_dist):
                self._far_rim_count += 1
            else:
                self._far_rim_count = 0

            if self._has_descended and self._far_rim_count >= self.far_rim_confirm_frames:
                out = MadeEvent(frame_idx=frame_idx, outcome="miss", details=f"early_miss_far_rim(dist={dist_to_rim:.1f})")
                self._reset()
                return out

            # Detect rim-plane crossing once: above -> below y_line (scaled epsilon),
            # and validate crossing x using rim gate.
            if len(self._pts) >= 2 and not self._passed_rim_plane:
                y_line = float(self._y_line)
                eps = float(y_eps + (0.0 if self._y_line_from_bbox else fb_eps))

                radii = rim_gate_radii(self._rim_bbox, self.gate_rx_rel, self.gate_ry_rel)

                for (_, x0, y0), (_, x1, y1) in zip(self._pts[:-1], self._pts[1:]):
                    cross = find_plane_crossing(prev_xy=(x0, y0), cur_xy=(x1, y1), y_line=y_line, eps=eps)
                    if cross is None:
                        continue

                    x_cross, _ = cross
                    if inside_rim_gate(
                        x=float(x_cross),
                        y=float(y_line),
                        rim_cx=float(self._rim_cx),
                        y_line=float(y_line),
                        radii=radii,
                        fallback_x_px=65.0,
                        fallback_y_px=scale_px(self.y_epsilon_px + 3.0, self._scale),
                    ):
                        self._passed_rim_plane = True
                        self._pass_frame = frame_idx
                        self._dbg_plane_cross_pt = (float(x_cross), float(y_line))
                        self._pass_details = f"pass_plane(x={x_cross:.1f})"
                        break

            # Final MADE rule.
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
            elapsed = frame_idx - self._start_frame
            if elapsed >= self.window_frames and self._has_descended:
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
