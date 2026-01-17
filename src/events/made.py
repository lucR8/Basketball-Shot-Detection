from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math

from src.track.ball_tracker import BallState


@dataclass
class MadeEvent:
    frame_idx: int
    outcome: str
    details: str = ""


class MadeDetector:
    """
    Robust MADE logic:
      1) Try to detect a rim-plane pass (contiguous / gap-bridge / band).
      2) Confirm by short cylinder under y_line while descending and inside rim gate.

    Critical patch:
      - If tracking breaks at the rim, we can still accept a "post-gap" made:
        if we have evidence the ball was above the rim plane earlier (_was_above_line)
        and then it is confirmed below the rim (debounced) inside the rim gate.
    """

    def __init__(
        self,
        window_frames: int = 90,
        max_window_frames: int = 210,
        near_rim_grace_frames: int = 18,
        below_confirm_frames: int = 4,
        below_rim_rel_y: float = 0.86,

        y_epsilon_px: float = 3.0,
        rim_line_rel_y: float = 0.28,
        made_depth_rel: float = 0.65,

        plane_pass_segments: int = 18,
        max_cross_gap_frames: int = 12,

        # Rim gate (ellipse)
        gate_rx_rel: float = 0.46,
        gate_ry_rel: float = 0.18,
        gate_y_slack_px: float = 2.0,

        require_near_rim_for_plane_pass: bool = False,

        near_rim_dist_px: float = 155.0,
        rim_expand_factor: float = 1.20,
        min_points_for_decision: int = 4,
        fallback_extra_y_epsilon_px: float = 6.0,

        x_tol_px: float = 65.0,
        x_tol_rel: float = 0.75,

        # NEW: allow post-gap plane validation when below rim is confirmed
        allow_postgap_plane: bool = True,
    ):
        self.window_frames = int(window_frames)
        self.max_window_frames = int(max_window_frames)
        self.near_rim_grace_frames = int(near_rim_grace_frames)
        self.below_confirm_frames = int(below_confirm_frames)
        self.below_rim_rel_y = float(below_rim_rel_y)

        self.y_epsilon_px = float(y_epsilon_px)
        self.rim_line_rel_y = float(rim_line_rel_y)
        self.made_depth_rel = float(made_depth_rel)

        self.plane_pass_segments = int(plane_pass_segments)
        self.max_cross_gap_frames = int(max_cross_gap_frames)

        self.gate_rx_rel = float(gate_rx_rel)
        self.gate_ry_rel = float(gate_ry_rel)
        self.gate_y_slack_px = float(gate_y_slack_px)
        self.require_near_rim_for_plane_pass = bool(require_near_rim_for_plane_pass)

        self.near_rim_dist_px = float(near_rim_dist_px)
        self.rim_expand_factor = float(rim_expand_factor)
        self.min_points_for_decision = int(min_points_for_decision)
        self.fallback_extra_y_epsilon_px = float(fallback_extra_y_epsilon_px)

        self.x_tol_px = float(x_tol_px)
        self.x_tol_rel = float(x_tol_rel)

        self.allow_postgap_plane = bool(allow_postgap_plane)

        self.active: bool = False
        self._start_frame: int = -10**9

        self._rim_cx: float = 0.0
        self._rim_cy: float = 0.0
        self._rim_bbox: Optional[Tuple[float, float, float, float]] = None

        self._y_line: float = 0.0
        self._y_line_from_bbox: bool = False

        self._pts: List[Tuple[int, float, float]] = []

        self._came_close_to_rim: bool = False
        self._was_above_line: bool = False

        self._passed_rim_plane: bool = False
        self._pass_details: str = ""
        self._pass_frame: int = -10**9

        self._last_ball_frame: int = -10**9
        self._last_near_rim_frame: int = -10**9
        self._below_rim_confirm: int = 0

        self._min_dist_to_rim: float = float("inf")

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
        if self._rim_bbox is None:
            return self._rim_cy, False
        x1, y1, x2, y2 = self._rim_bbox
        h = max(1.0, (y2 - y1))
        rel = min(0.60, max(0.05, self.rim_line_rel_y))
        return (y1 + rel * h), True

    def _compute_below_rim_line(self) -> Optional[float]:
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

    def _x_tol_dynamic(self) -> float:
        size = self._rim_size()
        if size is None:
            return self.x_tol_px
        rim_w, _ = size
        return max(self.x_tol_px, self.x_tol_rel * rim_w)

    def _rim_gate_radii(self) -> Optional[Tuple[float, float]]:
        if self._rim_bbox is None:
            return None
        x1, y1, x2, y2 = self._rim_bbox
        rim_w = max(1.0, x2 - x1)
        rim_h = max(1.0, y2 - y1)
        rx = max(6.0, self.gate_rx_rel * rim_w)
        ry = max(3.0, self.gate_ry_rel * rim_h)
        return rx, ry

    def _inside_rim_gate(self, x: float, y: float) -> bool:
        radii = self._rim_gate_radii()
        if radii is None:
            return (abs(x - self._rim_cx) <= self._x_tol_dynamic()) and (
                abs(y - self._y_line) <= (self.y_epsilon_px + 2.0)
            )
        rx, ry = radii
        dx = (x - self._rim_cx) / rx
        dy = (y - self._y_line) / ry
        return (dx * dx + dy * dy) <= 1.0

    def _near_rim_now(self, cx: float, cy: float) -> bool:
        d = self._dist((cx, cy), (self._rim_cx, self._rim_cy))
        if d <= self.near_rim_dist_px:
            return True
        exp = self._expanded_rim_bbox()
        if exp is not None and self._point_in_bbox(cx, cy, exp):
            return True
        return False

    def _reset(self) -> None:
        self.active = False
        self._start_frame = -10**9
        self._rim_cx = 0.0
        self._rim_cy = 0.0
        self._rim_bbox = None
        self._y_line = 0.0
        self._y_line_from_bbox = False
        self._pts.clear()

        self._came_close_to_rim = False
        self._was_above_line = False

        self._passed_rim_plane = False
        self._pass_details = ""
        self._pass_frame = -10**9

        self._last_ball_frame = -10**9
        self._last_near_rim_frame = -10**9
        self._below_rim_confirm = 0

        self._min_dist_to_rim = float("inf")

    # -----------------------------
    # Main API
    # -----------------------------
    def update(
        self,
        frame_idx: int,
        detections: List[Dict[str, Any]],
        ball_state: Optional[BallState],
        new_attempt=None,
    ) -> Optional[MadeEvent]:

        # 0) Start a new attempt window
        if new_attempt is not None:
            self._reset()
            self.active = True
            self._start_frame = frame_idx

            self._rim_cx = float(getattr(new_attempt, "rim_cx"))
            self._rim_cy = float(getattr(new_attempt, "rim_cy"))

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

        # 1) Update rim bbox while active
        rim_det = self._pick_rim(detections)
        if rim_det is not None:
            self._rim_bbox = (
                float(rim_det["x1"]),
                float(rim_det["y1"]),
                float(rim_det["x2"]),
                float(rim_det["y2"]),
            )
            self._y_line, self._y_line_from_bbox = self._compute_y_line()

        # 2) Collect ball points + evidence
        near_rim_now = False
        if ball_state is not None:
            self._last_ball_frame = frame_idx
            cx, cy = float(ball_state.cx), float(ball_state.cy)
            self._pts.append((frame_idx, cx, cy))

            if cy <= (self._y_line - 2.0):
                self._was_above_line = True

            self._min_dist_to_rim = min(self._min_dist_to_rim, self._dist((cx, cy), (self._rim_cx, self._rim_cy)))

            if self._near_rim_now(cx, cy):
                near_rim_now = True
                self._came_close_to_rim = True
                self._last_near_rim_frame = frame_idx

            # below rim confirm (debounce)
            y_below = self._compute_below_rim_line()
            if y_below is not None:
                if cy >= y_below:
                    self._below_rim_confirm += 1
                else:
                    self._below_rim_confirm = 0
            else:
                self._below_rim_confirm = 0

            if len(self._pts) > 240:
                self._pts = self._pts[-240:]

        # 3) Detect rim-plane pass (contig / gap-bridge / band)
        if len(self._pts) >= 2 and (not self._passed_rim_plane):
            y_line = self._y_line
            eps = self.y_epsilon_px + (0.0 if self._y_line_from_bbox else self.fallback_extra_y_epsilon_px)
            K = min(self.plane_pass_segments, len(self._pts) - 1)

            def _near_guard_ok() -> bool:
                if not self.require_near_rim_for_plane_pass:
                    return True
                if ball_state is None:
                    return False
                return self._near_rim_now(float(ball_state.cx), float(ball_state.cy))

            # A) contiguous segments
            for i in range(-K, 0):
                (f0, x0, y0) = self._pts[i - 1]
                (f1, x1, y1) = self._pts[i]
                if (y0 <= y_line - eps) and (y1 >= y_line + eps) and (y1 != y0):
                    t = (y_line - y0) / (y1 - y0)
                    x_cross = x0 + t * (x1 - x0)
                    if self._inside_rim_gate(x_cross, y_line) and _near_guard_ok():
                        self._passed_rim_plane = True
                        self._pass_frame = frame_idx
                        self._pass_details = f"pass_plane_contig (x_cross={x_cross:.1f}, y_line={y_line:.1f})"
                        break

            # B) gap-bridge
            if not self._passed_rim_plane:
                pts_recent = self._pts[-(K + 1):]
                last_above = None
                for (f, x, y) in reversed(pts_recent):
                    if y <= (y_line - eps):
                        last_above = (f, x, y)
                        break
                if last_above is not None:
                    fa, xa, ya = last_above
                    first_below = None
                    for (f, x, y) in pts_recent:
                        if f <= fa:
                            continue
                        if (f - fa) > self.max_cross_gap_frames:
                            break
                        if y >= (y_line + eps):
                            first_below = (f, x, y)
                            break
                    if first_below is not None:
                        fb, xb, yb = first_below
                        if yb != ya:
                            t = (y_line - ya) / (yb - ya)
                            x_cross = xa + t * (xb - xa)
                            if self._inside_rim_gate(x_cross, y_line) and _near_guard_ok():
                                self._passed_rim_plane = True
                                self._pass_frame = frame_idx
                                self._pass_details = (
                                    f"pass_plane_gapbridge (gap={fb-fa}f, x_cross={x_cross:.1f}, y_line={y_line:.1f})"
                                )

            # C) band fallback
            if not self._passed_rim_plane:
                radii = self._rim_gate_radii()
                if radii is not None:
                    _, ry = radii
                    y_band = ry + self.gate_y_slack_px
                    for (_, x, y) in self._pts[-K:]:
                        if abs(y - y_line) <= y_band and self._inside_rim_gate(x, y) and _near_guard_ok():
                            self._passed_rim_plane = True
                            self._pass_frame = frame_idx
                            self._pass_details = f"pass_plane_band (x={x:.1f}, y={y:.1f}, band={y_band:.1f})"
                            break

        # ✅ NEW: post-gap plane validation when tracker cuts at the rim
        if (
            self.allow_postgap_plane
            and (not self._passed_rim_plane)
            and self._was_above_line
            and self._below_rim_confirm >= self.below_confirm_frames
            and ball_state is not None
        ):
            cx, cy = float(ball_state.cx), float(ball_state.cy)
            if self._inside_rim_gate(cx, cy):
                self._passed_rim_plane = True
                self._pass_frame = frame_idx
                self._pass_details = f"pass_plane_postgap_below (below_confirm={self._below_rim_confirm})"

        # 3bis) confirm MADE by short cylinder
        if (
            ball_state is not None
            and self._passed_rim_plane
            and self._was_above_line
            and self._rim_bbox is not None
        ):
            x1, y1, x2, y2 = self._rim_bbox
            rim_h = max(1.0, (y2 - y1))
            depth_px = max(2.0, self.made_depth_rel * rim_h)

            cx, cy = float(ball_state.cx), float(ball_state.cy)
            y_line = self._y_line
            eps = self.y_epsilon_px + (0.0 if self._y_line_from_bbox else self.fallback_extra_y_epsilon_px)

            # descending?
            descending = False
            if getattr(ball_state, "vy", None) is not None:
                try:
                    descending = float(ball_state.vy) > 0.0
                except Exception:
                    descending = False
            elif len(self._pts) >= 2:
                descending = (self._pts[-1][2] - self._pts[-2][2]) > 0.0

            in_y = (y_line + eps) <= cy <= (y_line + depth_px)
            in_gate = self._inside_rim_gate(cx, cy)

            if descending and in_y and in_gate:
                out = MadeEvent(
                    frame_idx=frame_idx,
                    outcome="made",
                    details=f"made_confirmed ({self._pass_details}, cx={cx:.1f}, cy={cy:.1f}, depth={depth_px:.1f})",
                )
                self._reset()
                return out

        # 4) Adaptive timeout (never early miss)
        elapsed = frame_idx - self._start_frame
        if elapsed >= self.window_frames:
            if elapsed < self.max_window_frames:
                if (frame_idx - self._last_near_rim_frame) <= self.near_rim_grace_frames:
                    return None
                if self._below_rim_confirm < self.below_confirm_frames:
                    lost_for = frame_idx - self._last_ball_frame
                    if lost_for <= self.near_rim_grace_frames:
                        return None

            if len(self._pts) < self.min_points_for_decision:
                out = MadeEvent(frame_idx=frame_idx, outcome="unknown",
                               details=f"timeout_unknown (pts={len(self._pts)}, elapsed={elapsed})")
                self._reset()
                return out

            y_below = self._compute_below_rim_line()
            out = MadeEvent(
                frame_idx=frame_idx,
                outcome="miss",
                details=(
                    f"timeout_miss (elapsed={elapsed}, close={self._came_close_to_rim}, "
                    f"near_rim_last={frame_idx - self._last_near_rim_frame}, "
                    f"below_confirm={self._below_rim_confirm}/{self.below_confirm_frames}, "
                    f"passed_plane={self._passed_rim_plane}, "
                    f"y_line={self._y_line:.1f}, y_below={(y_below if y_below is not None else -1):.1f}, "
                    f"min_dist={self._min_dist_to_rim:.1f})"
                ),
            )
            self._reset()
            return out

        return None
