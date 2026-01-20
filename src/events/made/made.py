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
    MADE/MISS logic (robust, supports swish):

    1) Detect a rim-plane pass (contiguous / gap-bridge / band) using ball points vs y_line.
    2) Confirm MADE under the rim using MULTIPLE ball points.

    IMPORTANT PATCH (CENTER TRAJECTORY):
      - A MADE is ONLY valid if the BALL TRAJECTORY PASSES CLOSE TO THE RIM CENTER
      - Evidence does NOT need to be at the exact rim-plane frame.
      - This makes the logic robust to YOLO / tracking gaps.
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

        # Rim gate (ellipse) for plane-cross validation
        gate_rx_rel: float = 0.46,
        gate_ry_rel: float = 0.18,
        gate_y_slack_px: float = 2.0,

        require_near_rim_for_plane_pass: bool = False,
        near_rim_dist_px: float = 155.0,
        rim_expand_factor: float = 1.20,

        min_points_for_decision: int = 4,
        fallback_extra_y_epsilon_px: float = 6.0,

        # Swish MADE confirmation under rim
        below_gate_confirm_frames: int = 2,
        below_gate_window: int = 10,
        below_gate_radius_rel: float = 0.55,
        below_gate_min_px: float = 22.0,

        # Post-gap plane validation
        allow_postgap_plane: bool = True,

        # CENTER TRAJECTORY GATE
        center_gate_radius_rel: float = 0.35,
        center_gate_min_px: float = 18.0,
        center_gate_window: int = 6,
        center_gate_min_hits: int = 1,
    ):
        self.window_frames = int(window_frames)
        self.max_window_frames = int(max_window_frames)
        self.near_rim_grace_frames = int(near_rim_grace_frames)

        self.below_confirm_frames = int(below_confirm_frames)
        self.below_rim_rel_y = float(below_rim_rel_y)

        self.y_epsilon_px = float(y_epsilon_px)
        self.rim_line_rel_y = float(rim_line_rel_y)

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

        self.below_gate_confirm_frames = int(below_gate_confirm_frames)
        self.below_gate_window = int(below_gate_window)
        self.below_gate_radius_rel = float(below_gate_radius_rel)
        self.below_gate_min_px = float(below_gate_min_px)

        self.allow_postgap_plane = bool(allow_postgap_plane)

        self.center_gate_radius_rel = float(center_gate_radius_rel)
        self.center_gate_min_px = float(center_gate_min_px)
        self.center_gate_window = int(center_gate_window)
        self.center_gate_min_hits = int(center_gate_min_hits)

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

        # CENTER trajectory evidence
        self._center_pass_hits: int = 0

        # NEW — persistent center evidence (key fix)
        self._ever_center_evidence: bool = False
        
        self._ever_above_rim_center: bool = False

        # DEBUG
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

    def _center_x_gate_thr(self) -> Optional[float]:
        size = self._rim_size()
        if size is None:
            return None
        rim_w, _ = size
        return max(self.center_gate_min_px, self.center_gate_radius_rel * rim_w)

    def _reset(self) -> None:
        self.active = False
        self._start_frame = -10**9
        self._pts.clear()
        self._center_pass_hits = 0
        self._ever_center_evidence = False  # NEW
        self._dbg_center_hits_pts.clear()
        self._dbg_below_hits_pts.clear()
        self._dbg_plane_cross_pt = None
        self._ever_above_rim_center = False


    def _near_rim_now(self, cx: float, cy: float) -> bool:
        d = self._dist((cx, cy), (self._rim_cx, self._rim_cy))
        if d <= self.near_rim_dist_px:
            return True
        exp = self._expanded_rim_bbox()
        if exp is not None and self._point_in_bbox(cx, cy, exp):
            return True
        return False

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

        # 0) Start attempt
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

        # 1) Update rim reference
        if rim_stable_center is not None:
            self._rim_cx, self._rim_cy = map(float, rim_stable_center)

        if rim_stable_bbox is not None:
            self._rim_bbox = tuple(map(float, rim_stable_bbox))
            self._y_line, self._y_line_from_bbox = self._compute_y_line()

        # 2) Ball evidence
        if ball_state is not None:
            cx, cy = float(ball_state.cx), float(ball_state.cy)
            self._pts.append((frame_idx, cx, cy))
            self._last_ball_frame = frame_idx

            x_thr = self._center_x_gate_thr()
            if x_thr is not None and abs(cx - self._rim_cx) <= x_thr:
                self._ever_center_evidence = True  # NEW
                self._dbg_center_hits_pts.append((cx, cy))

            y_below = self._compute_below_rim_line()
            if y_below is not None:
                if cy >= y_below:
                    self._below_rim_confirm += 1
                    self._dbg_below_hits_pts.append((cx, cy))
                else:
                    self._below_rim_confirm = 0

            if self._near_rim_now(cx, cy):
                self._last_near_rim_frame = frame_idx
            
            if cy <= (self._rim_cy - 2.0):
                self._ever_above_rim_center = True


        # 3) Plane crossing
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

        # 4) MADE decision (KEY FIX)
        if (
            self._passed_rim_plane
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

        # 5) Timeout → MISS
        elapsed = frame_idx - self._start_frame
        if elapsed >= self.window_frames:
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
