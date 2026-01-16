from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math

from src.track.ball_tracker import BallState


@dataclass
class MadeEvent:
    """
    Event emitted when the outcome of an attempt is decided.

    outcome in {"made", "miss", "unknown"}.

    NOTE:
      You asked to DROP "airball" logic from the project.
      So we never emit "airball" anymore.
    """
    frame_idx: int
    outcome: str
    details: str = ""


class MadeDetector:
    """
    Decide MADE / MISS (and fallback UNKNOWN) after an AttemptEvent opens a time window.

    Important properties:
      - We NEVER emit miss early.
      - We use a rim crossing line y_line near the TOP of rim bbox (bbox includes net).
      - MADE:
          (1) crossing y_line downward within x tolerance
          (2) cylinder entry below y_line (rim-bounce re-entry fix)
      - Timeout is ADAPTIVE:
          after min_window_frames, we keep the window open if the ball is still interacting
          with the rim area (near rim recently) or not yet confirmed below rim.
    """

    def __init__(
        self,
        # Base / minimum window after attempt
        window_frames: int = 90,

        # Adaptive timeout: hard max (prevents infinite windows)
        max_window_frames: int = 210,

        # Adaptive timeout: we keep the window open if ball was near rim recently
        near_rim_grace_frames: int = 18,

        # Adaptive timeout: confirm "ball below rim" with debounce
        below_confirm_frames: int = 4,

        # Where we define "below rim line" inside rim bbox (towards bottom)
        # (bbox includes net, so we don't use y2 directly)
        below_rim_rel_y: float = 0.86,

        # MADE params
        x_tol_px: float = 65.0,
        x_tol_rel: float = 0.75,
        y_epsilon_px: float = 4.0,
        rim_line_rel_y: float = 0.28,
        made_depth_rel: float = 0.65,

        # Evidence zone
        near_rim_dist_px: float = 155.0,
        rim_expand_factor: float = 1.20,
        min_points_for_decision: int = 4,
        fallback_extra_y_epsilon_px: float = 6.0,
    ):
        # Base timeout
        self.window_frames = int(window_frames)
        self.max_window_frames = int(max_window_frames)
        self.near_rim_grace_frames = int(near_rim_grace_frames)
        self.below_confirm_frames = int(below_confirm_frames)
        self.below_rim_rel_y = float(below_rim_rel_y)

        # MADE params
        self.x_tol_px = float(x_tol_px)
        self.x_tol_rel = float(x_tol_rel)
        self.y_epsilon_px = float(y_epsilon_px)
        self.rim_line_rel_y = float(rim_line_rel_y)
        self.made_depth_rel = float(made_depth_rel)

        # Evidence params
        self.near_rim_dist_px = float(near_rim_dist_px)
        self.rim_expand_factor = float(rim_expand_factor)
        self.min_points_for_decision = int(min_points_for_decision)
        self.fallback_extra_y_epsilon_px = float(fallback_extra_y_epsilon_px)

        # Active attempt window state
        self.active: bool = False
        self._start_frame: int = -10**9

        # Frozen rim reference for this attempt (from AttemptEvent, stable)
        self._rim_cx: float = 0.0
        self._rim_cy: float = 0.0

        # Rim bbox (from YOLO) — updated while active
        self._rim_bbox: Optional[Tuple[float, float, float, float]] = None  # (x1,y1,x2,y2)

        # Crossing line y
        self._y_line: float = 0.0
        self._y_line_from_bbox: bool = False

        # Trajectory points during the window: (frame_idx, cx, cy)
        self._pts: List[Tuple[int, float, float]] = []

        # Evidence flags
        self._came_close_to_rim: bool = False
        self._was_above_line: bool = False

        # Adaptive timeout state
        self._last_ball_frame: int = -10**9
        self._last_near_rim_frame: int = -10**9
        self._below_rim_confirm: int = 0

        # Debug
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
        """
        Line used to confirm that the ball has passed under the rim area.
        We compute it inside the rim bbox (not at y2, because bbox includes net).
        """
        if self._rim_bbox is None:
            return None
        x1, y1, x2, y2 = self._rim_bbox
        h = max(1.0, (y2 - y1))
        rel = min(0.98, max(0.40, self.below_rim_rel_y))
        return (y1 + rel * h)

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

        self._last_ball_frame = -10**9
        self._last_near_rim_frame = -10**9
        self._below_rim_confirm = 0

        self._min_dist_to_rim = float("inf")

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

    # -----------------------------
    # Main API
    # -----------------------------
    def update(
        self,
        frame_idx: int,
        detections: List[Dict[str, Any]],
        ball_state: Optional[BallState],
        new_attempt=None,  # AttemptEvent (duck-typed)
    ) -> Optional[MadeEvent]:

        # ----------------------------------------------------------
        # 0) Start a new attempt window
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # 1) Update rim bbox while active
        # ----------------------------------------------------------
        rim_det = self._pick_rim(detections)
        if rim_det is not None:
            self._rim_bbox = (
                float(rim_det["x1"]),
                float(rim_det["y1"]),
                float(rim_det["x2"]),
                float(rim_det["y2"]),
            )
            self._y_line, self._y_line_from_bbox = self._compute_y_line()

        # ----------------------------------------------------------
        # 2) Collect ball points + evidence + adaptive timeout state
        # ----------------------------------------------------------
        near_rim_now = False

        if ball_state is not None:
            self._last_ball_frame = frame_idx

            cx, cy = float(ball_state.cx), float(ball_state.cy)
            self._pts.append((frame_idx, cx, cy))

            if cy <= (self._y_line - 2.0):
                self._was_above_line = True

            d = self._dist((cx, cy), (self._rim_cx, self._rim_cy))
            self._min_dist_to_rim = min(self._min_dist_to_rim, d)

            if d <= self.near_rim_dist_px:
                near_rim_now = True
                self._came_close_to_rim = True

            exp_bbox = self._expanded_rim_bbox()
            if exp_bbox is not None and self._point_in_bbox(cx, cy, exp_bbox):
                near_rim_now = True
                self._came_close_to_rim = True

            if near_rim_now:
                self._last_near_rim_frame = frame_idx

            # Confirm below-rim with debounce
            y_below = self._compute_below_rim_line()
            if y_below is not None:
                if cy >= y_below:
                    self._below_rim_confirm += 1
                else:
                    self._below_rim_confirm = 0
            else:
                # No bbox -> can't confirm below rim reliably
                self._below_rim_confirm = 0

            if len(self._pts) > 200:
                self._pts = self._pts[-200:]

        # ----------------------------------------------------------
        # 3) MADE by crossing y_line downward
        # ----------------------------------------------------------
        if len(self._pts) >= 2:
            (_, x0, y0) = self._pts[-2]
            (_, x1, y1) = self._pts[-1]

            y_line = self._y_line
            eps = self.y_epsilon_px + (0.0 if self._y_line_from_bbox else self.fallback_extra_y_epsilon_px)

            if (y0 <= y_line - eps) and (y1 >= y_line + eps) and (y1 != y0):
                t = (y_line - y0) / (y1 - y0)
                x_cross = x0 + t * (x1 - x0)

                x_tol = self._x_tol_dynamic()
                if abs(x_cross - self._rim_cx) <= x_tol:
                    out = MadeEvent(
                        frame_idx=frame_idx,
                        outcome="made",
                        details=(
                            f"made_by_crossing (y_line={y_line:.1f}, x_cross={x_cross:.1f}, "
                            f"rim_x={self._rim_cx:.1f}, x_tol={x_tol:.1f})"
                        ),
                    )
                    self._reset()
                    return out

        # ----------------------------------------------------------
        # 3bis) MADE by cylinder entry (rim bounce re-entry fix)
        # ----------------------------------------------------------
        if ball_state is not None and self._was_above_line and self._rim_bbox is not None:
            x1, y1, x2, y2 = self._rim_bbox
            rim_w = max(1.0, (x2 - x1))
            rim_h = max(1.0, (y2 - y1))

            x_tol = max(self.x_tol_px, self.x_tol_rel * rim_w)
            depth_px = max(2.0, self.made_depth_rel * rim_h)

            cx, cy = float(ball_state.cx), float(ball_state.cy)
            y_line = self._y_line
            eps = self.y_epsilon_px + (0.0 if self._y_line_from_bbox else self.fallback_extra_y_epsilon_px)

            descending = False
            if getattr(ball_state, "vy", None) is not None:
                try:
                    descending = float(ball_state.vy) > 0.0
                except Exception:
                    descending = False
            elif len(self._pts) >= 2:
                descending = (self._pts[-1][2] - self._pts[-2][2]) > 0.0

            in_x = abs(cx - self._rim_cx) <= x_tol
            in_y = (y_line + eps) <= cy <= (y_line + depth_px)

            if descending and in_x and in_y:
                out = MadeEvent(
                    frame_idx=frame_idx,
                    outcome="made",
                    details=(
                        f"made_by_cylinder (y_line={y_line:.1f}, cx={cx:.1f}, cy={cy:.1f}, "
                        f"rim_x={self._rim_cx:.1f}, x_tol={x_tol:.1f}, depth={depth_px:.1f})"
                    ),
                )
                self._reset()
                return out

        # ----------------------------------------------------------
        # 4) ADAPTIVE TIMEOUT: decide MISS vs UNKNOWN (still no early miss)
        # ----------------------------------------------------------
        elapsed = frame_idx - self._start_frame
        if elapsed >= self.window_frames:
            # Hard max window
            if elapsed < self.max_window_frames:
                # Keep alive if ball has been near rim recently
                if (frame_idx - self._last_near_rim_frame) <= self.near_rim_grace_frames:
                    return None

                # Keep alive if ball not yet confirmed below rim (debounced)
                if self._below_rim_confirm < self.below_confirm_frames:
                    # BUT if ball is lost for a while and we are not near rim recently,
                    # we should allow closing (prevents hanging windows)
                    lost_for = frame_idx - self._last_ball_frame
                    if lost_for <= self.near_rim_grace_frames:
                        return None

            # Now we are allowed to close the window (either safe or hard max reached)
            if len(self._pts) < self.min_points_for_decision:
                out = MadeEvent(
                    frame_idx=frame_idx,
                    outcome="unknown",
                    details=f"timeout_unknown (pts={len(self._pts)}, elapsed={elapsed})",
                )
                self._reset()
                return out

            # Not made within window => miss
            y_below = self._compute_below_rim_line()
            out = MadeEvent(
                frame_idx=frame_idx,
                outcome="miss",
                details=(
                    f"timeout_miss (elapsed={elapsed}, close={self._came_close_to_rim}, "
                    f"near_rim_last={frame_idx - self._last_near_rim_frame}, "
                    f"below_confirm={self._below_rim_confirm}/{self.below_confirm_frames}, "
                    f"y_line={self._y_line:.1f}, y_below={(y_below if y_below is not None else -1):.1f}, "
                    f"min_dist={self._min_dist_to_rim:.1f})"
                ),
            )
            self._reset()
            return out

        return None
