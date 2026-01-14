from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import math


@dataclass
class BallState:
    frame_idx: int
    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0
    conf: float = 0.0


class BallTracker:
    """
    Lightweight ball tracker for YOLO detections.

    Input each frame:
      - list of detections dicts (from yolo.py)
    Output:
      - current ball state (position + velocity) or None

    Strategy:
      1) Filter detections to keep only class "ball" (by name or class id).
      2) Pick best detection using:
           - proximity to last position (if available)
           - confidence
           - reasonable bbox size
      3) Smooth position with exponential moving average (EMA).
      4) Estimate velocity from smoothed positions.
      5) Handle temporary misses (occlusion) with max_lost frames.
    """

    def __init__(
        self,
        ball_class_name: str = "ball",
        max_lost: int = 8,
        ema_alpha: float = 0.6,
        max_jump_px: float = 120.0,
    ):
        """
        ball_class_name: class name in detection dicts ("ball")
        max_lost: allowed consecutive frames without ball detection before state reset
        ema_alpha: smoothing factor (0..1). higher = less smoothing, more reactive.
        max_jump_px: reject detections too far from last known position (helps reduce false positives)
        """
        self.ball_class_name = ball_class_name
        self.max_lost = max_lost
        self.ema_alpha = ema_alpha
        self.max_jump_px = max_jump_px

        self._lost = 0
        self._history: List[BallState] = []
        self._last_smoothed: Optional[Tuple[float, float]] = None

    @staticmethod
    def _center(det: Dict[str, Any]) -> Tuple[float, float]:
        cx = (det["x1"] + det["x2"]) / 2.0
        cy = (det["y1"] + det["y2"]) / 2.0
        return cx, cy

    @staticmethod
    def _bbox_area(det: Dict[str, Any]) -> float:
        return max(0.0, (det["x2"] - det["x1"])) * max(0.0, (det["y2"] - det["y1"]))

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def reset(self) -> None:
        self._lost = 0
        self._history.clear()
        self._last_smoothed = None

    def history(self) -> List[BallState]:
        return self._history

    def last(self) -> Optional[BallState]:
        return self._history[-1] if self._history else None

    def _filter_ball_dets(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Prefer name-based filtering
        balls = [d for d in detections if str(d.get("name", "")).lower() == self.ball_class_name.lower()]
        return balls

    def _select_best(self, ball_dets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not ball_dets:
            return None

        # If no previous position, take highest confidence (tie -> smaller area tends to be ball-like)
        if self._last_smoothed is None:
            return sorted(ball_dets, key=lambda d: (d["conf"], -self._bbox_area(d)), reverse=True)[0]

        last_pos = self._last_smoothed

        # Score = confidence - distance penalty - area penalty (area penalty is mild)
        best_det = None
        best_score = -1e18
        for d in ball_dets:
            cx, cy = self._center(d)
            dist = self._dist((cx, cy), last_pos)
            area = self._bbox_area(d)

            # hard gate: reject huge jumps to reduce false positives
            if dist > self.max_jump_px:
                continue

            # normalize terms
            score = (2.0 * float(d["conf"])) - (0.02 * dist) - (0.000001 * area)

            if score > best_score:
                best_score = score
                best_det = d

        # If all were rejected by max_jump, fallback to highest confidence anyway
        if best_det is None:
            best_det = sorted(ball_dets, key=lambda d: d["conf"], reverse=True)[0]

        return best_det

    def _ema(self, prev: Tuple[float, float], new: Tuple[float, float]) -> Tuple[float, float]:
        a = self.ema_alpha
        return (a * new[0] + (1 - a) * prev[0], a * new[1] + (1 - a) * prev[1])

    def update(self, frame_idx: int, detections: List[Dict[str, Any]]) -> Optional[BallState]:
        """
        Update tracker with detections of current frame.

        Returns:
          BallState if tracked (even if predicted during short occlusion, we keep last state),
          None if completely lost (after max_lost).
        """
        ball_dets = self._filter_ball_dets(detections)
        best = self._select_best(ball_dets)

        if best is None:
            # no detection -> occlusion
            self._lost += 1
            if self._lost > self.max_lost:
                self.reset()
                return None
            # keep last known state (do not add new points)
            return self.last()

        # We have a detection
        self._lost = 0
        cx, cy = self._center(best)

        # Smooth
        if self._last_smoothed is None:
            sm_cx, sm_cy = cx, cy
        else:
            sm_cx, sm_cy = self._ema(self._last_smoothed, (cx, cy))
        self._last_smoothed = (sm_cx, sm_cy)

        # Velocity (difference of smoothed positions)
        vx, vy = 0.0, 0.0
        prev = self.last()
        if prev is not None:
            dt = max(1, frame_idx - prev.frame_idx)
            vx = (sm_cx - prev.cx) / dt
            vy = (sm_cy - prev.cy) / dt

        state = BallState(
            frame_idx=frame_idx,
            cx=sm_cx,
            cy=sm_cy,
            vx=vx,
            vy=vy,
            conf=float(best.get("conf", 0.0)),
        )
        self._history.append(state)
        return state
