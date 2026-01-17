from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


def _center(det: dict) -> Tuple[float, float]:
    return (
        (float(det["x1"]) + float(det["x2"])) / 2.0,
        (float(det["y1"]) + float(det["y2"])) / 2.0,
    )


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class _Track:
    cx: float
    cy: float
    hits: int = 0
    still_hits: int = 0
    last_frame: int = -10**9


@dataclass
class StaticZone:
    cx: float
    cy: float
    r: float
    hits: int = 0


class FakeBallSuppressor:
    """
    Learns static "ball-like" detections (e.g., a basketball logo on a wall)
    and filters them out before tracking.

    It never removes moving balls; only detections that stay in the same place
    for long enough are added as static zones.
    """

    def __init__(
        self,
        match_radius_px: float = 14.0,     # association radius between frames
        still_radius_px: float = 6.0,      # considered "still" if movement <= this
        min_still_hits: int = 25,          # frames needed to classify as static
        zone_radius_px: float = 26.0,      # suppression radius around learned static center
        forget_frames: int = 90,           # forget temporary tracks if not seen
        max_tracks: int = 50,
    ):
        self.match_radius_px = float(match_radius_px)
        self.still_radius_px = float(still_radius_px)
        self.min_still_hits = int(min_still_hits)
        self.zone_radius_px = float(zone_radius_px)
        self.forget_frames = int(forget_frames)
        self.max_tracks = int(max_tracks)

        self._tracks: Dict[int, _Track] = {}
        self._next_id: int = 1
        self.zones: List[StaticZone] = []

    def _in_static_zone(self, cx: float, cy: float) -> bool:
        for z in self.zones:
            if _dist((cx, cy), (z.cx, z.cy)) <= z.r:
                return True
        return False

    def filter(self, frame_idx: int, dets: List[dict]) -> List[dict]:
        """
        Updates internal state using current detections,
        then returns a filtered list where static fake balls are removed.
        """

        # 1) update tracks using ball detections not already in static zones
        balls = [d for d in dets if str(d.get("name", "")).lower() == "ball"]
        ball_centers = []
        for d in balls:
            cx, cy = _center(d)
            if not self._in_static_zone(cx, cy):
                ball_centers.append((cx, cy, d))

        # simple greedy matching to existing tracks
        used_track_ids = set()
        for (cx, cy, det) in ball_centers:
            best_id: Optional[int] = None
            best_dist = 1e9

            for tid, tr in self._tracks.items():
                if tid in used_track_ids:
                    continue
                dd = _dist((cx, cy), (tr.cx, tr.cy))
                if dd < best_dist and dd <= self.match_radius_px:
                    best_dist = dd
                    best_id = tid

            if best_id is None:
                # create new track
                if len(self._tracks) < self.max_tracks:
                    tid = self._next_id
                    self._next_id += 1
                    self._tracks[tid] = _Track(cx=cx, cy=cy, hits=1, still_hits=0, last_frame=frame_idx)
                    used_track_ids.add(tid)
            else:
                tr = self._tracks[best_id]
                move = _dist((cx, cy), (tr.cx, tr.cy))
                tr.hits += 1
                if move <= self.still_radius_px:
                    tr.still_hits += 1

                # EMA update (keeps track stable)
                alpha = 0.25
                tr.cx = (1 - alpha) * tr.cx + alpha * cx
                tr.cy = (1 - alpha) * tr.cy + alpha * cy
                tr.last_frame = frame_idx
                used_track_ids.add(best_id)

                # Promote to static zone if still long enough
                if tr.still_hits >= self.min_still_hits:
                    self.zones.append(StaticZone(cx=tr.cx, cy=tr.cy, r=self.zone_radius_px, hits=tr.hits))
                    # remove track so it doesn't keep growing
                    del self._tracks[best_id]

        # 2) garbage collect old tracks
        to_del = [tid for tid, tr in self._tracks.items() if (frame_idx - tr.last_frame) > self.forget_frames]
        for tid in to_del:
            del self._tracks[tid]

        # 3) filter out ball detections inside static zones
        out = []
        for d in dets:
            if str(d.get("name", "")).lower() != "ball":
                out.append(d)
                continue
            cx, cy = _center(d)
            if self._in_static_zone(cx, cy):
                # dropped fake/static ball
                continue
            out.append(d)

        return out
