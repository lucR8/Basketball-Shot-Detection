from __future__ import annotations

import cv2
from pathlib import Path
from tqdm import tqdm
from collections import Counter, deque
from typing import Optional, Tuple, Dict, Any, List

from src.video.io import VideoReader, make_writer
from src.detect.yolo import YoloDetector

from src.track.ball_tracker import BallTracker
from src.track.rim_stabilizer import RimStabilizer, RimStable

from src.events.attempt import AttemptDetector
from src.events.made.made import MadeDetector

from src.detect.fake_ball_suppressor import FakeBallSuppressor
from src.video.draw import draw_attempt_gating_debug
from src.video.draw import (
    draw_ball_trace,
    draw_attempt_debug,
    draw_scoreboard,
    draw_result_flash,
    draw_made_debug,
)

VIDEO_PATH = "data/input/sample.mp4"
OUT_VIDEO_PATH = "data/output/output_main.mp4"

YOLO_WEIGHTS = r"models\\ball_rim_person_shoot_best_t.pt"
FRAME_STRIDE = 1
DEBUG = True

CONF_BY_CLASS = {
    "ball": 0.15,
    "rim": 0.20,
    "person": 0.30,
    "shoot": 0.15,
}


def draw_yolo_boxes(frame, dets):
    colors = {
        "ball": (0, 255, 0),
        "rim": (0, 255, 255),
        "person": (0, 200, 0),
        "shoot": (255, 0, 0),
    }
    for d in dets:
        name = str(d.get("name", "")).lower()
        conf = float(d.get("conf", 0.0))
        x1, y1 = int(d["x1"]), int(d["y1"])
        x2, y2 = int(d["x2"]), int(d["y2"])
        color = colors.get(name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def filter_by_class_conf(dets, conf_by_class, default_conf=0.25):
    out = []
    for d in dets:
        name = str(d.get("name", "")).lower()
        thr = conf_by_class.get(name, default_conf)
        if float(d.get("conf", 0.0)) >= thr:
            out.append(d)
    return out


def _draw_rim_stable_overlay(frame, rim_s: Optional[RimStable]):
    """
    Debug overlay: show stable rim center/bbox WITHOUT modifying YOLO dets.
    """
    if rim_s is None:
        return frame

    cx, cy = int(rim_s.cx), int(rim_s.cy)
    x1, y1, x2, y2 = rim_s.bbox
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

    # Draw stable bbox (thin) + center cross
    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (255, 255, 255), 1)
    cv2.drawMarker(frame, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

    label = f"rim_stable conf={rim_s.conf:.2f}"
    cv2.putText(frame, label, (x1i, max(20, y1i - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def main():
    Path("data/output").mkdir(parents=True, exist_ok=True)

    detector = YoloDetector(
        weights=YOLO_WEIGHTS,
        conf=0.06,
        iou=0.45,
        imgsz=640,
        device="0",
        classes=None,
    )

    fake_ball = FakeBallSuppressor(
        match_radius_px=14,
        still_radius_px=6,
        min_still_hits=25,
        zone_radius_px=28,
        forget_frames=90,
    )

    # Rim stabilizer (important for BallTracker + MadeDetector)
    rim_stab = RimStabilizer(alpha=0.12, conf_min=0.35, hold_frames=60, warmup_min_hits=8, max_step_px=0.0)

    cap = VideoReader(VIDEO_PATH).open()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_fps = fps / FRAME_STRIDE
    writer = make_writer(OUT_VIDEO_PATH, out_fps, w, h)

    tracker = BallTracker(ball_class_name="ball")

    attempt_detector = AttemptDetector(
        rim_recent_frames=15,
        ball_recent_frames=25,
        shoot_conf_min=0.10,
        shoot_arm_window=25,
        ball_person_max_dist_px=95,
        release_sep_increase_px=35,
        release_debounce_frames=2,
        enable_ball_size_filter=True,
        ball_area_min_px2=180,
        ball_area_max_px2=40000,
        require_ball_below_rim_to_rearm=True,
        below_margin_px=25,
        below_confirm_frames=1,
        shot_window_frames=25,
        enable_rim_scaling=True,
        rim_ref_w=110.0,
        rim_scale_min=0.65,
        rim_scale_max=1.80,
        ball_point_memory_frames=6,
        allow_oversize_ball_when_armed=True,
        oversize_ball_conf_min=0.20,
        oversize_ball_dist_boost=1.35,
    )

    made_detector = MadeDetector(
        window_frames=75,
        max_window_frames=210,
        near_rim_dist_px=155,
        rim_line_rel_y=0.28,
        below_confirm_frames=3,
        below_gate_confirm_frames=2,
        below_gate_window=10,
        below_gate_radius_rel=0.55,
        below_gate_min_px=22.0,
    )

    attempts = made = miss = unknown = 0
    ball_trace = []

    frame_idx = 0
    pbar = tqdm(total=total if total > 0 else None, desc="Processing")

    shoot_seen = 0
    shoot_max_conf = 0.0

    attempts_armed_via_rise = 0
    attempts_armed_via_streak = 0
    attempts_released_by_left = 0
    attempts_released_by_sep = 0
    attempts_shoot_from_memory = 0

    attempt_open = False
    attempt_open_frame = -10**9

    # must be >= made_detector.max_window_frames ideally
    ATTEMPT_MAX_FRAMES = 230
    BALL_FAR_FACTOR = 1.8
    BELOW_RIM_MARGIN = 12

    gate_hist = Counter()
    gate_hist_when_free = Counter()
    blocked_open_frames = 0

    miss_details = deque(maxlen=10)
    made_details = deque(maxlen=10)
    unknown_details = deque(maxlen=10)
    forced_unknown_details = deque(maxlen=10)

    def _count_gate(reason: str | None, *, free: bool):
        if not reason:
            return
        gate_hist[reason] += 1
        if free:
            gate_hist_when_free[reason] += 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if FRAME_STRIDE > 1 and (frame_idx % FRAME_STRIDE != 0):
                frame_idx += 1
                pbar.update(1)
                continue

            raw_dets = detector.predict_frame(frame)
            dets = filter_by_class_conf(raw_dets, CONF_BY_CLASS, default_conf=0.25)
            dets = fake_ball.filter(frame_idx, dets)

            shoot_confs = [float(d.get("conf", 0.0)) for d in dets if str(d.get("name", "")).lower() == "shoot"]
            if shoot_confs:
                shoot_seen += 1
                shoot_max_conf = max(shoot_max_conf, max(shoot_confs))

            # rim stable (center + bbox) -- does not touch dets
            rim_s = rim_stab.update(frame_idx, dets)
            rim_center = (rim_s.cx, rim_s.cy) if rim_s is not None else None

            ball_state = tracker.update(frame_idx, dets, rim_center=rim_center)
            if ball_state is not None:
                ball_trace.append((ball_state.cx, ball_state.cy))

            # -----------------------------
            # Attempt gating
            # -----------------------------
            attempt_evt = None

            if attempt_open:
                blocked_open_frames += 1

                if (frame_idx - attempt_open_frame) > ATTEMPT_MAX_FRAMES:
                    attempt_open = False
                    unknown += 1
                    forced_unknown_details.append(f"forced_unknown@{frame_idx}: attempt_open_timeout({ATTEMPT_MAX_FRAMES})")

                if attempt_open and (rim_center is not None) and (ball_state is not None):
                    rim_cx, rim_cy = rim_center
                    dx = ball_state.cx - rim_cx
                    dy = ball_state.cy - rim_cy
                    dist = (dx * dx + dy * dy) ** 0.5

                    far_thr = attempt_detector.enter_radius_px * BALL_FAR_FACTOR
                    ball_is_far = dist >= far_thr
                    ball_is_below = ball_state.cy >= (rim_cy + BELOW_RIM_MARGIN)
                    if ball_is_far and ball_is_below:
                        attempt_open = False

            free_to_attempt = not attempt_open

            if free_to_attempt:
                attempt_evt = attempt_detector.update(frame_idx, dets, ball_state)
                _count_gate(attempt_detector.last_debug.get("gate_reason"), free=True)
            else:
                _count_gate("blocked_attempt_open", free=False)

            if attempt_evt is not None:
                attempts += 1
                attempt_open = True
                attempt_open_frame = frame_idx

                dbg = attempt_detector.last_debug or {}
                if dbg.get("rise_arm_ok"):
                    attempts_armed_via_rise += 1
                if dbg.get("streak_arm_ok"):
                    attempts_armed_via_streak += 1
                if dbg.get("left_shoot"):
                    attempts_released_by_left += 1
                if dbg.get("sep_ok"):
                    attempts_released_by_sep += 1
                if dbg.get("shoot_from_memory"):
                    attempts_shoot_from_memory += 1

            # -----------------------------
            # MADE / MISS decision
            # pass rim stable (bbox + center)
            # -----------------------------
            rim_bbox_stable = rim_s.bbox if rim_s is not None else None
            rim_center_stable = (rim_s.cx, rim_s.cy) if rim_s is not None else None

            made_evt = made_detector.update(
                frame_idx,
                dets,
                ball_state,
                new_attempt=attempt_evt,
                rim_stable_bbox=rim_bbox_stable,
                rim_stable_center=rim_center_stable,
            )

            if made_evt is not None:
                attempt_open = False
                if made_evt.outcome == "made":
                    made += 1
                    made_details.append(f"made@{made_evt.frame_idx}: {made_evt.details}")
                elif made_evt.outcome == "miss":
                    miss += 1
                    miss_details.append(f"miss@{made_evt.frame_idx}: {made_evt.details}")
                else:
                    unknown += 1
                    unknown_details.append(f"unknown@{made_evt.frame_idx}: {made_evt.details}")

            if DEBUG:
                frame = draw_yolo_boxes(frame, dets)  # raw YOLO boxes (will remain unstable)
                frame = _draw_rim_stable_overlay(frame, rim_s)  # stable rim overlay (white)
                frame = draw_ball_trace(frame, ball_trace, max_length=25)
                frame = draw_attempt_debug(
                    frame,
                    attempt_evt,
                    attempts,
                    radius_px=int(attempt_detector.enter_radius_px),
                )
                frame = draw_scoreboard(frame, attempts, made, miss, unknown)
                frame = draw_result_flash(frame, made_evt)
                frame = draw_attempt_gating_debug(frame, attempt_detector, org=(10, 160))
                frame = draw_made_debug(frame, made_detector)

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        writer.release()     

    # -------------------------------------------------
    # FORCE UNKNOWN if video ends with unresolved attempts
    # -------------------------------------------------
    resolved = made + miss + unknown
    missing = attempts - resolved

    if missing > 0:
        for _ in range(missing):
            unknown += 1
            forced_unknown_details.append(
                f"forced_unknown@end_of_video(frame={frame_idx})"
            )

    print("Saved:", OUT_VIDEO_PATH)
    print(f"Attempts={attempts} Made={made} Miss={miss} Unknown={unknown}")
    if attempts > 0:
        print(f"FG% = {100.0 * made / attempts:.1f}%")

    print(f"Shoot frames detected: {shoot_seen} | max shoot conf: {shoot_max_conf:.3f}")
    print(f"Attempts armed via rise:   {attempts_armed_via_rise}")
    print(f"Attempts armed via streak: {attempts_armed_via_streak}")
    print(f"Attempts shoot_from_memory:{attempts_shoot_from_memory}")
    print(f"Release by left_shoot:     {attempts_released_by_left}")
    print(f"Release by sep_ok:         {attempts_released_by_sep}")

    print("\n--- Gate reason histogram (all counted frames) ---")
    for k, v in gate_hist.most_common():
        print(f"{k:30s} {v}")

    print("\n--- Gate reason histogram (ONLY when free_to_attempt = True) ---")
    for k, v in gate_hist_when_free.most_common():
        print(f"{k:30s} {v}")

    print("\n--- Blocking stats ---")
    print("Frames with attempt_open=True:", blocked_open_frames)
    print("Frames blocked by attempt_open:", gate_hist.get("blocked_attempt_open", 0))

    print("\n--- Miss details (last 10) ---")
    for s in list(miss_details)[-10:]:
        print(s)

    print("\n--- Made details (last 10) ---")
    for s in list(made_details)[-10:]:
        print(s)

    print("\n--- Unknown details (last 10) ---")
    for s in list(unknown_details)[-10:]:
        print(s)

    print("\n--- Forced unknown (attempt_open timeout, last 10) ---")
    for s in list(forced_unknown_details)[-10:]:
        print(s)


if __name__ == "__main__":
    main()
