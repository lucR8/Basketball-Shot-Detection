from __future__ import annotations

import cv2
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from src.video.io import VideoReader, make_writer
from src.detect.yolo import YoloDetector

from src.track.ball_tracker import BallTracker
from src.events.attempt import AttemptDetector
from src.events.made import MadeDetector

from src.detect.fake_ball_suppressor import FakeBallSuppressor
from src.video.draw import draw_attempt_gating_debug

from src.video.draw import (
    draw_ball_trace,
    draw_attempt_debug,
    draw_scoreboard,
    draw_result_flash,
)

VIDEO_PATH = "data/input/sample3.mp4"
OUT_VIDEO_PATH = "data/output/output_main3.mp4"

YOLO_WEIGHTS = r"models\\ball_rim_person_shoot_best_t.pt"
FRAME_STRIDE = 1
DEBUG = False

CONF_BY_CLASS = {
    "ball": 0.15,
    "rim": 0.25,
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
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def filter_by_class_conf(dets, conf_by_class, default_conf=0.25):
    out = []
    for d in dets:
        name = str(d.get("name", "")).lower()
        thr = conf_by_class.get(name, default_conf)
        if float(d.get("conf", 0.0)) >= thr:
            out.append(d)
    return out


def best_rim_center(dets):
    rims = [d for d in dets if str(d.get("name", "")).lower() == "rim"]
    if not rims:
        return None
    best = max(rims, key=lambda d: float(d.get("conf", 0.0)))
    cx = (float(best["x1"]) + float(best["x2"])) / 2.0
    cy = (float(best["y1"]) + float(best["y2"])) / 2.0
    return (cx, cy)


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

    cap = VideoReader(VIDEO_PATH).open()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_fps = fps / FRAME_STRIDE
    writer = make_writer(OUT_VIDEO_PATH, out_fps, w, h)

    tracker = BallTracker(ball_class_name="ball")

    # -----------------------------
    # AttemptDetector (responsive + robust ball)
    # -----------------------------
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
        # IMPORTANT: on assouplit (et attempt.py re-scale avec le rim)
        ball_area_max_px2=40000,

        require_ball_below_rim_to_rearm=True,
        below_margin_px=25,
        below_confirm_frames=1,
        shot_window_frames=25,

        # NEW: rim scaling
        enable_rim_scaling=True,
        rim_ref_w=110.0,          # si besoin on ajustera après 1 run debug
        rim_scale_min=0.65,
        rim_scale_max=1.80,

        # NEW: robust ball point
        ball_point_memory_frames=6,
        allow_oversize_ball_when_armed=True,
        oversize_ball_conf_min=0.20,
        oversize_ball_dist_boost=1.35,
    )

    made_detector = MadeDetector(
        window_frames=75,
        x_tol_px=65,
        rim_line_rel_y=0.28,
        near_rim_dist_px=155,
    )

    attempts = made = miss = unknown = 0
    ball_trace = []

    frame_idx = 0
    pbar = tqdm(total=total if total > 0 else None, desc="Processing")

    shoot_seen = 0
    shoot_max_conf = 0.0
    shoot_rise_count = 0
    shoot_release_count = 0

    attempt_open = False
    attempt_open_frame = -10**9

    ATTEMPT_MAX_FRAMES = 120  # >= made_detector.window_frames
    BALL_FAR_FACTOR = 1.8
    BELOW_RIM_MARGIN = 12

    gate_hist = Counter()
    gate_hist_when_free = Counter()
    blocked_open_frames = 0
    blocked_shot_window_frames = 0

    def _count_gate(reas: str | None, *, free: bool):
        if not reas:
            return
        gate_hist[reas] += 1
        if free:
            gate_hist_when_free[reas] += 1

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

            shoot_confs = [
                float(d.get("conf", 0.0))
                for d in dets
                if str(d.get("name", "")).lower() == "shoot"
            ]
            if shoot_confs:
                shoot_seen += 1
                shoot_max_conf = max(shoot_max_conf, max(shoot_confs))

            rim_center = best_rim_center(dets)

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
                _count_gate(attempt_detector.last_debug.get("gate_reason"), free=False)

            if attempt_evt is not None:
                attempts += 1
                attempt_open = True
                attempt_open_frame = frame_idx

                details = (attempt_evt.details or "").lower()
                if "shoot_rise" in details:
                    shoot_rise_count += 1
                if "shoot_release" in details:
                    shoot_release_count += 1

            made_evt = made_detector.update(frame_idx, dets, ball_state, new_attempt=attempt_evt)
            if made_evt is not None:
                attempt_open = False
                if made_evt.outcome == "made":
                    made += 1
                elif made_evt.outcome == "miss":
                    miss += 1
                else:
                    unknown += 1

            if DEBUG:
                frame = draw_yolo_boxes(frame, dets)
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

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        writer.release()

    print("Saved:", OUT_VIDEO_PATH)
    print(f"Attempts={attempts} Made={made} Miss={miss} Unknown={unknown}")
    if attempts > 0:
        print(f"FG% = {100.0 * made / attempts:.1f}%")
    print(f"Shoot frames detected: {shoot_seen} | max shoot conf: {shoot_max_conf:.3f}")
    print("Shoot-rise attempts:", shoot_rise_count)
    print("Shoot-release attempts:", shoot_release_count)

    print("\n--- Gate reason histogram (all counted frames) ---")
    for k, v in gate_hist.most_common():
        print(f"{k:30s} {v}")

    print("\n--- Gate reason histogram (ONLY when free_to_attempt = True) ---")
    for k, v in gate_hist_when_free.most_common():
        print(f"{k:30s} {v}")

    print("\n--- Blocking stats ---")
    print("Frames with attempt_open=True:", attempts)
    print("Frames where gate_reason == blocked_shot_window (while free_to_attempt):", gate_hist_when_free.get("blocked_shot_window", 0))


if __name__ == "__main__":
    main()
