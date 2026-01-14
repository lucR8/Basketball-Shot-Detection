from __future__ import annotations

import cv2
from pathlib import Path
from tqdm import tqdm

from src.video.io import VideoReader, make_writer  # <-- votre io.py :contentReference[oaicite:4]{index=4}
from src.detect.yolo import YoloDetector           # <-- votre yolo.py :contentReference[oaicite:5]{index=5}

from src.track.ball_tracker import BallTracker
from src.events.attempt import AttemptDetector
from src.events.made import MadeDetector

from src.video.draw import (
    draw_ball_trace,
    draw_attempt_debug,
    draw_scoreboard,
    draw_result_flash,
)

VIDEO_PATH = "data/input/sample.mp4"
OUT_VIDEO_PATH = "data/output/output_annotated.mp4"

YOLO_WEIGHTS = "runs/detect/train2/weights/best.pt"  # adaptez train2 -> votre dossier
FRAME_STRIDE = 1
DEBUG = True

def main():
    Path("data/output").mkdir(parents=True, exist_ok=True)

    # Detector YOLO (ball + rim only conseillé si votre modèle est ball/rim)
    detector = YoloDetector(
        weights=YOLO_WEIGHTS,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device="cpu",   # "0" si GPU
        classes=None    # si vous voulez filtrer par ids, on peut le faire
    )

    cap = VideoReader(VIDEO_PATH).open()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_fps = fps / FRAME_STRIDE
    writer = make_writer(OUT_VIDEO_PATH, out_fps, w, h)

    tracker = BallTracker(ball_class_name="ball")
    attempt_detector = AttemptDetector(enter_radius_px=85, vy_min=0.5, cooldown_frames=25)
    made_detector = MadeDetector(window_frames=45, x_tol_px=55, y_margin_px=10)

    attempts = made = miss = airball = unknown = 0
    ball_trace = []

    frame_idx = 0
    pbar = tqdm(total=total if total > 0 else None, desc="Processing")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if FRAME_STRIDE > 1 and frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            dets = detector.predict_frame(frame)

            ball_state = tracker.update(frame_idx, dets)
            if ball_state is not None:
                ball_trace.append((ball_state.cx, ball_state.cy))

            attempt_evt = attempt_detector.update(frame_idx, dets, ball_state)
            if attempt_evt is not None:
                attempts += 1

            made_evt = made_detector.update(frame_idx, dets, ball_state, new_attempt=attempt_evt)
            if made_evt is not None:
                if made_evt.outcome == "made":
                    made += 1
                elif made_evt.outcome == "miss":
                    miss += 1
                elif made_evt.outcome == "airball":
                    airball += 1
                else:
                    unknown += 1

            if DEBUG:
                frame = draw_ball_trace(frame, ball_trace, max_length=25)
                frame = draw_attempt_debug(frame, attempt_evt, attempts, radius_px=int(attempt_detector.enter_radius_px))
                frame = draw_scoreboard(frame, attempts, made, miss, airball, unknown)
                frame = draw_result_flash(frame, made_evt)

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        writer.release()

    print("Saved:", OUT_VIDEO_PATH)
    print(f"Attempts={attempts} Made={made} Miss={miss} Airball={airball} Unknown={unknown}")
    if attempts > 0:
        print(f"FG% = {100.0 * made / attempts:.1f}%")

if __name__ == "__main__":
    main()
