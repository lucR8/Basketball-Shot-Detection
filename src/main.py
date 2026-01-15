from __future__ import annotations

import cv2
from pathlib import Path
from tqdm import tqdm

from src.video.io import VideoReader, make_writer  # <-- votre io.py
from src.detect.yolo import YoloDetector           # <-- votre yolo.py

from src.track.ball_tracker import BallTracker
from src.events.attempt import AttemptDetector
from src.events.made import MadeDetector

from src.video.draw import (
    draw_ball_trace,
    draw_attempt_debug,
    draw_scoreboard,
    draw_result_flash,
)

VIDEO_PATH = "data/input/sample2.mp4"
OUT_VIDEO_PATH = "data/output/output_main2.mp4"

YOLO_WEIGHTS = "models\\ball_rim_person_shoot_best_t.pt"
FRAME_STRIDE = 1
DEBUG = True

# Seuils par classe (à ajuster)
CONF_BY_CLASS = {
    "ball": 0.12,
    "rim": 0.25,
    "person": 0.30,
    "shoot": 0.15,
}


def filter_by_class_conf(dets, conf_by_class, default_conf=0.25):
    """Filtre les détections avec un seuil par classe."""
    out = []
    for d in dets:
        name = str(d.get("name", "")).lower()
        thr = conf_by_class.get(name, default_conf)
        if float(d.get("conf", 0.0)) >= thr:
            out.append(d)
    return out


def best_rim_center(dets):
    """Retourne (cx, cy) du meilleur rim (confiance max), sinon None."""
    rims = [d for d in dets if str(d.get("name", "")).lower() == "rim"]
    if not rims:
        return None
    best = max(rims, key=lambda d: float(d.get("conf", 0.0)))
    cx = (float(best["x1"]) + float(best["x2"])) / 2.0
    cy = (float(best["y1"]) + float(best["y2"])) / 2.0
    return (cx, cy)


def main():
    Path("data/output").mkdir(parents=True, exist_ok=True)

    # Detector YOLO (votre modèle ball/rim/person/shoot)
    # IMPORTANT: on met une conf globale basse, puis on filtre par classe via CONF_BY_CLASS
    detector = YoloDetector(
        weights=YOLO_WEIGHTS,
        conf=0.06,
        iou=0.45,
        imgsz=640,
        device="0",     # "0" si GPU, "cpu" sinon
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

    attempt_detector = AttemptDetector(
        enter_radius_px=85,
        vy_min=0.2,
        cooldown_frames=12,
        require_approach=True,
        approach_window=6,
        rim_recent_frames=15,
        ball_recent_frames=25,
        shoot_conf_min=0.14,
        shoot_conf_strong=0.30,

        require_ball_below_rim_to_rearm=True,
        below_margin_px=45,
        below_confirm_frames=3
    )

    made_detector = MadeDetector(window_frames=45, x_tol_px=55, y_margin_px=10)

    attempts = made = miss = airball = unknown = 0
    ball_trace = []

    frame_idx = 0
    pbar = tqdm(total=total if total > 0 else None, desc="Processing")

    shoot_seen = 0
    shoot_max_conf = 0.0
    shoot_rise_count = 0
    shoot_release_count = 0

    # ------------------------------------------------------------
    # Attempt gating (Attempt -> ouvre une fenêtre, Made/Miss la ferme)
    # ------------------------------------------------------------
    attempt_open = False
    attempt_open_frame = -10**9

    # fenêtre max pendant laquelle on considère qu'un attempt est "en cours"
    ATTEMPT_MAX_FRAMES = 70  # ~2.3s à 30fps (ajustez)
    BALL_FAR_FACTOR = 1.8    # ball doit être loin du rim pour autoriser un nouveau tir
    BELOW_RIM_MARGIN = 12    # px, tolérance sous le rim

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Skip si stride > 1
            if FRAME_STRIDE > 1 and (frame_idx % FRAME_STRIDE != 0):
                frame_idx += 1
                pbar.update(1)
                continue

            # 1) YOLO
            raw_dets = detector.predict_frame(frame)
            dets = filter_by_class_conf(raw_dets, CONF_BY_CLASS, default_conf=0.25)

            # Stats shoot (présence/conf max)
            shoot_confs = [
                float(d.get("conf", 0.0))
                for d in dets
                if str(d.get("name", "")).lower() == "shoot"
            ]
            if shoot_confs:
                shoot_seen += 1
                shoot_max_conf = max(shoot_max_conf, max(shoot_confs))

            # 2) Rim center (pour tracking plus permissif près du panier)
            rim_center = best_rim_center(dets)

            # 3) Tracking balle
            ball_state = tracker.update(frame_idx, dets, rim_center=rim_center)
            if ball_state is not None:
                ball_trace.append((ball_state.cx, ball_state.cy))

            # 4) Attempt
            # IMPORTANT: on ne dépend PAS de made_detector.active() pour autoriser un attempt.
            # On gère notre propre "fenêtre attempt" (attempt_open).
            attempt_evt = None

            # On peut refermer la fenêtre attempt si:
            # - outcome final reçu (géré plus bas)
            # - ou timeout
            # - ou balle clairement loin + sous le rim (anti rebond)
            if attempt_open:
                # Timeout
                if (frame_idx - attempt_open_frame) > ATTEMPT_MAX_FRAMES:
                    attempt_open = False

                # Anti-rebond géométrique (si on a rim + ball)
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

            # Tant qu'un attempt est ouvert: on n'en déclenche pas de nouveau
            if not attempt_open:
                attempt_evt = attempt_detector.update(frame_idx, dets, ball_state)
                if attempt_evt is not None:
                    attempts += 1
                    attempt_open = True
                    attempt_open_frame = frame_idx

                    details = (attempt_evt.details or "").lower()
                    if "shoot_rise" in details:
                        shoot_rise_count += 1
                    if "shoot_release" in details:
                        shoot_release_count += 1

            # 5) Made/Miss
            made_evt = made_detector.update(frame_idx, dets, ball_state, new_attempt=attempt_evt)
            if made_evt is not None:
                # Si made/miss/airball est émis, ça ferme la fenêtre attempt
                attempt_open = False

                if made_evt.outcome == "made":
                    made += 1
                elif made_evt.outcome == "miss":
                    miss += 1
                elif made_evt.outcome == "airball":
                    airball += 1
                else:
                    unknown += 1

            # 6) Debug visuel
            if DEBUG:
                frame = draw_ball_trace(frame, ball_trace, max_length=25)
                frame = draw_attempt_debug(
                    frame,
                    attempt_evt,
                    attempts,
                    radius_px=int(attempt_detector.enter_radius_px),
                )
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
    print(f"Shoot frames detected: {shoot_seen} | max shoot conf: {shoot_max_conf:.3f}")
    print("Shoot-rise attempts:", shoot_rise_count)
    print("Shoot-release attempts:", shoot_release_count)


if __name__ == "__main__":
    main()
