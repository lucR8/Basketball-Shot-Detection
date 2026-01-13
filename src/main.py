from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from tqdm import tqdm
import cv2

from src.video.io import VideoReader, make_writer
from src.video.draw import draw_boxes
from src.detect.yolo import YoloDetector

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)

    in_path = cfg["video"]["input_path"]
    out_path = cfg["video"]["output_path"]
    max_frames = int(cfg["video"]["max_frames"])
    draw = bool(cfg["video"]["draw"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    det = YoloDetector(
        weights=cfg["yolo"]["weights"],
        conf=float(cfg["yolo"]["conf"]),
        iou=float(cfg["yolo"]["iou"]),
        imgsz=int(cfg["yolo"]["imgsz"]),
        device=str(cfg["yolo"]["device"]),
        classes=cfg["yolo"].get("classes", []),
    )

    cap = VideoReader(in_path).open()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = make_writer(out_path, fps, w, h)

    all_frames = []
    n = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    bar = tqdm(total=total if total > 0 else None, desc="Processing")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            dets = det.predict_frame(frame)
            all_frames.append({"frame_idx": n, "detections": dets})

            if draw:
                frame = draw_boxes(frame, dets)
            writer.write(frame)

            n += 1
            bar.update(1)
            if max_frames > 0 and n >= max_frames:
                break
    finally:
        bar.close()
        cap.release()
        writer.release()

    if cfg["runtime"]["save_json"]:
        json_path = cfg["runtime"]["json_path"]
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_frames, f, ensure_ascii=False, indent=2)

    print(f"✅ Done. Video: {out_path}")
    if cfg["runtime"]["save_json"]:
        print(f"✅ Detections: {cfg['runtime']['json_path']}")

if __name__ == "__main__":
    main()
