from __future__ import annotations
from dataclasses import dataclass
import cv2

@dataclass
class VideoReader:
    path: str

    def open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")
        return cap

def make_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {output_path}")
    return writer
