from __future__ import annotations
from dataclasses import dataclass
import cv2


@dataclass
class VideoReader:
    """
    Minimal wrapper around cv2.VideoCapture with explicit error handling.

    Keeping I/O isolated makes the main pipeline easier to test and read.
    """
    path: str

    def open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")
        return cap


def make_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """
    Create a video writer with a standard codec (mp4v).

    This function fails fast if the writer cannot be opened, which avoids
    silently producing empty outputs.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {output_path}")
    return writer
