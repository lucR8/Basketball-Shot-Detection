from __future__ import annotations
import cv2

def draw_boxes(frame, detections, show_label: bool = True):
    """
    detections: list of dict with keys: x1,y1,x2,y2,conf,cls,name
    """
    for d in detections:
        x1, y1, x2, y2 = map(int, (d["x1"], d["y1"], d["x2"], d["y2"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if show_label:
            label = f'{d.get("name","cls")} {d["conf"]:.2f}'
            cv2.putText(frame, label, (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame
