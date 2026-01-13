from __future__ import annotations
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weights: str, conf: float, iou: float, imgsz: int, device: str, classes: list[int] | None = None):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.classes = classes if classes else None

    def predict_frame(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            verbose=False
        )
        r = results[0]
        names = r.names

        dets = []
        if r.boxes is None:
            return dets

        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0].item())
            cls = int(b.cls[0].item())
            dets.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": conf,
                "cls": cls,
                "name": names.get(cls, str(cls)),
            })
        return dets
