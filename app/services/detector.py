import base64
import os
import tempfile

import cv2
import numpy as np
from ultralytics import YOLO

from app.schemas.response import BoundingBox, Detection, ImageDetectionResponse

_model: YOLO | None = None


def load_model(model_path: str) -> None:
    global _model
    _model = YOLO(model_path)


def get_model() -> YOLO | None:
    return _model


def _parse_detections(results) -> list[Detection]:
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append(
                Detection(
                    class_id=int(box.cls),
                    class_name=r.names[int(box.cls)],
                    confidence=round(float(box.conf), 4),
                    bbox=BoundingBox(
                        x1=float(box.xyxy[0][0]),
                        y1=float(box.xyxy[0][1]),
                        x2=float(box.xyxy[0][2]),
                        y2=float(box.xyxy[0][3]),
                    ),
                )
            )
    return detections


def predict_image(image_bytes: bytes, conf: float = 0.25) -> ImageDetectionResponse:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    results = _model(img, conf=conf, verbose=False)
    detections = _parse_detections(results)
    annotated = results[0].plot()
    _, buf = cv2.imencode(".jpg", annotated)
    return ImageDetectionResponse(
        detections=detections,
        count=len(detections),
        annotated_image=base64.b64encode(buf).decode(),
    )


def predict_video(video_bytes: bytes, conf: float = 0.25) -> tuple[bytes, int, int]:
    """Returns (annotated_video_bytes, frame_count, total_detections)."""
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.mp4")
        out_path = os.path.join(tmp, "output.mp4")

        with open(in_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        frame_count = 0
        total_detections = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = _model(frame, conf=conf, verbose=False)
            writer.write(results[0].plot())
            total_detections += len(_parse_detections(results))
            frame_count += 1

        cap.release()
        writer.release()

        with open(out_path, "rb") as f:
            return f.read(), frame_count, total_detections
