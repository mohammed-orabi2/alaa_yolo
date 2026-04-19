from pydantic import BaseModel
from typing import List


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class ImageDetectionResponse(BaseModel):
    detections: List[Detection]
    count: int
    annotated_image: str  # base64-encoded JPEG
