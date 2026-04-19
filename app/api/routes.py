from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.schemas.response import ImageDetectionResponse
from app.services import detector

router = APIRouter(prefix="/api/v1")

_MAX_IMAGE_BYTES = 20 * 1024 * 1024   # 20 MB
_MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MB


@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector.get_model() is not None}


@router.post("/detect/image", response_model=ImageDetectionResponse)
async def detect_image(
    file: UploadFile = File(..., description="Image file (JPEG / PNG)"),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
):
    data = await file.read()
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit")
    return detector.predict_image(data, conf=conf)


@router.post("/detect/video")
async def detect_video(
    file: UploadFile = File(..., description="Video file (MP4 / AVI)"),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
):
    data = await file.read()
    if len(data) > _MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Video exceeds 500 MB limit")
    video_bytes, frames, detections = detector.predict_video(data, conf=conf)
    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "X-Frame-Count": str(frames),
            "X-Total-Detections": str(detections),
            "Content-Disposition": 'attachment; filename="annotated.mp4"',
        },
    )
