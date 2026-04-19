import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.services.detector import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("MODEL_PATH", "models/best.pt")
    load_model(model_path)
    yield


app = FastAPI(
    title="YOLOv12 Detection API",
    description="Object detection for Egyptian food products in shopping carts",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
