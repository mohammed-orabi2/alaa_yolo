FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# OpenCV runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (avoids downloading the ~2 GB CUDA build)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Core inference deps
RUN pip install --no-cache-dir \
    "ultralytics>=8.3" \
    "opencv-python-headless>=4.9.0" \
    "timm>=1.0.0"

# API layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Downgrade numpy LAST — torch 2.2.2 was compiled against NumPy 1.x ABI
RUN pip install --no-cache-dir "numpy<2"

COPY app/ ./app/

RUN mkdir -p models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
