FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime deps + git (needed for pip install git+https)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch 2.11 — matches training torch version, supports NumPy 2.x natively
RUN pip install --no-cache-dir \
    torch==2.11.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install the sunsmarterjie/yolov12 fork — the exact codebase used during training.
# The official PyPI ultralytics 8.3.63 does NOT have A2C2f/AAttn (YOLOv12 blocks);
# the fork does and reports itself as version 8.3.63.
RUN pip install --no-cache-dir \
    git+https://github.com/sunsmarterjie/yolov12.git \
    "opencv-python-headless>=4.9.0" \
    "timm>=1.0.0"

# API layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
