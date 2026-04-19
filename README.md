# YOLOv12 Egyptian Food Detection API

Real-time object detection for Egyptian food products in shopping carts.  
Powered by YOLOv12m — 30 product classes, image & video support, REST API.

---

## Project Structure

```
alaa-yolo/
├── app/
│   ├── main.py                  # FastAPI app & model startup
│   ├── api/routes.py            # HTTP endpoints
│   ├── services/detector.py     # YOLO inference logic
│   └── schemas/response.py      # Pydantic response models
├── models/                      # Place best.pt here (git-ignored, Docker volume)
├── egyptian_food/
│   └── yolov12m_v16_video/      # Latest training run outputs & weights
│       └── weights/best.pt      # Trained model — copy to models/
├── yolov12m.pt                  # Pretrained YOLOv12m backbone
├── yolov12n.pt                  # Pretrained YOLOv12n backbone
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/health` | Health check |
| `POST` | `/api/v1/detect/image` | Detect objects in an image |
| `POST` | `/api/v1/detect/video` | Detect objects in a video |

### Query Parameters (both detect endpoints)

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `conf` | float | `0.25` | Confidence threshold (0–1) |

### Image Response (`/detect/image`)

```json
{
  "count": 3,
  "detections": [
    {
      "class_id": 7,
      "class_name": "Pepsi",
      "confidence": 0.9231,
      "bbox": { "x1": 120.5, "y1": 88.0, "x2": 220.3, "y2": 310.1 }
    }
  ],
  "annotated_image": "<base64 JPEG string>"
}
```

### Video Response (`/detect/video`)

Returns the annotated video as `video/mp4` with headers:
- `X-Frame-Count` — number of frames processed
- `X-Total-Detections` — sum of all detections across all frames

---

## Quick Start (Local — no Docker)

```bash
# 1. Install CPU-only PyTorch
pip install torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Install inference + API dependencies
pip install "ultralytics>=8.3" "opencv-python-headless>=4.9.0" "timm>=1.0.0"
pip install -r requirements.txt

# 3. Copy the trained model weights
cp egyptian_food/yolov12m_v16_video/weights/best.pt models/best.pt

# 4. Run
MODEL_PATH=models/best.pt uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

### Test with curl

```bash
# Image detection
curl -X POST "http://localhost:8000/api/v1/detect/image?conf=0.3" \
  -F "file=@/path/to/image.jpg" | jq .count

# Video detection — saves annotated output
curl -X POST "http://localhost:8000/api/v1/detect/video?conf=0.25" \
  -F "file=@/path/to/video.mp4" \
  --output annotated.mp4
```

---

## Training Phase

### 1. Dataset Setup

The training dataset is managed via [Roboflow](https://roboflow.com).  
The model was trained on a custom shopping cart dataset with 30 product classes.

```bash
pip install roboflow

python - <<'EOF'
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("YOUR_PROJECT_SLUG")
project.version(3).download("yolov8")   # downloads to ./cart-version-3/
EOF
```

### 2. Install Training Dependencies

```bash
pip install "ultralytics>=8.3" torch torchvision timm flash-attn --no-build-isolation
```

> `flash-attn` is optional but speeds up YOLOv12's attention layers on GPU.

### 3. Run Training

```python
from ultralytics import YOLO

model = YOLO("yolov12m.pt")   # pretrained backbone (already in project root)

model.train(
    data="cart-version-3/data.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    optimizer="AdamW",
    lr0=0.0002,
    patience=30,
    amp=True,           # mixed precision — GPU only
    device=0,           # GPU id; use "cpu" for CPU training
    project="egyptian_food",
    name="yolov12m_v17",
    save_period=10,     # checkpoint every 10 epochs
)
```

Outputs land in `egyptian_food/yolov12m_v17/`:
- `weights/best.pt` — best checkpoint (copy to `models/best.pt` to deploy)
- Precision / Recall / F1 curves, confusion matrix, `results.csv`

### 4. Evaluate

```python
from ultralytics import YOLO

model = YOLO("egyptian_food/yolov12m_v17/weights/best.pt")
metrics = model.val(data="cart-version-3/data.yaml", split="test")
print(metrics.box.map)   # mAP@50
```

### 5. Export (optional — ONNX / edge devices)

```python
model.export(format="onnx", imgsz=640, opset=17)
```

---

## Deployment Phase

### Option A — Local Docker Compose

```bash
# 1. Copy trained weights into models/
cp egyptian_food/yolov12m_v16_video/weights/best.pt models/best.pt

# 2. Build image
docker compose build

# 3. Start container
docker compose up -d

# 4. Verify
curl http://localhost:8000/api/v1/health
```

---

### Option B — AWS EC2 (Step-by-Step)

#### Step 1 — Launch an EC2 Instance

1. Open **AWS Console → EC2 → Launch Instance**
2. **Name**: `yolov12-api`
3. **AMI**: Ubuntu Server 22.04 LTS (64-bit x86)
4. **Instance type**:
   - CPU inference → `t3.medium` (2 vCPU, 4 GB RAM, ~$0.04/hr)
   - GPU inference → `g4dn.xlarge` (4 vCPU, 16 GB, T4 GPU, ~$0.53/hr)
5. **Key pair**: create or select an existing `.pem` key
6. **Security group** — allow inbound:

   | Type | Port | Source |
   |------|------|--------|
   | SSH | 22 | Your IP |
   | Custom TCP | 8000 | 0.0.0.0/0 |
   | HTTP | 80 | 0.0.0.0/0 (if using Nginx) |

7. **Storage**: 30 GB gp3
8. Click **Launch Instance** and wait ~1 minute

#### Step 2 — Connect via SSH

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

#### Step 3 — Install Docker on EC2

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker ubuntu
newgrp docker

docker --version   # verify
```

#### Step 4 — Transfer Project Files to EC2

Run this **on your local machine**:

```bash
rsync -avz --progress \
  --exclude='.git' \
  --exclude='egyptian_food/*/weights/epoch*.pt' \
  --exclude='*.ipynb' \
  -e "ssh -i your-key.pem" \
  /home/ahmed/alaa-yolo/ \
  ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/alaa-yolo/
```

The `models/best.pt` file is excluded from git (see `.gitignore`), so transfer it separately:

```bash
scp -i your-key.pem \
  /home/ahmed/alaa-yolo/egyptian_food/yolov12m_v16_video/weights/best.pt \
  ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/alaa-yolo/models/best.pt
```

#### Step 5 — Build and Run on EC2

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

cd /home/ubuntu/alaa-yolo

docker compose build        # ~5–10 min on first run
docker compose up -d

docker compose ps
curl http://localhost:8000/api/v1/health
# → {"status":"ok","model_loaded":true}
```

#### Step 6 — Test from Your Machine

```bash
curl http://<EC2_PUBLIC_IP>:8000/api/v1/health

curl -X POST "http://<EC2_PUBLIC_IP>:8000/api/v1/detect/image?conf=0.3" \
  -F "file=@test.jpg" | jq .count
```

#### Step 7 (Optional) — Nginx Reverse Proxy on Port 80

```bash
sudo apt-get install -y nginx

sudo tee /etc/nginx/sites-available/yolov12-api > /dev/null <<'EOF'
server {
    listen 80;
    client_max_body_size 600M;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/yolov12-api \
           /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

API is now reachable on port 80 (no port number in the URL).

#### Step 8 (Optional) — HTTPS with Let's Encrypt

Requires a domain pointed at the EC2 IP.

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

#### Step 9 — Auto-restart on Reboot

`restart: unless-stopped` in `docker-compose.yml` handles container restart.  
Enable Docker itself to start on boot:

```bash
sudo systemctl enable docker
```

---

## Useful Commands

```bash
# Live logs
docker compose logs -f

# Stop
docker compose down

# Rebuild after code changes
docker compose build && docker compose up -d

# Shell inside container
docker compose exec api bash
```

---

## Limitations & Production Notes

| Concern | Current behaviour | Recommendation |
|---------|------------------|----------------|
| Concurrency | Single worker — video blocks all requests | Add a task queue (Celery + Redis) |
| Video size | 500 MB hard limit | Use S3 pre-signed URLs for large files |
| GPU support | CPU-only Docker image | Change base to `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` and remove the CPU wheel |
| Model updates | Copy new `best.pt` to `models/` and restart container | No rebuild needed — model is a mounted volume |
