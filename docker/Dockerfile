# ============================================================
# Traffic Surveillance System Dockerfile
# Build targets:
#   CPU (default):  docker build --target cpu -t traffic-surv:cpu .
#   GPU:            docker build --target gpu -t traffic-surv:gpu .
# ============================================================

# ── Shared base ─────────────────────────────────────────────
FROM python:3.12-slim AS base
LABEL maintainer="your@email.com"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for OpenCV headless + EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Install CPU-only torch first (smaller image)
RUN pip install torch==2.4.1+cpu torchvision==0.19.1+cpu \
      --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-deps ultralytics==8.3.40 && \
    pip install -r requirements.txt

COPY . .
RUN mkdir -p output/snapshots output/clips models

# ── CPU target ──────────────────────────────────────────────
FROM base AS cpu
ENV DEVICE=cpu
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]

# ── GPU target ───────────────────────────────────────────────
# Build: docker build --target gpu -t traffic-surv:gpu .
# Run:   docker run --gpus all -v $(pwd)/output:/app/output traffic-surv:gpu run --source /data/video.mp4
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3-pip \
    libgl1 libglib2.0-0 libgomp1 ffmpeg \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch==2.4.1 torchvision==0.19.1 \
      --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements.txt

COPY . .
RUN mkdir -p output/snapshots output/clips models

FROM gpu-base AS gpu
ENV DEVICE=cuda
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
