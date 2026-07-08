# Traffic Surveillance System Dockerfile
# Build CPU: docker build --target cpu -t traffic-surv:cpu -f docker/Dockerfile .
# Build GPU: docker build --target gpu -t traffic-surv:gpu -f docker/Dockerfile .

FROM python:3.12-slim AS base
LABEL org.opencontainers.image.title="MS-ADA Traffic Accident Detection"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch==2.4.1+cpu torchvision==0.19.1+cpu \
      --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-deps ultralytics==8.3.40 && \
    pip install -r requirements.txt

COPY . .
RUN mkdir -p output/snapshots output/clips models

FROM base AS cpu
ENV DEVICE=cpu
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]

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
