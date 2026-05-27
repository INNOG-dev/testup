FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3.10 python3.10-venv python3.10-dev python3-pip ffmpeg wget curl \
    build-essential libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

# PyTorch (CUDA 11.8)
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Clone Hallo2
RUN git clone --depth 1 https://github.com/fudan-generative-vision/hallo2.git /app/hallo2

WORKDIR /app/hallo2

# Python deps
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod requests

# Download pretrained models (~20GB) — baked into image for fast cold starts
RUN pip install --no-cache-dir huggingface_hub && \
    huggingface-cli download fudan-generative-ai/hallo2 --local-dir ./pretrained_models

# Copy handler
COPY handler.py /app/hallo2/handler.py

CMD ["python", "handler.py"]
