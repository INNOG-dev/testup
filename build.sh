#!/bin/bash
set -e

echo "=== Step 1: Creating build files ==="
mkdir -p /workspace/build
cd /workspace/build

cat > handler.py << 'PYEOF'
import os, sys, uuid, base64, subprocess, requests, runpod

HALLO2_DIR = "/app/hallo2"
CONFIG = os.path.join(HALLO2_DIR, "configs/inference/long.yaml")

def dl(url, dest):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)

def handler(job):
    ji = job["input"]
    jid = str(uuid.uuid4())[:8]
    wd = f"/tmp/h2/{jid}"
    os.makedirs(wd, exist_ok=True)
    try:
        img = f"{wd}/src.png"
        dl(ji["image_url"], img)
        raw = f"{wd}/raw"
        dl(ji["audio_url"], raw)
        wav = f"{wd}/a.wav"
        cmd = ["ffmpeg", "-y", "-i", raw]
        if ji.get("audio_start", 0) > 0:
            cmd += ["-ss", str(ji["audio_start"])]
        if ji.get("duration", 0) > 0:
            cmd += ["-t", str(ji["duration"])]
        cmd += ["-ar", "16000", "-ac", "1", wav]
        subprocess.run(cmd, check=True, capture_output=True)

        r = subprocess.run(
            [sys.executable, f"{HALLO2_DIR}/scripts/inference_long.py",
             "-c", CONFIG,
             "--source_image", img,
             "--driving_audio", wav,
             "--pose_weight", str(ji.get("pose_weight", 1.0)),
             "--face_weight", str(ji.get("face_weight", 1.0)),
             "--lip_weight", str(ji.get("lip_weight", 1.0)),
             "--face_expand_ratio", "1.2"],
            capture_output=True, text=True, cwd=HALLO2_DIR, timeout=600
        )
        if r.returncode != 0:
            return {"error": r.stderr[-500:]}

        vid = None
        for rt, _, fs in os.walk(wd):
            for f in fs:
                if f.endswith(".mp4"):
                    vid = os.path.join(rt, f)
                    break
            if vid:
                break
        if not vid:
            for rt, _, fs in os.walk(f"{HALLO2_DIR}/output"):
                for f in fs:
                    if f.endswith(".mp4"):
                        vid = os.path.join(rt, f)
                        break
                if vid:
                    break
        if not vid:
            return {"error": "No output video"}
        with open(vid, "rb") as f:
            return {"video_base64": base64.b64encode(f.read()).decode()}
    except Exception as e:
        return {"error": str(e)}
    finally:
        subprocess.run(["rm", "-rf", wd], capture_output=True)

runpod.serverless.start({"handler": handler})
PYEOF

cat > Dockerfile << 'DFEOF'
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git python3.10 python3.10-venv python3.10-dev python3-pip ffmpeg wget curl build-essential libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
WORKDIR /app
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
RUN git clone --depth 1 https://github.com/fudan-generative-vision/hallo2.git /app/hallo2
WORKDIR /app/hallo2
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod requests
RUN pip install --no-cache-dir huggingface_hub && huggingface-cli download fudan-generative-ai/hallo2 --local-dir ./pretrained_models
COPY handler.py /app/hallo2/handler.py
CMD ["python", "handler.py"]
DFEOF

echo "=== Step 2: Building Docker image ==="
docker build -t berkanozmen/hallo2-runpod:latest .

echo "=== Step 3: Login to Docker Hub ==="
docker login -u berkanozmen

echo "=== Step 4: Pushing image ==="
docker push berkanozmen/hallo2-runpod:latest

echo "==============================="
echo "=== ALL DONE! Image pushed! ==="
echo "==============================="
