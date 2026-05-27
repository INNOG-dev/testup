#!/bin/bash
set -e

echo "=== Installing Hallo2 on Network Volume ==="
cd /workspace

echo "=== Step 1: System deps ==="
apt-get update -qq && apt-get install -y -qq ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev build-essential >/dev/null 2>&1
echo "Done."

echo "=== Step 2: Clone Hallo2 ==="
if [ -d "/workspace/hallo2" ]; then
  echo "Already cloned, pulling latest..."
  cd /workspace/hallo2 && git pull
else
  git clone --depth 1 https://github.com/fudan-generative-vision/hallo2.git /workspace/hallo2
fi
cd /workspace/hallo2

echo "=== Step 3: Python venv ==="
python3 -m venv /workspace/hallo2/venv
source /workspace/hallo2/venv/bin/activate

echo "=== Step 4: PyTorch ==="
pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

echo "=== Step 5: Requirements ==="
pip install --no-cache-dir -r requirements.txt

echo "=== Step 6: RunPod SDK ==="
pip install --no-cache-dir runpod requests

echo "=== Step 7: Download models (~20GB) ==="
pip install --no-cache-dir huggingface_hub
huggingface-cli download fudan-generative-ai/hallo2 --local-dir ./pretrained_models

echo "=== Step 8: Install handler ==="
cp /workspace/testup/handler.py /workspace/hallo2/handler.py

echo "=================================="
echo "=== INSTALLATION COMPLETE! ==="
echo "=================================="
echo ""
echo "To test: cd /workspace/hallo2 && source venv/bin/activate && python handler.py"
