"""
RunPod Serverless Handler for Hallo2.
Receives image_url + audio_url, generates lip-synced video, returns base64 or URL.
"""

import os
import sys
import uuid
import base64
import subprocess
import tempfile
import requests
import runpod
import yaml

HALLO2_DIR = "/runpod-volume/hallo2"
VENV_PYTHON = os.path.join(HALLO2_DIR, "venv/bin/python")
VENV_SITE = os.path.join(HALLO2_DIR, "venv/lib/python3.11/site-packages")
CONFIG_PATH = os.path.join(HALLO2_DIR, "configs/inference/long.yaml")
OUTPUT_DIR = "/tmp/hallo2_output"


def download_file(url: str, dest: str):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


def convert_audio_to_wav(input_path: str, output_path: str, start: float = 0, duration: float = 0):
    """Convert audio to WAV format, optionally trimming a segment."""
    cmd = ["ffmpeg", "-y", "-i", input_path]
    if start > 0:
        cmd += ["-ss", str(start)]
    if duration > 0:
        cmd += ["-t", str(duration)]
    cmd += ["-ar", "16000", "-ac", "1", output_path]
    subprocess.run(cmd, check=True, capture_output=True)


def handler(job):
    job_input = job["input"]

    image_url = job_input["image_url"]
    audio_url = job_input["audio_url"]
    audio_start = job_input.get("audio_start", 0)
    duration = job_input.get("duration", 0)
    pose_weight = job_input.get("pose_weight", 1.0)
    face_weight = job_input.get("face_weight", 1.0)
    lip_weight = job_input.get("lip_weight", 1.0)

    job_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        # 1. Download image
        img_path = os.path.join(work_dir, "source.png")
        download_file(image_url, img_path)

        # 2. Download and convert audio
        raw_audio = os.path.join(work_dir, "raw_audio")
        download_file(audio_url, raw_audio)

        wav_path = os.path.join(work_dir, "audio.wav")
        convert_audio_to_wav(raw_audio, wav_path, start=audio_start, duration=duration)

        # 3. Run Hallo2 inference
        output_video = os.path.join(work_dir, "output.mp4")

        cmd = [
            VENV_PYTHON, os.path.join(HALLO2_DIR, "scripts/inference_long.py"),
            "-c", CONFIG_PATH,
            "--source_image", img_path,
            "--driving_audio", wav_path,
            "--pose_weight", str(pose_weight),
            "--face_weight", str(face_weight),
            "--lip_weight", str(lip_weight),
            "--face_expand_ratio", "1.2",
        ]

        env = os.environ.copy()
        env["HALLO2_OUTPUT_DIR"] = work_dir
        env["PYTHONPATH"] = VENV_SITE + ":" + HALLO2_DIR + ":" + env.get("PYTHONPATH", "")
        env["PATH"] = os.path.join(HALLO2_DIR, "venv/bin") + ":" + env.get("PATH", "")

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=HALLO2_DIR, env=env, timeout=600
        )

        if result.returncode != 0:
            return {"error": f"Hallo2 inference failed: {result.stderr[-500:]}"}

        # Find the output video (Hallo2 saves as merge_video.mp4)
        merge_video = None
        for root, dirs, files in os.walk(work_dir):
            for f in files:
                if f.endswith(".mp4"):
                    merge_video = os.path.join(root, f)
                    break
            if merge_video:
                break

        if not merge_video:
            # Check default hallo2 output location
            default_output = os.path.join(HALLO2_DIR, "output")
            for root, dirs, files in os.walk(default_output):
                for f in files:
                    if f.endswith(".mp4"):
                        merge_video = os.path.join(root, f)
                        break
                if merge_video:
                    break

        if not merge_video or not os.path.exists(merge_video):
            return {"error": "No output video found", "stdout": result.stdout[-500:]}

        # 4. Return video as base64
        with open(merge_video, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "video_base64": video_b64,
            "duration_seconds": duration if duration > 0 else "unknown",
        }

    except subprocess.TimeoutExpired:
        return {"error": "Inference timed out (>600s)"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup
        subprocess.run(["rm", "-rf", work_dir], capture_output=True)


runpod.serverless.start({"handler": handler})
