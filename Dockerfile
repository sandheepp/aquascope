# AquaScope — laptop / non-Jetson Docker image.
#
# This image is for trying AquaScope WITHOUT a Jetson. It runs the dashboard
# on CPU (or NVIDIA GPU on Linux hosts via --gpus all) and serves the live
# stream on :8080.
#
# Build:
#   docker build -t aquascope .
#
# Run against a sample video (no webcam needed):
#   docker run --rm -p 8080:8080 -v "$PWD/samples:/app/samples" aquascope \
#     --no-display --video samples/aquarium.mp4
#
# Run against a USB webcam (Linux host only — macOS/Windows can't pass /dev
# through to a Linux container):
#   docker run --rm -p 8080:8080 --device=/dev/video0:/dev/video0 aquascope \
#     --no-display
#
# Then open http://localhost:8080
#
# Note: this image is NOT for the Jetson. On the Jetson use scripts/run_dashboard.sh
# directly so you get the JetPack-tuned PyTorch + TensorRT engines.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV needs a few system libs for video decode + the offscreen Qt platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so subsequent code edits don't bust the layer cache.
COPY scripts/requirements_local.txt /app/scripts/requirements_local.txt
RUN pip install -r /app/scripts/requirements_local.txt

# Copy source. .dockerignore keeps datasets, venvs, and recordings out.
COPY . /app

EXPOSE 8080
ENV QT_QPA_PLATFORM=offscreen

# Default: headless dashboard on port 8080. Override CMD with any flag from
# app/fish_tracker.py (e.g. --video samples/clip.mp4, --camera 1, ...).
ENTRYPOINT ["python", "app/fish_tracker.py"]
CMD ["--no-display"]
