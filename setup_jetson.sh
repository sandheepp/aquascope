#!/bin/bash
# ============================================================
#  AquaScope — Jetson Orin Nano Setup Script
#  Tested on: JetPack 6.1 (L4T R36.5), CUDA 12.6, Python 3.10
# ============================================================

set -e
echo "╔══════════════════════════════════════════════════╗"
echo "║   AquaScope — Jetson Setup                       ║"
echo "╚══════════════════════════════════════════════════╝"

# ── 1. System packages ────────────────────────────────────
echo ""
echo "▸ [1/6] Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    v4l-utils \
    libv4l-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    libcudnn9-cuda-12 \
    nvidia-tensorrt \
    tensorrt \
    tensorrt-libs \
    git

echo ""
echo "▸ [1.1/6] Installing cuSPARSELt (required by torch 2.5.0)..."
# cuSPARSELt is not bundled in JetPack — must be installed from CUDA repo
if ! dpkg -l libcusparselt0 &>/dev/null; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    rm -f cuda-keyring_1.1-1_all.deb
fi
sudo apt-get install -y libcusparselt0 libcusparselt-dev
echo "  ✓ cuSPARSELt installed"

echo ""
echo "▸ [1.2/6] Verifying Jetson runtime stack..."
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | head -n 2
else
    echo "  ⚠ nvcc not found"
fi
if command -v trtexec >/dev/null 2>&1; then
    trtexec --version | head -n 1
else
    echo "  ⚠ trtexec not found"
fi
echo "  L4T version: $(cat /etc/nv_tegra_release | grep -oP 'R\d+ \(release\), REVISION: [\d.]+')"
echo "  cuDNN: $(dpkg -l | grep libcudnn9-cuda | awk '{print $3}' || echo 'not found')"

# ── 2. Set Jetson to max performance ─────────────────────
echo ""
echo "▸ [2/6] Setting Jetson to max performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks
echo "  ✓ Performance mode set (15W, clocks maxed)"

# ── 3. Python packages ────────────────────────────────────
echo ""
echo "▸ [3/6] Installing Python packages..."

# Remove any CPU-only apt/pip torch builds that shadow the Jetson wheel
sudo apt-get remove -y python3-torch python3-torchvision python3-torchaudio 2>/dev/null || true
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true

pip3 install --upgrade pip

# ── PyTorch: exact Jetson wheel (JetPack 6.1 / CUDA 12.6 / cuDNN 9 / Python 3.10) ──
TORCH_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
echo "  Installing torch 2.5.0 Jetson wheel..."
pip3 install --no-cache "$TORCH_WHEEL"

# ── torchvision: pip install (jetson_compat.py patches the NMS ops at runtime) ──
echo "  Installing torchvision 0.26.0..."
pip3 install torchvision==0.26.0 --no-deps

# ── Verify CUDA is available ──
python3 -c "
import torch, sys
print(f'  torch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('  ⚠ CUDA not available — check driver and cuSPARSELt installation')
else:
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

# ── pip-installable packages from requirements.txt ──
echo "  Installing requirements.txt packages..."
pip3 install \
    numpy==1.26.4 \
    ultralytics==8.3.0 \
    sympy==1.13.1 \
    filelock==3.25.2 \
    fsspec==2026.3.0 \
    "Jinja2==3.1.6" \
    "MarkupSafe==2.0.1" \
    mpmath==1.3.0 \
    networkx==3.4.2 \
    "typing_extensions==4.15.0" \
    "lap==0.5.13" \
    "Pillow==9.0.1" \
    "PyYAML==5.4.1" \
    "requests==2.25.1" \
    "tqdm==4.67.3" \
    "matplotlib==3.5.1" \
    "pandas==1.3.5" \
    "seaborn==0.13.2" \
    "psutil==7.2.2" \
    "scipy==1.10.1" \
    "py-cpuinfo==9.0.0" \
    "ultralytics-thop==2.0.18" \
    "pyngrok==8.0.0" \
    "jetson-stats==4.3.2"

echo "  ✓ Python packages installed"

# ── 4. Verify camera ──────────────────────────────────────
echo ""
echo "▸ [4/6] Checking Logitech C920 camera..."
if v4l2-ctl --list-devices 2>/dev/null | grep -i "logitech\|C920\|video0"; then
    echo "  ✓ Camera detected"
    v4l2-ctl -d /dev/video0 --list-formats-ext 2>/dev/null | head -20
else
    echo "  ⚠ Camera not detected. Plug in your Logitech C920 and retry."
    echo "    Run: v4l2-ctl --list-devices"
fi

# ── 5. Download YOLOv8n model and export to TensorRT ──────
echo ""
echo "▸ [5/6] Downloading YOLOv8n model and exporting to TensorRT..."
cd "$(dirname "$0")"

python3 -c "
import os
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
print('✓ Model downloaded: yolov8n.pt')

engine = 'yolov8n.engine'
if os.path.exists(engine):
    print(f'✓ TensorRT engine already exists: {engine}')
else:
    print('Exporting to TensorRT FP16 (this takes several minutes)...')
    model.export(format='engine', device=0, half=True, imgsz=416)
    print(f'✓ TensorRT engine created: {engine}')
"

# ── 6. Create output directory ────────────────────────────
echo ""
echo "▸ [6/6] Creating output directories..."
mkdir -p fish_logs
echo "  ✓ fish_logs/ created"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Setup Complete!                                ║"
echo "║                                                  ║"
echo "║   Run the tracker:                               ║"
echo "║   python3 fish_tracker.py --no-display --stream  ║"
echo "║                                                  ║"
echo "║   With TensorRT engine:                          ║"
echo "║   python3 fish_tracker.py --model yolov8n.engine ║"
echo "║                                                  ║"
echo "║   With public stream:                            ║"
echo "║   python3 fish_tracker.py --stream --public      ║"
echo "╚══════════════════════════════════════════════════╝"
