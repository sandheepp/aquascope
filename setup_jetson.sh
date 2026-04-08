#!/bin/bash
# ============================================================
#  AquaScope — Jetson Orin Nano Setup Script
#  Prerequisites: JetPack 5.x or 6.x installed on Orin Nano
# ============================================================

set -e
echo "╔══════════════════════════════════════════════════╗"
echo "║   AquaScope — Jetson Setup           ║"
echo "╚══════════════════════════════════════════════════╝"

# ── 1. System packages ──
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
    nvidia-l4t-core \
    cuda-toolkit-12-6 \
    libcudnn9-cuda-12 \
    nvidia-tensorrt \
    tensorrt \
    tensorrt-libs \
    nvidia-jetpack-runtime

echo ""
echo "▸ [1.1/6] Verifying Jetson runtime stack..."
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

echo ""
echo "▸ [1.2/6] Verifying JetPack package..."
dpkg -l | grep -E 'nvidia-jetpack|nvidia-jetpack-runtime' || echo "  ⚠ nvidia-jetpack package not installed"

# ── 2. Set Jetson to max performance ──
echo ""
echo "▸ [2/6] Setting Jetson to max performance mode..."
sudo nvpmodel -m 0          # MAX power mode
sudo jetson_clocks           # Lock clocks to max frequency
echo "  ✓ Performance mode set (15W, clocks maxed)"

# ── 3. Python packages ──
echo ""
echo "▸ [3/6] Installing Python packages..."
# The apt python3-torch is a CPU-only legacy build — remove it if present
sudo apt-get remove -y python3-torch python3-torchvision python3-torchaudio 2>/dev/null || true
python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

pip3 install --upgrade pip

# Install NVIDIA's CUDA-enabled PyTorch for Jetson
# pip indexes serve CPU-only aarch64 wheels — Jetson needs cuDNN-linked wheels
# from the NVIDIA forum: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
#
# cuDNN version must match the wheel:
#   JetPack 6.1 (L4T R36.4+): cuDNN 9 → PyTorch 2.5.0 wheel
#   JetPack 6.0 (L4T R36.3):  cuDNN 8 → PyTorch 2.3.0 wheel
#   JetPack 5.x (L4T R35.x):  cuDNN 8 → PyTorch 2.1.0 wheel
#
# Check your revision: grep REVISION /etc/nv_tegra_release
echo ""
echo "  ⚠ PyTorch for Jetson must be installed manually from the NVIDIA forum."
echo "  The pip indexes serve CPU-only wheels that will not work with CUDA."
echo ""
echo "  1. Go to: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
echo "  2. Download the wheel matching your JetPack version and Python 3.10"
echo "  3. Run: pip3 install <downloaded-wheel>.whl"
echo ""
echo "  Then re-run this script to continue setup."
echo ""
echo "  Current L4T version:"
cat /etc/nv_tegra_release | grep -E "^# R"
echo "  Current cuDNN:"
dpkg -l | grep libcudnn | awk '{print $2, $3}' || echo "  (none installed)"
echo ""
read -p "  Press Enter once PyTorch is installed, or Ctrl+C to exit..." _dummy

# Verify CUDA is available before continuing
python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: torch.cuda.is_available() is False.')
    print('Install the correct Jetson PyTorch wheel before continuing.')
    sys.exit(1)
print(f'  ✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')
"

pip3 install "numpy<2"                 # Must stay below 2.0 for Jetson PyTorch ABI compatibility
pip3 install --no-deps ultralytics

# ── 4. Verify camera ──
echo ""
echo "▸ [4/6] Checking Logitech C920 camera..."
if v4l2-ctl --list-devices 2>/dev/null | grep -i "logitech\|C920\|video0"; then
    echo "  ✓ Camera detected"
    v4l2-ctl -d /dev/video0 --list-formats-ext 2>/dev/null | head -20
else
    echo "  ⚠ Camera not detected. Plug in your Logitech C920 and retry."
    echo "    Run: v4l2-ctl --list-devices"
fi

# ── 5. Download YOLOv8 model and export to TensorRT ──
echo ""
echo "▸ [5/6] Downloading YOLOv8n model..."
cd "$(dirname "$0")"

python3 -c "
from ultralytics import YOLO
print('Downloading YOLOv8n...')
model = YOLO('yolov8n.pt')
print('✓ Model downloaded')
print()
print('Exporting to TensorRT (FP16)...')
print('This will take several minutes on first run.')
model.export(format='engine', device=0, half=True, imgsz=640)
print('✓ TensorRT engine created: yolov8n.engine')
"

# ── 6. Create output directory ──
echo ""
echo "▸ [6/6] Creating output directories..."
mkdir -p fish_logs

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Setup Complete!                                ║"
echo "║                                                  ║"
echo "║   Run the tracker:                               ║"
echo "║   python3 fish_tracker.py                        ║"
echo "║                                                  ║"
echo "║   Or with recording:                             ║"
echo "║   python3 fish_tracker.py --record               ║"
echo "║                                                  ║"
echo "║   Fine-tuning (optional):                        ║"
echo "║   python3 train_fish_model.py                    ║"
echo "╚══════════════════════════════════════════════════╝"
