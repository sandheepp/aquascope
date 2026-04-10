# AquaScope

Real-time fish detection and tracking for home aquariums using **NVIDIA Jetson Orin Nano** and a **Logitech C920** webcam.

## Project Structure

```
aquascope/
├── app/                        # Core tracker application
│   ├── fish_tracker.py         # Entry point (CLI)
│   ├── tracker.py              # Main tracking loop (YOLOv8s + ByteTrack)
│   ├── stream.py               # MJPEG HTTP server + dashboard UI
│   ├── camera.py               # Camera initialisation (V4L2)
│   ├── config.py               # Default config & constants
│   ├── jetson_compat.py        # torchvision NMS patch for Jetson
│   └── bytetrack.yaml          # ByteTrack tuning reference
├── training/                   # Model fine-tuning & analysis
│   ├── quick_train.py          # Quick YOLOv8s training script
│   ├── train_fish_model.py     # Roboflow download + full training pipeline
│   └── analyse_training.py     # Analyse training_mem.csv + plot graphs
├── monitoring/                 # Jetson resource monitoring
│   └── jetson_monitor.py       # Logs CPU/GPU/RAM/temp to CSV during training
├── scripts/                    # Deployment & setup
│   ├── setup_jetson.sh         # One-shot Jetson setup (run once after flash)
│   └── deploy_to_jetson.sh     # rsync project to Jetson over LAN
├── requirements.txt
├── .env                        # Local secrets (gitignored)
└── fish_logs/                  # Runtime output: JSON stats + screenshots
```

## Architecture

```
┌──────────────┐     USB      ┌──────────────────┐     TensorRT      ┌──────────────┐
│ Logitech C920 │ ──────────▶ │  Jetson Orin Nano │ ───────────────▶  │  YOLOv8s     │
│   (1280×720)  │             │  (JetPack 6.1)   │                   │  Detection   │
└──────────────┘              └────────┬─────────┘                   └──────┬───────┘
                                       │                                     │
                                       │              ┌──────────────────────┘
                                       │              ▼
                                       │       ┌──────────────┐
                                       │       │  ByteTrack   │
                                       │       │  (supervision)│
                                       │       └──────┬───────┘
                                       │              │
                                       ▼              ▼
                                ┌─────────────────────────────┐
                                │  Dashboard (MJPEG stream)   │
                                │  • Fish trails & IDs        │
                                │  • Count & activity stats   │
                                │  • Screenshots / filmstrip  │
                                │  • Cloudflare public URL    │
                                └─────────────────────────────┘
```

## Hardware Setup

### What you need
- NVIDIA Jetson Orin Nano Developer Kit (8GB recommended)
- Logitech C920 HD Pro webcam
- MicroSD card (64GB+ recommended)
- USB-A to USB-A cable (for C920)
- Monitor + keyboard for initial setup (or SSH)
- Power supply (included with Jetson kit)

### Camera mounting
1. Position the C920 facing the front glass of your aquarium
2. Distance: 30–60 cm from the glass works best
3. Avoid reflections: angle slightly downward (5–10°) and use a matte background
4. Ensure even lighting — avoid direct overhead lights causing glare

## Software Setup

### Step 1: Flash JetPack

If not already done, flash JetPack 6.1 (or 5.1.3+) to your Jetson:
- Download from: https://developer.nvidia.com/jetson-orin-nano-developer-kit
- Use NVIDIA SDK Manager or balenaEtcher with the SD card image

### Step 2: Deploy code to Jetson

From your **host machine** (Mac/Linux):

```bash
# Configure connection in .env (copy from .env.example)
cp .env.example .env
# Edit: JETSON_HOST, JETSON_USER, JETSON_PASSWORD, JETSON_PATH

# Deploy project files to Jetson
bash scripts/deploy_to_jetson.sh
```

### Step 3: Run setup on Jetson

SSH into the Jetson, then:

```bash
ssh jetson@<jetson-ip>
cd ~/projects/aquascope

# One-time setup: installs all dependencies, downloads YOLOv8s, exports TensorRT engine
bash scripts/setup_jetson.sh
```

This installs all Python packages, sets max performance mode, downloads YOLOv8s,
and exports it to a TensorRT FP16 engine.

### Step 4: Run the tracker

All commands are run from the **project root** (`~/projects/aquascope`):

```bash
# Local display (requires monitor)
python3 app/fish_tracker.py

# Headless + browser dashboard (recommended for SSH)
python3 app/fish_tracker.py --no-display --stream

# With TensorRT engine (~2× faster)
python3 app/fish_tracker.py --model yolov8s.engine --no-display --stream

# Public URL via Cloudflare tunnel
python3 app/fish_tracker.py --model yolov8s.engine --no-display --stream --public

# SAHI sliced inference (better small-fish recall, lower FPS)
python3 app/fish_tracker.py --sahi --no-display --stream

# Custom confidence + resolution
python3 app/fish_tracker.py --conf 0.3 --resolution 1080p --no-display --stream

# Manual camera exposure (e.g. dimly lit tank)
python3 app/fish_tracker.py --exposure -6 --no-display --stream
```

Open the dashboard at `http://<jetson-ip>:8080` in your browser.

### Keyboard controls (local display mode only)
- `q` — Quit
- `r` — Reset all trails

## Training (Optional)

Fine-tuning on aquarium data dramatically improves detection accuracy
over the default COCO-trained YOLOv8s.

### Option A: Quick train (Roboflow dataset)

```bash
# Download dataset (get free API key from https://app.roboflow.com)
python3 training/train_fish_model.py download --api-key YOUR_KEY

# Fine-tune (~2–4 hours on Jetson for 50 epochs)
python3 training/quick_train.py

# Export best weights to TensorRT after training
python3 -c "
from ultralytics import YOLO
YOLO('runs/detect/train/weights/best.pt') \
  .export(format='engine', device=0, half=True, imgsz=640)
"

# Run with custom model
python3 app/fish_tracker.py --model runs/detect/train/weights/best.engine \
  --no-display --stream
```

### Option B: Label your own fish

```bash
# Capture images from your tank
python3 training/train_fish_model.py capture --num 200

# Upload to https://app.roboflow.com, label bounding boxes, download YOLOv8 format
# Then train:
python3 training/train_fish_model.py train --data path/to/data.yaml
```

### Monitor Jetson resources during training

```bash
# In a separate terminal — logs CPU/GPU/RAM/temp to CSV
python3 monitoring/jetson_monitor.py --interval 2 --output training_mem.csv

# After training, analyse the log and generate graphs
python3 training/analyse_training.py --csv training_mem.csv
```

## Output & Logs

The tracker saves JSON stats to `fish_logs/` every 60 seconds:

```json
{
  "timestamp": "20260403_143022",
  "total_frames": 1800,
  "unique_fish": 5,
  "fish": {
    "1": {
      "first_seen": "2026-04-03T14:29:22",
      "last_seen": "2026-04-03T14:30:22",
      "total_distance_px": 4521.7,
      "frame_count": 1650
    }
  }
}
```

Screenshots taken via the dashboard are saved to `fish_logs/screenshots/`.

## Performance Tuning

| Setting                        | Command / Action                              | Effect              |
|-------------------------------|-----------------------------------------------|---------------------|
| Max Jetson performance        | `sudo nvpmodel -m 0 && sudo jetson_clocks`    | +20–40% FPS         |
| Use TensorRT FP16             | `--model yolov8s.engine`                      | ~2× faster          |
| Reduce input size             | `--imgsz 416`                                 | +30% FPS            |
| Lower resolution              | `--resolution 480p`                           | Less CPU overhead   |
| Higher confidence threshold   | `--conf 0.5`                                  | Fewer false positives|
| Stream quality                | `--stream-quality 70`                         | Lower bandwidth     |

## Troubleshooting

**Low FPS (<15)?**
- Ensure `nvpmodel -m 0` and `jetson_clocks` are active
- Use TensorRT engine (`--model yolov8s.engine`), not PyTorch `.pt`
- Close other GPU-intensive apps

**Camera not detected?**
- Run `v4l2-ctl --list-devices` to check
- Try `--camera 1` if video0 is taken
- Ensure USB cable supports data (not charge-only)

**Fish not being detected?**
- Lower confidence: `--conf 0.2`
- Try SAHI: `--sahi` (better recall for small fish)
- Fine-tune on your specific aquarium (see Training section above)
- Check lighting — too dark or too much glare hurts detection

**ID switches (fish swapping IDs)?**
- Increase `track_buffer` in `app/bytetrack.yaml`
- A fine-tuned model produces more stable detections

## License

This project is provided as-is for personal/educational use.
YOLOv8 is licensed under AGPL-3.0 by Ultralytics.
