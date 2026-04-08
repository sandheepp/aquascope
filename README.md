# AquaScope

Real-time fish detection and tracking for home aquariums using **NVIDIA Jetson Orin Nano** and a **Logitech C920** webcam.

## Architecture

```
┌──────────────┐     USB      ┌──────────────────┐     TensorRT      ┌──────────────┐
│ Logitech C920 │ ──────────▶ │  Jetson Orin Nano │ ───────────────▶  │  YOLOv8n     │
│   (1280×720)  │             │  (JetPack 5/6)   │                   │  Detection   │
└──────────────┘              └────────┬─────────┘                   └──────┬───────┘
                                       │                                     │
                                       │              ┌──────────────────────┘
                                       │              ▼
                                       │       ┌──────────────┐
                                       │       │  ByteTrack   │
                                       │       │  Tracker     │
                                       │       └──────┬───────┘
                                       │              │
                                       ▼              ▼
                                ┌─────────────────────────────┐
                                │  Annotated Display + Logs   │
                                │  • Fish trails & IDs        │
                                │  • Count & activity stats   │
                                │  • JSON logging             │
                                └─────────────────────────────┘
```

## Why YOLOv8n + ByteTrack?

| Criterion             | YOLOv8n + ByteTrack           | Alternatives                      |
|----------------------|-------------------------------|-----------------------------------|
| **Orin Nano FPS**    | ~30-43 FPS (TensorRT FP16)    | YOLOv8s: ~20 FPS, YOLOv5: ~35    |
| **Accuracy**         | High (mAP 37.3 COCO)         | Comparable for fish detection     |
| **Tracking**         | ByteTrack: low ID switches    | DeepSORT: heavier, slower         |
| **Memory**           | ~1.5 GB                       | YOLOv8m: ~3 GB                    |
| **Ecosystem**        | Ultralytics (batteries incl.) | Manual integration needed         |
| **Fish research**    | Validated in aquaculture      | Limited published results         |

**Key reasons:**
- YOLOv8n is the smallest variant (~3.2M params), ideal for the Orin Nano's 8GB RAM
- TensorRT export gives ~2-3x speedup over PyTorch inference
- ByteTrack outperforms BoT-SORT for fish tracking due to its two-stage matching that handles occlusions well
- The Ultralytics library includes built-in ByteTrack support — zero extra code
- Multiple aquaculture studies validate this exact combo (YOLOv8 + ByteTrack) for fish detection

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
2. Distance: 30-60 cm from the glass works best
3. Avoid reflections: angle slightly downward (5-10°) and use a matte background
4. Ensure even lighting — avoid direct overhead lights causing glare
5. Disable autofocus (done in software) for stable detection

## Software Setup

### Step 1: Flash JetPack

If not already done, flash JetPack 6.1 (or 5.1.3+) to your Jetson:
- Download from: https://developer.nvidia.com/jetson-orin-nano-developer-kit
- Use NVIDIA SDK Manager or balenaEtcher with the SD card image

### Step 2: Run the setup script

```bash
# Copy this project to your Jetson


# SSH into the Jetson
ssh jetson@<jetson-ip>

# Run setup
cd ~/fish_tracker
chmod +x setup_jetson.sh
./setup_jetson.sh
```

This installs all dependencies, sets max performance mode, downloads YOLOv8n,
and exports it to a TensorRT engine.

### Step 3: Run the tracker

```bash
# Basic usage
python3 fish_tracker.py

# With recording
python3 fish_tracker.py --record

# Headless (no display, just logging)
python3 fish_tracker.py --no-display

# Custom confidence threshold
python3 fish_tracker.py --conf 0.4
```

### Keyboard controls
- `q` — Quit
- `s` — Save snapshot
- `r` — Reset all trails

## Fine-Tuning for Your Aquarium (Optional but Recommended)

The default YOLOv8n is trained on COCO (80 classes). It can detect some fish out
of the box, but fine-tuning on aquarium data dramatically improves accuracy.

### Option A: Use Roboflow Aquarium Dataset (quick start)

```bash
# Get a free API key from https://app.roboflow.com
python3 train_fish_model.py download --api-key YOUR_KEY

# Train (runs on Jetson GPU — ~2-4 hours for 100 epochs)
python3 train_fish_model.py train --data datasets/data.yaml --epochs 100
```

### Option B: Label Your Own Fish (best accuracy)

```bash
# 1. Capture images from your tank
python3 train_fish_model.py capture --num 200

# 2. Upload to https://app.roboflow.com and label bounding boxes
#    Label classes: "fish" (or specific species)

# 3. Download in YOLOv8 format

# 4. Train
python3 train_fish_model.py train --data path/to/data.yaml
```

## Output & Logs

The tracker saves JSON logs to `fish_logs/` every 60 seconds:

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

## Performance Tuning

| Setting                        | Command / Action                     | Effect              |
|-------------------------------|--------------------------------------|---------------------|
| Max Jetson performance        | `sudo nvpmodel -m 0 && sudo jetson_clocks` | +20-40% FPS  |
| Use TensorRT FP16             | Export with `half=True`              | ~2x faster          |
| Use TensorRT INT8             | Export with `int8=True`              | ~3x faster (needs calibration) |
| Reduce input size             | `--imgsz 416`                        | +30% FPS, slight accuracy drop |
| Lower camera resolution       | Change `camera_width/height`         | Less CPU overhead   |
| Increase confidence threshold | `--conf 0.5`                         | Fewer false positives |

## Troubleshooting

**Low FPS (<15)?**
- Ensure `nvpmodel -m 0` and `jetson_clocks` are active
- Use TensorRT engine (`.engine`) not PyTorch (`.pt`)
- Close other GPU-intensive apps

**Camera not detected?**
- Run `v4l2-ctl --list-devices` to check
- Try `/dev/video0` or `/dev/video1`
- Ensure USB cable supports data (not charge-only)

**Fish not being detected?**
- Lower confidence: `--conf 0.2`
- Fine-tune on your specific aquarium (see above)
- Check lighting — too dark or too much glare hurts detection

**ID switches (fish swapping IDs)?**
- Increase `track_buffer` in `bytetrack.yaml`
- Lower `match_thresh` for stricter matching
- Fine-tuned model produces more stable detections

## License

This project is provided as-is for personal/educational use.
YOLOv8 is licensed under AGPL-3.0 by Ultralytics.
