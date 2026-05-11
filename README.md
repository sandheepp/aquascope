# AquaScope

Real-time fish & shrimp detection, tracking, and on-device fine-tuning for
home aquariums. Runs on an **NVIDIA Jetson Orin Nano** with a **Logitech
C920** webcam and a browser dashboard for viewing the stream, labeling
detections, and retraining the model — all without leaving the page.

## Project Structure

```
aquascope/
├── app/                        # Core tracker application
│   ├── fish_tracker.py         # CLI entry point
│   ├── tracker.py              # Capture + inference + ByteTrack loop
│   ├── stream.py               # MJPEG server + dashboard UI + labeling/training endpoints
│   ├── model.py                # Model loader (ultralytics .pt / TensorRT .engine)
│   ├── camera.py               # V4L2 camera initialisation
│   ├── enhancer.py             # CLAHE frame enhancer (toggleable from dashboard)
│   ├── config.py               # Default config + trail colour palette
│   ├── jetson_compat.py        # torchvision NMS shim for JetPack
│   └── bytetrack.yaml          # ByteTrack tuning reference
├── training/
│   ├── train_jetson.py         # Jetson trainer (used by the dashboard "Train Model" button)
│   ├── mac_train.py            # Mac trainer (Apple Silicon / MPS, exports CoreML)
│   └── analyse_training.py     # Plot CPU/GPU/RAM curves from a monitor CSV
├── monitoring/
│   └── jetson_monitor.py       # Logs CPU/GPU/RAM/temp to CSV during long runs
├── scripts/
│   ├── setup_jetson.sh         # One-shot Jetson setup (after JetPack flash)
│   ├── deploy_to_jetson.sh     # rsync project to Jetson over LAN
│   ├── run_dashboard.sh        # Launch the dashboard via the project venv
│   └── requirements.txt        # pip deps (see header for Jetson-only steps)
├── models/                     # Weights & engines (best.engine, best_v<N>.engine, …)
├── dataset/user_recorded/      # Labels produced by the dashboard's labeling tab
├── fish_logs/                  # Runtime output: JSON stats + screenshots
├── .env.example                # Template for the gitignored .env (JETSON_HOST / USER / etc.)
└── LICENSE                     # GNU AGPL-3.0 (same family as YOLOv8)
```

## Architecture

```
┌──────────────┐     USB      ┌──────────────────┐     TensorRT      ┌──────────────┐
│ Logitech C920│ ──────────▶ │  Jetson Orin Nano │ ───────────────▶  │  YOLOv8s     │
│  (1280×720)  │             │   (JetPack 6.1)   │                   │  Detection   │
└──────────────┘              └────────┬──────────┘                   └──────┬───────┘
                                       │                                     │
                                       │              ┌──────────────────────┘
                                       │              ▼
                                       │       ┌──────────────┐
                                       │       │  ByteTrack   │
                                       │       │ (supervision)│
                                       │       └──────┬───────┘
                                       ▼              ▼
                              ┌────────────────────────────────────┐
                              │   Dashboard (MJPEG @ :8080)        │
                              │   • Live trails, IDs, counts       │
                              │   • Snapshots / filmstrip          │
                              │   • Labeling tab → user_recorded/  │
                              │   • Train Model → train_jetson.py  │
                              │   • Cloudflare public URL          │
                              └────────────────────────────────────┘
```

## Hardware

- NVIDIA Jetson Orin Nano Developer Kit (8 GB recommended)
- Logitech C920 HD Pro webcam
- 64 GB+ microSD card
- USB-A ↔ USB-A data cable for the C920
- Power supply (included with the Jetson kit)
- Monitor + keyboard for first boot, or SSH access

**Camera mounting tips**

1. Face the C920 at the front glass, 30–60 cm out.
2. Tilt 5–10° downward to reduce reflections; matte background helps too.
3. Diffuse, even lighting beats bright overhead lights (which cause glare).

## Setup

### 1. Flash JetPack

Flash JetPack 6.1 (or 5.1.3+) to the Jetson via NVIDIA SDK Manager or
balenaEtcher. Image and instructions:
https://developer.nvidia.com/jetson-orin-nano-developer-kit

### 2. Deploy code to the Jetson

From your host machine:

```bash
cp .env.example .env   # then fill in JETSON_HOST / JETSON_USER / JETSON_PASSWORD / JETSON_PATH
bash scripts/deploy_to_jetson.sh
```

### 3. One-shot Jetson setup

SSH in and run the setup script — it installs system & Python deps, sets
max-performance mode, downloads YOLOv8s, and exports the baseline TensorRT
engine to `models/`:

```bash
ssh jetson@<jetson-ip>
cd ~/projects/aquascope
bash scripts/setup_jetson.sh
```

Note the requirements file ([scripts/requirements.txt](scripts/requirements.txt))
has a header listing the Jetson-only manual steps (PyTorch wheel, cuSPARSELt,
torchvision-from-source, apt OpenCV) — `setup_jetson.sh` handles them for you.

### 4. Run the dashboard

From the project root on the Jetson:

```bash
scripts/run_dashboard.sh                       # local display + browser stream
scripts/run_dashboard.sh --no-display          # headless + browser stream (recommended over SSH)
scripts/run_dashboard.sh --no-display --public # + public URL via Cloudflare tunnel
scripts/run_dashboard.sh --sahi                # sliced inference (better small-fish recall)
scripts/run_dashboard.sh --conf 0.3 --resolution 1080p
scripts/run_dashboard.sh --exposure -6         # manual V4L2 exposure for a dim tank
```

Anything you pass is forwarded to `app/fish_tracker.py`. Open the dashboard
at `http://<jetson-ip>:8080`.

**Local-display keys:** `q` quit, `r` reset trails.

## Dashboard

The dashboard is a single page served from `app/stream.py` with these tabs:

- **Live Feed** — MJPEG stream, fish/shrimp counts, current FPS, and toggles
  for trails, the CLAHE enhancer, and the "hat" overlay.
- **Analytics** — per-track activity stats from the JSON log files.
- **Snapshots** — screenshots taken from the dashboard, stored under
  [fish_logs/screenshots/](fish_logs/screenshots/).
- **Label Fish** — opens straight onto the most recent live frame with all
  predicted boxes pre-drawn. Each predicted box has a red ✕ to remove false
  positives; click-drag to add missing fish; *Save & next* writes a YOLO
  label under [dataset/user_recorded/](dataset/user_recorded/) and pulls
  the next frame.
- **Settings** — model dropdown (lists every `models/best*.engine`
  on disk), confidence threshold, resolution, exposure.

### Train a new model from the dashboard

When `dataset/user_recorded/` has enough labels (the *Train Model* button
shows the current count + minimum), clicking it spawns
[training/train_jetson.py](training/train_jetson.py) as a subprocess. The
modal polls `/train/status` for live epoch + ETA, and on success the new
engine is written to `models/best_v<N>.engine` and appears in the model
dropdown.

The endpoints are documented in the docstring at the top of
[app/stream.py](app/stream.py).

## Training (manual / off-dashboard)

Both trainers consume the labels produced by the dashboard's Labeling tab
(`dataset/user_recorded/`) and produce class-locked weights for **fish**
and **shrimp**.

### On the Jetson

```bash
# What the dashboard's "Train Model" button runs:
python3 training/train_jetson.py --epochs 30 --batch 2

# Custom run name + status file (mirrors what the dashboard uses):
python3 training/train_jetson.py --epochs 50 --batch 2 \
    --status-file /tmp/aq_train.json
```

90/10 train/val split over `dataset/user_recorded/`, AMP FP16, rectangular
batching, mosaic/mixup off (Jetson RAM headroom). On success it copies the
best weights to `models/best_v<N>.pt`, exports them to
`models/best_v<N>.engine` (TensorRT FP16), and wipes the user-recorded set
so the next labeling round starts clean.

### On a Mac (Apple Silicon / MPS)

```bash
python3 training/mac_train.py --epochs 50 --batch 8
```

Uses MPS, full augmentation, and exports to CoreML (`.mlpackage`). If you
want the model on the Jetson, copy the `.pt` over and re-export to
TensorRT there (the engine format is hardware-specific).

### Resource monitoring

```bash
# Separate terminal — logs CPU/GPU/RAM/temp every 2 s
python3 monitoring/jetson_monitor.py --interval 2 --output training_log.csv

# Plot the result
python3 training/analyse_training.py --csv training_log.csv
```

## Output

`fish_logs/` accumulates JSON stats every 60 seconds:

```json
{
  "timestamp": "20260508_143022",
  "total_frames": 1800,
  "unique_fish": 5,
  "fish": {
    "1": {
      "first_seen": "2026-05-08T14:29:22",
      "last_seen":  "2026-05-08T14:30:22",
      "total_distance_px": 4521.7,
      "frame_count": 1650
    }
  }
}
```

Snapshots taken from the dashboard live in `fish_logs/screenshots/`.

## Performance Tuning

| Setting                       | Command / Action                              | Effect                |
|-------------------------------|-----------------------------------------------|-----------------------|
| Max Jetson performance        | `sudo nvpmodel -m 0 && sudo jetson_clocks`    | +20–40% FPS           |
| Use TensorRT FP16             | `--model models/best.engine`                  | ~2× faster than `.pt` |
| Smaller inference size        | `--imgsz 416`                                 | +30% FPS              |
| Lower capture resolution      | `--resolution 720p` (or `480p`)               | Less CPU overhead     |
| Higher confidence threshold   | `--conf 0.5`                                  | Fewer false positives |
| Cheaper JPEG stream           | `--stream-quality 60 --stream-fps 15`         | Lower bandwidth       |

## Troubleshooting

**Low FPS (<15)**
- Ensure `nvpmodel -m 0` and `jetson_clocks` are active.
- Use the TensorRT engine (`--model models/best.engine`), not a `.pt`.
- Close other GPU-intensive apps.

**Camera not detected**
- `v4l2-ctl --list-devices` to check.
- Try `--camera 1` if `/dev/video0` is taken.
- Confirm the USB cable carries data, not just power.

**Detections look bad**
- Lower the confidence (`--conf 0.2`) or try `--sahi` for small fish.
- Fine-tune on your tank — capture a few hundred labels via the Labeling
  tab and click *Train Model*.
- Check lighting: too dark or strong glare both hurt detection.

**ID switches**
- Increase `track_buffer` in [app/bytetrack.yaml](app/bytetrack.yaml).
- A fine-tuned model produces much more stable detections.

## License

AquaScope is released under the GNU Affero General Public License v3.0
([LICENSE](LICENSE)). The AGPL applies to network use: if you run a
modified version of this code as a service, you must make the modified
source available to its users. YOLOv8 (the upstream detector) is also
AGPL-3.0 by Ultralytics, so the licenses match.
