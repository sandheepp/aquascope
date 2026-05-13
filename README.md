# AquaScope

> Off-the-shelf YOLO thought my guppy was a puffin. So I built a browser tab
> where I tick *yes/no* on its guesses for ten minutes, hit a button, and ten
> minutes later it knows my fish.

AquaScope is a real-time fish & shrimp detector you can teach from a browser
tab. It runs the whole loop on one device — capture, detect, track, label,
fine-tune — with no cloud, no upload, no API key.

<!--
  Hero artifact placeholder — replace with a 6–10 MB looping GIF of:
  live tank → boxes → Labeling tab → tick crops → Train Model → better boxes.
  Drop the file at docs/hero.gif and reference it with:
    ![AquaScope demo](docs/hero.gif)
-->

[![CI](https://github.com/sandheepp/aquascope/actions/workflows/ci.yml/badge.svg)](https://github.com/sandheepp/aquascope/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hardware: Jetson + laptop](https://img.shields.io/badge/runs%20on-Jetson%20%7C%20laptop%20%7C%20Mac-success)](docs/01-how-it-works.md)

## What it does

- **Live detection + tracking** of fish and shrimp at 20–30 FPS on a Jetson
  Orin Nano (or whatever your laptop manages).
- **In-browser labeling on the latest frame**: predicted boxes appear
  pre-filled — tap the red ✕ to drop false positives, click-drag to add
  fish the model missed, hit **Save & next** for a fresh frame.
- **One-click fine-tuning**: when you've saved enough labels (threshold
  configurable in Settings), the **Train model** button retrains YOLOv8
  on YOUR fish, in YOUR tank, under YOUR lighting.
- **Live training charts in the page**: two-stage progress bar (epochs →
  TensorRT export), loss curves (total + box / cls / dfl), GPU
  utilization, GPU memory, RAM — all updating every 2 s.
- **Auto-export + hot-swap**: on a Jetson the new model is compiled to
  TensorRT FP16 (`models/best_v<N>.engine`) and appears in the Model
  dropdown without a restart.
- **Analytics history**: every successful training run is saved as
  `models/best_v<N>.status.json`, and the **Analytics** tab lets you
  re-open any past version's loss curves + resource usage from a dropdown.
- **Multi-user aware**: when someone starts a training run, every
  connected dashboard auto-opens the training modal; when it ends, the
  modal dismisses itself and inference resumes for everyone.
- **Runs offline**: nothing leaves your network. Optional Cloudflare tunnel
  if you want to share the dashboard publicly.

## Quick start

### Without a Jetson — laptop, Mac, or any Linux machine

```bash
git clone https://github.com/sandheepp/aquascope.git
cd aquascope
bash scripts/run_local.sh                                 # USB webcam
# or — no webcam? Use any video file:
bash scripts/run_local.sh --video samples/aquarium.mp4
```

Then open <http://localhost:8080>. The script bootstraps a venv, installs
the laptop deps, and downloads `yolov8n.pt` on first run.

### With Docker (most portable)

```bash
docker build -t aquascope .
docker run --rm -p 8080:8080 -v "$PWD/samples:/app/samples" aquascope \
  --no-display --video samples/aquarium.mp4
```

### On a Jetson Orin Nano

The full path with TensorRT engines + max-perf mode:

```bash
ssh jetson@<jetson-ip>
cd ~/projects/aquascope
bash scripts/setup_jetson.sh        # one-time: deps + JetPack tweaks
bash scripts/run_dashboard.sh       # 25–30 FPS at 720p
```

See [Setup → Jetson](#setup-on-a-jetson) below for the full recipe.

## Take the tour

Open the dashboard at `http://<host>:8080`. Five tabs in the left sidebar:

1. **📹 Live Feed** — MJPEG video with track IDs + trails + FPS / Total
   IDs / Frame counter on the right.
2. **📊 Analytics** — pick any past trained model (`best_v<N>`) from the
   dropdown to see its full training summary: final loss, dataset size,
   elapsed time, device, plus the same loss/GPU/RAM charts the live
   training modal showed while it was running.
3. **🖼️ Snapshots** — grid of every snapshot you've taken with
   per-card open / download / delete actions.
4. **🏷️ Label Fish** — lands you directly on the predicted-frame screen.
   Predicted fish are pre-drawn with red ✕ buttons to remove false
   positives; click-drag for missed ones; **Save & next** writes a YOLO
   label and pulls a fresh frame. The Train-model card on the right
   tells you how many more labels you need to unlock training.
5. **⚙️ Settings** — two columns: **Inference** (detection model,
   resolution, confidence slider, trails, underwater enhance) and
   **Training** (labeling model, epoch count, editable
   minimum-labels-to-train threshold). Settings stays in sync with the
   inline controls on other tabs.

When you click **🧠 Train model** on Label Fish: a confirmation dialog
shows label count + ETA, then the training modal opens with a two-stage
progress bar (Stage 1 epochs, Stage 2 TensorRT engine export), tabs for
**📊 Charts** (loss + GPU util + GPU memory + RAM) and **📜 Logs**
(subprocess output). The modal is broadcast — every connected user sees
it open + auto-dismiss together.

For the screenshot-by-screenshot walkthrough see
[docs/02-the-label-train-loop.md](docs/02-the-label-train-loop.md).

## How it works

```
┌──────────────┐                   ┌──────────────┐                ┌──────────────┐
│ Webcam OR    │ ─── frame ──────▶ │  YOLOv8      │ ── detections ▶│  ByteTrack   │
│ video file   │                   │ (.pt/.engine)│                │   (sv lib)   │
└──────────────┘                   └──────────────┘                └──────┬───────┘
                                                                          │ tracked boxes
                                                                          ▼
                              ┌────────────────────────────────────────────────────┐
                              │ Dashboard (single-file HTTP server, app/stream.py) │
                              │  /stream            ── Live MJPEG video            │
                              │  /label/predictions ── boxes for the labeling tab  │
                              │  /label/manual      ── save user-drawn YOLO labels │
                              │  /train/start       ── spawn trainer subprocess    │
                              │  /train/status      ── live stages + loss + usage  │
                              │  /train/history     ── per-version saved runs      │
                              │  /train/min-labels  ── editable label threshold    │
                              │  /models, /model    ── list + select active engine │
                              └────────────────────────────────────────────────────┘
```

For the conceptual deep-dive (what YOLO actually does, why ByteTrack matters,
how fine-tuning works) see [docs/01-how-it-works.md](docs/01-how-it-works.md).
For terminology see [docs/glossary.md](docs/glossary.md).

## Hardware

AquaScope runs on three target classes of machine:

| Target | Notes |
|---|---|
| **Jetson Orin Nano** + Logitech C920 | Primary target. 25–30 FPS at 720p with TensorRT FP16. |
| **macOS / Apple Silicon** | Auto-uses MPS. Fine for labeling + small training runs. |
| **Linux/Windows laptop**, with or without an NVIDIA GPU | CPU mode just works. NVIDIA GPU gets CUDA + AMP. |

Camera mounting tips for an actual aquarium:

1. Face the camera at the front glass, 30–60 cm out.
2. Tilt 5–10° downward to reduce reflections; matte background helps.
3. Diffuse, even lighting beats bright overhead lights (which cause glare).

## Setup on a Jetson

### 1. Flash JetPack

Flash JetPack 6.1 (or 5.1.3+) to the Jetson via NVIDIA SDK Manager or
balenaEtcher: <https://developer.nvidia.com/jetson-orin-nano-developer-kit>

### 2. Deploy code to the Jetson

From your dev machine:

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

The Jetson-only manual steps (PyTorch wheel, cuSPARSELt, torchvision-from-
source, apt OpenCV) are documented at the top of
[scripts/requirements.txt](scripts/requirements.txt) — `setup_jetson.sh`
handles them for you.

### 4. Run the dashboard

```bash
bash scripts/run_dashboard.sh                       # local display + browser stream
bash scripts/run_dashboard.sh --no-display          # headless + browser stream
bash scripts/run_dashboard.sh --no-display --public # + public URL via Cloudflare tunnel
bash scripts/run_dashboard.sh --sahi                # sliced inference (better small-fish recall)
bash scripts/run_dashboard.sh --conf 0.3 --resolution 1080p
bash scripts/run_dashboard.sh --exposure -6         # manual V4L2 exposure for a dim tank
```

Open the dashboard at `http://<jetson-ip>:8080`.

**Local-display keys:** `q` quit, `s` snapshot, `r` reset trails.

## Training (manual, off the dashboard)

The dashboard's **Train Model** button just spawns
[training/train_jetson.py](training/train_jetson.py) for you. To train from
the CLI directly:

```bash
# What the dashboard runs (works on Jetson, laptop GPU, MPS, or CPU):
python3 training/train_jetson.py --epochs 30 --batch 2

# Smaller / faster iteration:
python3 training/train_jetson.py --epochs 10 --batch 2 --imgsz 384
```

90/10 train/val split over `dataset/user_recorded/`. On CUDA the best weights
are exported to `models/best_v<N>.engine`; on MPS/CPU you get
`models/best_v<N>.pt` and the dashboard picks it up via the same dropdown.

The Mac trainer ([training/mac_train.py](training/mac_train.py)) is still
here as a more aggressive M-series option (full augmentation, CoreML export)
if you want it.

### Resource monitoring (Jetson)

```bash
# Separate terminal — logs CPU/GPU/RAM/temp every 2 s
python3 monitoring/jetson_monitor.py --interval 2 --output training_log.csv

# Plot the result
python3 training/analyse_training.py --csv training_log.csv
```

## Performance tuning

| Setting                       | Command / Action                              | Effect                |
|-------------------------------|-----------------------------------------------|-----------------------|
| Max Jetson performance        | `sudo nvpmodel -m 0 && sudo jetson_clocks`    | +20–40% FPS           |
| Use TensorRT FP16             | `--model models/best.engine`                  | ~2× faster than `.pt` |
| Smaller inference size        | `--imgsz 416`                                 | +30% FPS              |
| Lower capture resolution      | `--resolution 720p` (or `480p`)               | Less CPU overhead     |
| Higher confidence threshold   | `--conf 0.5`                                  | Fewer false positives |
| Cheaper JPEG stream           | `--stream-quality 60 --stream-fps 15`         | Lower bandwidth       |

## Troubleshooting

**Low FPS (<15) on Jetson**
- Ensure `nvpmodel -m 0` and `jetson_clocks` are active.
- Use the TensorRT engine (`--model models/best.engine`), not a `.pt`.
- Close other GPU-intensive apps.

**Camera not detected**
- Linux: `v4l2-ctl --list-devices`. Try `--camera 1` if `/dev/video0` is taken.
- macOS: System Settings → Privacy → Camera; grant Terminal access.
- Confirm the USB cable carries data, not just power.

**Detections look bad**
- Lower the confidence (`--conf 0.2`) or try `--sahi` for small fish.
- Fine-tune on your tank — capture a few hundred labels via the Labeling
  tab and click *Train Model*. See [docs/02-the-label-train-loop.md](docs/02-the-label-train-loop.md).
- Check lighting: too dark or strong glare both hurt detection.

**ID switches**
- Increase `track_buffer` in [app/bytetrack.yaml](app/bytetrack.yaml).
- A fine-tuned model produces much more stable detections.

**`run_local.sh` fails on first run**
- It installs deps from PyPI on first use; check your network and pip version.
- On Linux you may need `sudo apt install python3-venv ffmpeg libgl1`.

## Project layout

<details>
<summary>File-by-file (click to expand)</summary>

```
aquascope/
├── app/                        # Core tracker application
│   ├── fish_tracker.py         # CLI entry point
│   ├── tracker.py              # Capture + inference + ByteTrack loop
│   ├── stream.py               # MJPEG server + dashboard UI + label/train endpoints
│   ├── model.py                # Model loader (.pt / TensorRT .engine), device auto-pick
│   ├── camera.py               # Webcam (V4L2/AVFoundation/MSMF) or video-file capture
│   ├── enhancer.py             # CLAHE frame enhancer (toggleable from dashboard)
│   ├── config.py               # Default config + trail colour palette
│   ├── jetson_compat.py        # JetPack torchvision shim (no-op on regular machines)
│   └── bytetrack.yaml          # ByteTrack tuning reference
├── training/
│   ├── train_jetson.py         # Cross-platform trainer (CUDA → engine, MPS/CPU → .pt)
│   ├── mac_train.py            # Mac-specific trainer (Apple Silicon / MPS, CoreML export)
│   └── analyse_training.py     # Plot CPU/GPU/RAM curves from a monitor CSV
├── monitoring/
│   └── jetson_monitor.py       # Logs CPU/GPU/RAM/temp to CSV during long runs
├── scripts/
│   ├── setup_jetson.sh         # One-shot Jetson setup (after JetPack flash)
│   ├── deploy_to_jetson.sh     # rsync project to Jetson over LAN
│   ├── run_dashboard.sh        # Launch the dashboard via the Jetson venv
│   ├── run_local.sh            # Launch the dashboard from a fresh laptop venv
│   ├── requirements.txt        # Jetson pip deps (header lists manual JetPack steps)
│   └── requirements_local.txt  # Laptop / non-Jetson pip deps
├── docs/                       # Concept explainers, walkthroughs, glossary
├── samples/                    # Drop-zone for demo videos (--video flag)
├── models/                     # Weights & engines (best.engine, best_v<N>.engine, …)
├── dataset/user_recorded/      # Labels produced by the dashboard's labeling tab
├── fish_logs/                  # Runtime output: JSON stats + screenshots
├── Dockerfile                  # Laptop-mode Docker image
├── .env.example                # Template for the gitignored .env
└── LICENSE                     # GNU AGPL-3.0 (same family as YOLOv8)
```

</details>

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

Every successful training run is also persisted to
`models/best_v<N>.status.json` — the full final status (per-epoch loss
breakdown, GPU/RAM samples, validation metrics, device + dataset +
hyperparameters). The Analytics tab reads these directly via
`/train/history` so you can re-open any past run's charts without
keeping the training process alive.

## License

The AquaScope code in this repository is **AGPL-3.0**, mirroring its main
dependency (Ultralytics YOLOv8). For commercial / closed-source use see the
note at the bottom of [LICENSE](LICENSE).

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Especially valuable:
platform fixes for macOS/Windows, sample videos, dashboard polish.
