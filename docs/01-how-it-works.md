# How AquaScope works

A 10-minute read for someone who has never built a CV system before. If terms
like "YOLO", "ByteTrack", or "fine-tuning" are familiar — skim. If not, this
is for you.

## The high-level flow

```
camera (or video file)
  │
  ▼
┌──────────────┐    every frame
│  Detection   │   ───────────────►  list of bounding boxes + class
│   (YOLO)     │
└──────┬───────┘
       │
       ▼
┌──────────────┐    every frame
│   Tracking   │   ───────────────►  same boxes, but each one keeps an ID
│ (ByteTrack)  │                     across frames so "fish #3" stays #3
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Dashboard (browser)                                         │
│  • Live MJPEG video with boxes drawn on it                   │
│  • Counts, FPS, trail visualisations                         │
│  • A "Labeling" tab that turns each detection into a         │
│    candidate crop you can keep or discard                    │
│  • A "Train Model" button that retrains on what you kept     │
└──────────────────────────────────────────────────────────────┘
```

The whole pipeline runs on one device (a Jetson, a laptop, whatever) — there
is no cloud step, no upload, no API key. That matters because (a) it's cheap,
(b) it works without an internet connection once installed, and (c) your tank
footage never leaves your network.

## Detection: what YOLO is actually doing

**YOLO** ("You Only Look Once") is a family of neural networks that takes a
single image and produces bounding boxes around objects it recognizes. Each
box comes with:

- **xyxy coordinates** — top-left and bottom-right corners in pixels.
- **A class label** — "fish", "shrimp", etc. (depends on what the model was
  trained on).
- **A confidence score** — how sure the model is, between 0 and 1.

The "v8s" in `yolov8s.pt` is just a size. There are five sizes (n=nano,
s=small, m=medium, l=large, x=xlarge). Bigger is more accurate, slower, and
hungrier for memory. AquaScope ships with `yolov8s` because it's the sweet
spot for a Jetson Orin Nano: about 25 FPS at 720p with 11 M parameters. On a
laptop CPU you'll want `yolov8n` instead.

### Why "small" matters: TensorRT and the .engine file

A `.pt` file is a PyTorch checkpoint — portable but not particularly fast.
A `.engine` file is the same model **compiled** by NVIDIA's TensorRT runtime
for one specific GPU. The compile step bakes in graph optimizations and FP16
math, which on the Jetson roughly **doubles** inference speed.

Two things to know:

1. **Engines are hardware-specific.** An engine built on an Orin Nano won't
   run on a laptop GPU and vice versa. That's why AquaScope's training code
   re-runs the export step at the end of each fine-tune.
2. **Engines have a fixed input size.** Whatever `imgsz` you exported with
   is the only image size the engine accepts. Train and export at 640 → feed
   the engine 640 forever. AquaScope locks this to 640 by default.

If you don't have a GPU, AquaScope skips the engine and just uses the `.pt`
file directly. Slower, but works on anything.

## Tracking: why detection alone isn't enough

If you only had detection, every frame would give you a fresh, anonymous list
of boxes. There would be no concept of "the same fish from frame 1 and frame
2" — you couldn't draw a trail, count unique fish, or measure how long a fish
has been visible.

**Tracking** assigns a stable ID to each detected object across frames. When
the same fish moves a bit between frames, the tracker says "that's still
fish #3" instead of treating it as a new detection.

AquaScope uses **ByteTrack** (via the [supervision](https://supervision.roboflow.com/)
library). The intuition:

1. Predict where each existing track *should* be in the next frame, using a
   simple motion model (Kalman filter).
2. Match new detections to existing tracks by overlap (IoU = Intersection-
   over-Union of the predicted box and the new box).
3. Keep tracks alive for a few frames after the fish disappears, in case it
   re-emerges from behind a plant.

The two parameters that matter most:

| Parameter | Effect |
|---|---|
| `track_buffer` | How many frames a track survives without a matching detection. Higher = fewer ID switches when fish briefly hide, but stale tracks linger. |
| `match_threshold` | Minimum IoU to call a new detection the same fish. Higher = stricter matching, more new IDs spawned. |

Tweaks live in [app/bytetrack.yaml](../app/bytetrack.yaml).

## Fine-tuning: the most important part

Off-the-shelf YOLO is trained on a generic dataset (COCO + a Roboflow
"aquarium combined" dataset for the bundled weights). It will detect *some*
fish in *most* tanks, but it'll also:

- Confuse your guppy with "puffin" or "stingray".
- Miss small or partially-hidden shrimp completely.
- Give you boxes that are too loose, leading to wobbly trails.

**Fine-tuning** is the process of taking a pretrained model and continuing
the training on YOUR data — labels of YOUR fish, in YOUR tank, under YOUR
lighting. The model keeps everything it learned about general object features
(edges, textures, motion) but specializes the final classifier to your two
classes.

The key win: you don't need 10,000 labeled images. With a pretrained backbone,
**a few hundred labels of your tank** typically buys you most of the accuracy
you'd get from training from scratch on millions.

That's the loop AquaScope automates:

1. Run the dashboard.
2. Open the Labeling tab — it shows you each detection as a thumbnail. Click
   ✓ to keep, ✗ to reject. You can also draw new boxes manually for things
   the model missed.
3. After ~50–500 labels, click **Train Model**. The dashboard:
   - Pauses inference to free GPU memory.
   - Splits your labels 90/10 train/val.
   - Fine-tunes `yolov8s` for N epochs (default 30).
   - On a Jetson, exports the result to `best_v<N>.engine` (TensorRT FP16).
   - On a laptop, just saves `best_v<N>.pt`.
   - Wipes the labeled set so the next round starts clean.
4. The new model auto-loads. Detections are now better. Repeat from step 2
   if you want them even better.

This is the killer idea of the project. Read [02-the-label-train-loop.md](02-the-label-train-loop.md)
next for the screenshot-by-screenshot walkthrough.

## What the code actually does, file by file

| File | Job |
|---|---|
| [app/fish_tracker.py](../app/fish_tracker.py) | CLI argument parsing, builds the config dict, kicks off the tracker. |
| [app/tracker.py](../app/tracker.py) | The main loop: read frame → infer → track → draw → push to stream. Also handles training-pause and post-train model reload. |
| [app/model.py](../app/model.py) | Loads the YOLO model, picks the device (CUDA/MPS/CPU), runs inference (with optional SAHI tiling for small fish). |
| [app/camera.py](../app/camera.py) | Opens a webcam (V4L2 on Linux, AVFoundation on Mac, MSMF on Windows) or a video file. |
| [app/stream.py](../app/stream.py) | The dashboard — a single-file HTTP server that serves the MJPEG video, the HTML/JS UI, and all the labeling/training endpoints. |
| [app/enhancer.py](../app/enhancer.py) | Optional CLAHE contrast enhancement, toggleable from the dashboard for dim tanks. |
| [training/train_jetson.py](../training/train_jetson.py) | The trainer the dashboard's "Train Model" button spawns as a subprocess. Cross-platform despite the name — exports TensorRT on CUDA, just saves .pt on MPS/CPU. |
| [app/jetson_compat.py](../app/jetson_compat.py) | Imported before torch on Jetson to patch a quirk in the JetPack torchvision build. No-op on regular machines. |

## What's NOT in this project (and why)

- **Cloud inference.** Everything is local. The `--public` flag tunnels the
  dashboard via Cloudflare for *viewing* but the model still runs on your
  Jetson.
- **Multi-camera.** One source at a time. Multi-camera would mean coordinating
  tracker state across cameras (re-ID), which is a much harder problem.
- **Re-ID across sessions.** Track IDs reset when you restart the tracker.
  Nothing tries to remember "this is the same fish from yesterday."
- **Species-level classification beyond fish/shrimp.** The dashboard's
  trainer is locked to two classes. You can re-purpose it for more classes
  by editing `_USER_CLASSES` in `train_jetson.py`.

## Next steps

- See the dashboard end-to-end: [02-the-label-train-loop.md](02-the-label-train-loop.md).
- Look up an unfamiliar term: [glossary.md](glossary.md).
- Read the code: start at [app/fish_tracker.py](../app/fish_tracker.py).
