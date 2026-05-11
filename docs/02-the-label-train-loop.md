# The label → train loop

This is the part of AquaScope that's worth your attention. It's how a generic
"can spot fish, sometimes" model becomes a specific "knows MY fish, reliably"
model — without you ever leaving the browser.

## The four stages

```
   [1] Run the dashboard           [2] Label some detections
        │                                  │
        ▼                                  ▼
   ┌─────────┐  detections show     ┌─────────────┐  YOLO label
   │ tracker │  up as candidate    →│ Labeling    │  files land in
   │  loop   │  thumbnails          │ tab         │  dataset/user_recorded/
   └─────────┘                      └──────┬──────┘
                                           │
                                           ▼
                                    [3] Click Train Model
                                           │
                                           ▼
                                    ┌────────────────┐
                                    │ train_jetson   │  best_v<N>.engine
                                    │  subprocess    │  (or .pt on laptop)
                                    └──────┬─────────┘
                                           │
                                           ▼
                                    [4] Auto-reload; better detections
```

## Stage 1: Run the dashboard

```bash
# Jetson
bash scripts/run_dashboard.sh --no-display --public

# Laptop
bash scripts/run_local.sh
# or with a video file:
bash scripts/run_local.sh --video samples/aquarium.mp4
```

Open the browser at `http://<your-host>:8080`. You should see:

- The live feed with bounding boxes and IDs ("Fish #1 (87%)").
- An FPS counter top-left.
- Tabs for **Live**, **Analytics**, **Snapshots**, **Labeling**, and **Settings**.

If the boxes look bad — too few, wrong class, jittery — that's normal at this
point. You haven't fine-tuned yet.

## Stage 2: Label some detections

Click **Labeling**. The tab does two things at once:

1. **Drops the inference confidence threshold to 5%** while you're labeling,
   so the model surfaces weak detections instead of filtering them. Even
   a noisy guess is useful — you decide whether to keep it.
2. **Shows each unique detection as a candidate crop.** For each candidate
   you have three options:

| Action | Effect |
|---|---|
| **✓** (keep) | The crop's bounding box is saved to `dataset/user_recorded/labels/<frame>.txt` in YOLO format, and the source frame is saved as the matching `.jpg`. |
| **✗** (reject) | The candidate is discarded. |
| **+ Manual box** | Open the manual labeller — draw one or more boxes on the current frame for things the model didn't detect. |

The bottom-right counter shows your label count and the minimum needed to
train (default: 25). Aim higher than the minimum if you want a noticeable
quality jump — 100–500 labels is the sweet spot for two classes.

### What does a label file actually look like?

Each `dataset/user_recorded/labels/<name>.txt` is a few lines like:

```
0 0.5234 0.4812 0.1280 0.0825
1 0.7142 0.3914 0.0950 0.0612
```

That's YOLO format: one line per box,

```
<class_idx> <center_x> <center_y> <width> <height>
```

All four box numbers are normalized to [0, 1] relative to image width/height.
`class_idx` is `0` for fish, `1` for shrimp (defined by `_USER_CLASSES` in
`train_jetson.py`). The matching image is at `dataset/user_recorded/images/<name>.jpg`.

That's it — that's the entire ground-truth format. It's simple on purpose:
YOLO can find the labels by replacing `/images/` with `/labels/` in the
image path.

## Stage 3: Click Train Model

Once you have enough labels, the **Train Model** button at the bottom of the
Labeling tab activates. Set the epochs (5/10/30/50) and click it.

A modal pops up showing live training progress:

- **State**: `starting` → `training` → `exporting` → `done`.
- **Epoch counter**: `Epoch 4 / 30`.
- **ETA**: how long until done, based on rolling average epoch time.

While training runs:

- **Inference is paused** — the tracker drops its model and releases the
  camera, freeing GPU memory and RAM for the trainer. The live feed shows
  a "Training in progress" placeholder.
- **The trainer subprocess** runs `training/train_jetson.py` with these
  defaults:
  - `--batch 2` (Jetson Orin Nano headroom; bumps fine on a real GPU)
  - `--imgsz 512` (training res; lower → less RAM)
  - `--export-imgsz 640` (engine input shape; matches inference imgsz)
  - `mosaic=0.0`, `mixup=0.0` (these augmentations 4× per-sample RAM)
  - `rect=True` (rectangular batching cuts padding waste)
  - 90/10 train/val split
- **On success** (CUDA path):
  1. The best epoch's weights land in `runs/detect/fish_train_jetson/weights/best.pt`.
  2. They're copied to `models/best_v<N>.pt`.
  3. They're exported to `models/best_v<N>.engine` (TensorRT FP16).
  4. `dataset/user_recorded/` is wiped so the next labeling round starts clean.

On laptop (no CUDA), step 3 is skipped — the `.pt` is the artifact.

## Stage 4: Auto-reload, see the difference

When you click **Close** on the success modal:

1. The tracker reloads the model. It picks the highest version available
   (`best_v<N>.engine` if present, else `best_v<N>.pt`).
2. The camera reopens.
3. Inference resumes — same as before, but now with your fine-tuned weights.

You should see:

- **Tighter boxes** that hug the fish more accurately.
- **Fewer false positives** (the plastic plant stops getting flagged as a
  shark).
- **Stable IDs** — better detections give the tracker more to work with, so
  fish keep their IDs longer.

## How many labels do I actually need?

Rough guidance for the bundled fish/shrimp setup:

| Labels | Expected result |
|---|---|
| **0–25** | Train button locked. The minimum guard exists for a reason — fewer than this and you'll overfit to a handful of frames. |
| **25–100** | First useful fine-tune. Boxes get noticeably tighter, false-positive rate drops a lot. |
| **100–500** | Strong improvement. Most species-level confusion (puffin, stingray) goes away. |
| **500+** | Diminishing returns. At this point invest in *variety* (different lighting, fish poses, partial occlusions) rather than raw count. |

Diversity matters more than count. 200 labels covering 5 lighting conditions
beat 1000 labels of the same well-lit half hour.

## When to retrain

There's no hard rule. Good signals:

- IDs flicker more than they used to (lighting drift, new fish added).
- A new species in the tank that the model doesn't know yet.
- You changed the camera angle, distance, or backdrop.
- You feel like it. Each retrain takes 5–30 min on a Jetson, less on a real
  GPU. It's cheap to iterate.

## What can go wrong

- **"No CUDA GPU detected"** on the Jetson → JetPack didn't install the
  CUDA runtime correctly. Re-flash or check `nvidia-smi`.
- **Out-of-memory during training** on the Jetson → drop `--batch` to 1 or
  `--imgsz` to 384.
- **Training finishes but boxes look identical** → either too few labels
  (<25 useful ones) or all your labels were on the same frame. Vary your
  capture conditions and try again.
- **Engine export fails** but `best.pt` exists → use the `.pt` directly via
  the Settings dropdown. The dashboard supports both.

## Next steps

- Read [01-how-it-works.md](01-how-it-works.md) for the conceptual
  background on YOLO + ByteTrack.
- Look up jargon in [glossary.md](glossary.md).
- See the trainer code: [training/train_jetson.py](../training/train_jetson.py).
