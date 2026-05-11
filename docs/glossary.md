# Glossary

Plain-English one-liners for terms used across the AquaScope code, README,
and docs. If you're new to computer vision, read this top-to-bottom; otherwise
ctrl-F.

## Models & inference

**YOLO** — "You Only Look Once". A family of object-detection neural networks
that produce bounding boxes for an entire image in one forward pass. AquaScope
uses YOLOv8 from [Ultralytics](https://github.com/ultralytics/ultralytics).

**Bounding box (bbox)** — The rectangle around a detected object. Stored as
either `xyxy` (top-left + bottom-right pixel coords) or `xywh` (center +
width/height).

**Class** — The label assigned to a detection ("fish", "shrimp", "puffin",
…). The number of classes a model knows is set at training time and baked
into the weights.

**Confidence** — The model's self-reported probability that a detection is
real, between 0 and 1. AquaScope's dashboard exposes a slider to filter on
this.

**IoU (Intersection-over-Union)** — A measure of how much two boxes overlap,
from 0 (no overlap) to 1 (identical). Used everywhere in detection: NMS,
tracker matching, mAP scoring.

**NMS (Non-Maximum Suppression)** — When a detector fires multiple boxes for
the same object, NMS keeps the highest-confidence one and discards
overlapping lower-confidence ones. The IoU threshold determines "overlapping".

**mAP (mean Average Precision)** — The standard accuracy metric for detection,
between 0 and 1. Higher is better. `mAP@0.5` means averaging over detections
where IoU with ground truth is at least 0.5.

**Inference** — Running the model on a new image to produce predictions
(as opposed to *training* it on labeled data).

**SAHI (Slicing Aided Hyper Inference)** — A trick for detecting small
objects: chop the image into overlapping tiles, run inference on each tile,
then merge boxes. Great for tiny shrimp; costs FPS. `--sahi` enables it.

## Training

**Pretrained weights** — A model that's already been trained on a large
generic dataset (typically COCO). You start fine-tuning from these instead
of random weights, which buys you a huge head start.

**Fine-tuning** — Continuing to train a pretrained model on YOUR data so it
specializes to your scene. The opposite is "training from scratch", which
needs vastly more data.

**Epoch** — One full pass through the training set. AquaScope defaults to
30 epochs for the Jetson, which is enough on a few hundred labels.

**Batch / batch size** — How many images the GPU processes at once. Bigger
batches → faster training but more memory. The Jetson Orin Nano with 8 GB
unified memory only fits batch=2 reliably.

**AMP (Automatic Mixed Precision)** — Train in FP16 (half-precision floats)
instead of FP32, with FP32 kept where stability matters. Roughly halves GPU
memory use. CUDA-only.

**FP16 / FP32** — Floating-point number formats. FP16 is half the bytes (and
roughly 2× the throughput on Tensor Cores) at the cost of precision. AquaScope
trains with AMP and exports the engine in FP16.

**Mosaic / MixUp** — Data augmentation tricks where you stitch four images
into one (mosaic) or blend two (mixup). Boost accuracy on a regular GPU,
but they 4× per-sample RAM and we turn them off on the Jetson.

**Train/val split** — Splitting your labeled data into a training set (the
model learns from these) and a validation set (the model is scored on these
but never trained on them). AquaScope uses 90/10.

**Overfitting** — When a model memorizes training data instead of learning
generalizable patterns. Symptom: train accuracy keeps improving while
validation accuracy plateaus or drops.

**Patience (early stopping)** — How many epochs of no validation improvement
to wait before bailing. AquaScope uses 10. Stops you from over-training.

## Tracking

**ByteTrack** — A simple, fast multi-object tracker that uses Kalman-filter
motion predictions + IoU-based matching. AquaScope wraps the
[supervision](https://supervision.roboflow.com/) implementation.

**Track ID** — A persistent integer assigned to a tracked object. "Fish #3"
keeps that ID across frames as long as the tracker can match it.

**ID switch** — When the tracker mistakenly assigns a new ID to the same
object (often when fish cross paths or temporarily disappear). The most
common tracker failure mode.

**Kalman filter** — A simple math trick for predicting where a moving object
will be in the next frame given where it's been. Used by ByteTrack to predict
each track's next position before matching.

**track_buffer** — Number of frames a track stays "alive" without a matching
detection before being dropped. Higher = more robust to brief occlusions,
but stale tracks linger longer. Set in [bytetrack.yaml](../app/bytetrack.yaml).

## Hardware & deployment

**Jetson Orin Nano** — NVIDIA's $250 ARM-based dev board with 8 GB unified
memory and a small CUDA GPU. AquaScope's primary target. Uses JetPack as its
OS image.

**JetPack** — NVIDIA's all-in-one OS + CUDA + cuDNN + TensorRT bundle for
Jetson hardware. Flashed to the SD card or eMMC. AquaScope is tested on
JetPack 6.1.

**TensorRT** — NVIDIA's GPU inference runtime. Compiles a model for one
specific GPU into an `.engine` file that's faster than the equivalent
PyTorch model.

**.pt file** — A PyTorch checkpoint. Portable across machines, slower than
TensorRT.

**.engine file** — A TensorRT compiled model. Hardware-specific (an Orin
Nano engine won't run on a laptop GPU). Has a fixed input shape.

**MPS (Metal Performance Shaders)** — Apple's GPU compute backend, used on
Apple Silicon Macs. PyTorch supports it; AquaScope auto-detects it.

**CUDA** — NVIDIA's GPU compute platform. Required for TensorRT and for
GPU-accelerated PyTorch on NVIDIA hardware.

**V4L2 (Video4Linux 2)** — The Linux kernel video API. AquaScope uses it
to pull frames from USB webcams on Linux. macOS uses AVFoundation, Windows
uses MSMF.

**MJPEG** — A streaming format where each frame is a standalone JPEG. Cheap
to decode in a browser; AquaScope's dashboard streams MJPEG over HTTP.

## Camera & image processing

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** — A contrast-
boosting trick that works well on uneven lighting. Toggleable from the
dashboard for dim tanks.

**Exposure** — How long the camera sensor collects light per frame. V4L2
exposes both auto and manual modes; AquaScope's `--exposure` flag sets the
manual value.

**Resolution** — Frame dimensions. Lower (480p, 720p) → faster, less memory.
Higher (1080p) → more detail for small fish, but more CPU on the encode
path.

**FPS (Frames Per Second)** — Throughput. AquaScope shows the rolling FPS
in the HUD; on a Jetson Orin Nano with the TensorRT engine you should see
20–30 FPS at 720p.

## See also

- [01-how-it-works.md](01-how-it-works.md) — high-level concepts.
- [02-the-label-train-loop.md](02-the-label-train-loop.md) — dashboard
  walkthrough.
