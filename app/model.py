"""
Model loading and inference for AquaScope — Grounding DINO backend.

Uses the HuggingFace transformers GroundingDinoForObjectDetection API.
Supports both full-frame and SAHI sliced inference.

Model IDs (set via config["model_path"]):
  "IDEA-Research/grounding-dino-tiny"   — Swin-T backbone, publicly downloadable (recommended)
  "IDEA-Research/grounding-dino-base"   — Swin-B backbone, publicly downloadable (slower)

  Note: Grounding DINO 1.5 Edge and 1.6 Edge weights are not publicly released —
  they are DeepDataSpace cloud-API-only models and cannot be run locally.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection


@dataclass
class GroundingDinoDetector:
    """Bundles the processor + model so they travel as one object."""
    processor: AutoProcessor
    model: GroundingDinoForObjectDetection
    device: str


# ── Model loading ─────────────────────────────────────────

def load_model(config: dict) -> GroundingDinoDetector:
    """Load a Grounding DINO model from HuggingFace Hub.

    Forced to CPU: Grounding DINO's multi-scale deformable attention uses a
    cumsum kernel (spatial_shapes.prod(1).cumsum(0)) that raises a CUDA driver
    error on Jetson regardless of dtype. Recompiling the custom CUDA ops for
    Jetson's CUDA version would fix this, but CPU inference is the safe path.
    """
    model_id = config["model_path"]
    device = "cpu"

    print(f"[MODEL] Loading Grounding DINO: {model_id}")
    print(f"[MODEL] Device: {device}  |  dtype: torch.float32")
    print(f"[MODEL] Note: CUDA deformable attention is incompatible with Jetson driver; running on CPU")

    processor = AutoProcessor.from_pretrained(model_id)
    model = (
        GroundingDinoForObjectDetection
        .from_pretrained(model_id)
        .to(device)
        .eval()
    )
    print("[MODEL] Grounding DINO ready")
    return GroundingDinoDetector(processor=processor, model=model, device=device)


# ── SAHI tile helper ──────────────────────────────────────

def slice_tiles(
    frame: np.ndarray, config: dict
) -> list[tuple[np.ndarray, int, int]]:
    """Slice *frame* into overlapping tiles.

    Returns a list of ``(tile, x_offset, y_offset)`` tuples.
    """
    h, w = frame.shape[:2]
    sh = config["sahi_slice_height"]
    sw = config["sahi_slice_width"]
    stride_h = int(sh * (1 - config["sahi_overlap_ratio"]))
    stride_w = int(sw * (1 - config["sahi_overlap_ratio"]))
    tiles: list[tuple[np.ndarray, int, int]] = []
    y = 0
    while True:
        y2 = min(y + sh, h)
        y1 = max(y2 - sh, 0)
        x = 0
        while True:
            x2 = min(x + sw, w)
            x1 = max(x2 - sw, 0)
            tiles.append((frame[y1:y2, x1:x2], x1, y1))
            if x2 == w:
                break
            x += stride_w
        if y2 == h:
            break
        y += stride_h
    return tiles


# ── Inference ─────────────────────────────────────────────

def run_inference(
    detector: GroundingDinoDetector,
    frame: np.ndarray,
    config: dict,
    conf_threshold: float,
) -> sv.Detections:
    """Run Grounding DINO on *frame* and return supervision ``Detections``.

    Uses SAHI sliced inference when ``config["sahi"]`` is truthy.
    """
    if config.get("sahi"):
        all_xyxy: list[np.ndarray] = []
        all_confs: list[np.ndarray] = []

        for tile, x_off, y_off in slice_tiles(frame, config):
            dets = _infer_single(detector, tile, config, conf_threshold)
            if len(dets) == 0:
                continue
            boxes = dets.xyxy.copy()
            boxes[:, [0, 2]] += x_off
            boxes[:, [1, 3]] += y_off
            all_xyxy.append(boxes)
            all_confs.append(dets.confidence)

        if all_xyxy:
            xyxy = np.concatenate(all_xyxy, axis=0).astype(np.float32)
            confs = np.concatenate(all_confs, axis=0).astype(np.float32)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confs = np.empty(0, dtype=np.float32)

        detections = sv.Detections(xyxy=xyxy, confidence=confs)
        detections = detections.with_nms(
            threshold=config["iou_threshold"], class_agnostic=True
        )
        return detections

    return _infer_single(detector, frame, config, conf_threshold)


def _post_process(detector, outputs, inputs, box_threshold, text_threshold, h, w):
    """Call post_process_grounded_object_detection, adapting to transformers API version."""
    import inspect
    fn = detector.processor.post_process_grounded_object_detection
    params = inspect.signature(fn).parameters

    # Build kwargs supported by this transformers version
    kwargs: dict = {"target_sizes": [(h, w)]}
    if "box_threshold" in params:
        kwargs["box_threshold"] = box_threshold
    if "text_threshold" in params:
        kwargs["text_threshold"] = text_threshold
    if "threshold" in params and "box_threshold" not in params:
        kwargs["threshold"] = box_threshold

    # input_ids — some versions take it, some don't
    if "input_ids" in params:
        kwargs["input_ids"] = inputs["input_ids"]

    try:
        return fn(outputs=outputs, **kwargs)
    except TypeError as e:
        # Fall back: pass outputs positionally in case `outputs` kw isn't supported
        print(f"[MODEL] post_process kwarg failed ({e}), retrying positionally")
        return fn(outputs, **kwargs)


def _infer_single(
    detector: GroundingDinoDetector,
    frame: np.ndarray,
    config: dict,
    conf_threshold: float,
) -> sv.Detections:
    """Single forward pass on one frame / tile."""
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    prompt = config.get("detection_prompt", "fish .")

    inputs = detector.processor(
        images=image_pil,
        text=prompt,
        return_tensors="pt",
    ).to(detector.device)

    with torch.no_grad():
        outputs = detector.model(**inputs)

    h, w = frame.shape[:2]
    results = _post_process(
        detector, outputs, inputs, conf_threshold,
        config.get("text_threshold", 0.25), h, w,
    )

    result = results[0]
    if len(result["boxes"]) == 0:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
        )

    xyxy = result["boxes"].cpu().float().numpy()
    confs = result["scores"].cpu().float().numpy()
    return sv.Detections(xyxy=xyxy, confidence=confs)
