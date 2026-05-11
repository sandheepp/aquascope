"""
Model loading and inference for AquaScope.

Wraps YOLOv8 (+ optional SAHI sliced inference) and returns supervision Detections.

Device selection is automatic: CUDA (Jetson, NVIDIA laptop) → MPS (Apple
Silicon) → CPU. The active device is cached in `config["_device"]` after
the first load so inference doesn't re-detect every frame.

Model fallback chain (so a fresh laptop clone "just works"):
  1. The `--model` value the user passed.
  2. If that's a `.engine` and the file is missing, try the matching `.pt`.
  3. If THAT is also missing AND the user didn't override the default, fall
     back to `yolov8n.pt` — Ultralytics will download it on first use, so a
     laptop user can run the dashboard immediately without a Jetson engine.
"""

import os

import numpy as np
import supervision as sv
from ultralytics import YOLO


def _detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' — first one that's actually usable."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # MPS (Apple Silicon): older PyTorch builds don't have the attribute.
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _resolve_model_path(path: str) -> str:
    """Walk the fallback chain (.engine → .pt → yolov8n.pt). Returns whatever
    we'll actually hand to YOLO()."""
    if os.path.exists(path):
        return path

    # .engine missing → try sibling .pt
    if path.endswith(".engine"):
        pt = path.replace(".engine", ".pt")
        if os.path.exists(pt):
            print(f"[MODEL] Engine not found ({path}); using {pt} instead.")
            return pt
        # Default model location wasn't built yet (fresh clone, no Jetson) — fall
        # through to yolov8n.pt rather than crashing.
        if path == "models/best.engine":
            print(f"[MODEL] No fine-tuned weights yet; using yolov8n.pt "
                  f"(Ultralytics will download on first run).")
            return "yolov8n.pt"

    # .pt missing — let YOLO() try to download it (works for the standard
    # yolov8{n,s,m,l,x}.pt names) or raise a clearer error.
    print(f"[MODEL] {path} not found locally; YOLO() will attempt a download.")
    return path


def load_model(config: dict) -> YOLO:
    """Load a YOLOv8 model with sensible cross-platform defaults."""
    resolved = _resolve_model_path(config["model_path"])
    config["model_path"] = resolved

    device = config.get("_device") or _detect_device()
    config["_device"] = device
    print(f"[MODEL] Loading {resolved} on device='{device}'")

    model = YOLO(resolved, task="detect")
    # Ultralytics moves the model to whatever `device=` the predict/train call
    # uses, so we don't need an explicit .to(device) here — but stash it on
    # config so run_inference() doesn't have to re-detect.
    return model


def slice_tiles(
    frame: np.ndarray, config: dict
) -> list[tuple[np.ndarray, int, int]]:
    """Slice *frame* into overlapping tiles → [(tile, x_off, y_off)]."""
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


def run_inference(
    model: YOLO,
    frame: np.ndarray,
    config: dict,
    conf_threshold: float,
) -> sv.Detections:
    device = config.get("_device") or _detect_device()
    config["_device"] = device

    if config.get("sahi"):
        all_xyxy: list[np.ndarray] = []
        all_confs: list[np.ndarray] = []
        for tile, x_off, y_off in slice_tiles(frame, config):
            results = model.predict(
                source=tile,
                conf=conf_threshold,
                iou=config["iou_threshold"],
                imgsz=config["imgsz"],
                verbose=False,
                classes=config.get("detect_classes"),
                device=device,
            )
            if not (results and results[0].boxes is not None and len(results[0].boxes)):
                continue
            boxes = results[0].boxes.xyxy.cpu().numpy().copy()
            boxes[:, [0, 2]] += x_off
            boxes[:, [1, 3]] += y_off
            all_xyxy.append(boxes)
            all_confs.append(results[0].boxes.conf.cpu().numpy())

        if all_xyxy:
            xyxy = np.concatenate(all_xyxy, axis=0).astype(np.float32)
            confs = np.concatenate(all_confs, axis=0).astype(np.float32)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confs = np.empty(0, dtype=np.float32)
        detections = sv.Detections(xyxy=xyxy, confidence=confs)
        return detections.with_nms(threshold=config["iou_threshold"], class_agnostic=True)

    results = model.predict(
        source=frame,
        conf=conf_threshold,
        iou=config["iou_threshold"],
        imgsz=config["imgsz"],
        verbose=False,
        classes=config.get("detect_classes"),
        device=device,
    )
    if results and results[0].boxes is not None and len(results[0].boxes):
        xyxy = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = results[0].boxes.conf.cpu().numpy().astype(np.float32)
    else:
        xyxy = np.empty((0, 4), dtype=np.float32)
        confs = np.empty(0, dtype=np.float32)
    return sv.Detections(xyxy=xyxy, confidence=confs)
