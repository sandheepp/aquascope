"""
Model loading and inference for AquaScope.

Wraps YOLOv8 (+ optional SAHI sliced inference) and returns supervision Detections.
"""

import os

import numpy as np
import supervision as sv
from ultralytics import YOLO


def load_model(config: dict) -> YOLO:
    """Load a YOLOv8 model, falling back from .engine to .pt when needed."""
    path = config["model_path"]
    if path.endswith(".engine") and not os.path.exists(path):
        pt = path.replace(".engine", ".pt")
        print(f"[MODEL] Engine not found: {path}")
        print(f"[MODEL] Falling back to: {pt}")
        print(
            f"[MODEL] Export hint: python3 -c \"from ultralytics import YOLO; "
            f"YOLO('{pt}').export(format='engine', device=0, half=True, "
            f"imgsz={config['imgsz']})\""
        )
        path = pt
    print(f"[MODEL] Loading YOLO: {path}")
    return YOLO(path, task="detect")


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
                device=0,
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
        device=0,
    )
    if results and results[0].boxes is not None and len(results[0].boxes):
        xyxy = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = results[0].boxes.conf.cpu().numpy().astype(np.float32)
    else:
        xyxy = np.empty((0, 4), dtype=np.float32)
        confs = np.empty(0, dtype=np.float32)
    return sv.Detections(xyxy=xyxy, confidence=confs)
