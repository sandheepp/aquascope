"""
Model loading and inference for AquaScope.

Two backends — selected automatically from config["model_path"]:

  YOLOv8 (Ultralytics)   — .pt, .engine, or *_openvino_model/ directory
  OpenVINO IR (custom)   — any .xml file exported from your own model

OpenVINO export from a fine-tuned YOLOv8:
    yolo export model=fish_best.pt format=openvino imgsz=640
    → produces fish_best_openvino_model/   (pass the directory, handled as YOLO)

    OR:
    yolo export model=fish_best.pt format=onnx
    mo --input_model fish_best.onnx ...    → produces fish_best.xml   (pass the .xml, uses raw OV runtime)
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


# ── OpenVINO detector wrapper ─────────────────────────────

@dataclass
class OpenVINODetector:
    """Wraps a compiled OpenVINO IR model."""
    compiled: object      # openvino.CompiledModel
    output_layer: object  # output tensor descriptor
    input_h: int
    input_w: int


# ── Model loading ─────────────────────────────────────────

def load_model(config: dict) -> "YOLO | OpenVINODetector":
    """Return a YOLO or OpenVINODetector depending on config['model_path']."""
    path = config["model_path"]
    if path.endswith(".xml"):
        return _load_openvino(path, config)
    return _load_yolo(path, config)


def _load_yolo(path: str, config: dict) -> YOLO:
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


def _load_openvino(xml_path: str, config: dict) -> OpenVINODetector:
    try:
        from openvino.runtime import Core
    except ImportError:
        raise ImportError("OpenVINO not installed. Run: pip install openvino")

    h, w = config.get("model_input_size", [640, 640])
    device = config.get("openvino_device", "CPU")

    print(f"[MODEL] Loading OpenVINO IR: {xml_path}")
    print(f"[MODEL] OV device: {device}  |  input size: {w}×{h}")

    ie = Core()
    model = ie.read_model(xml_path)
    compiled = ie.compile_model(model, device)
    print("[MODEL] OpenVINO model ready")
    return OpenVINODetector(
        compiled=compiled,
        output_layer=compiled.output(0),
        input_h=h,
        input_w=w,
    )


# ── SAHI tile helper ──────────────────────────────────────

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


# ── Inference dispatcher ──────────────────────────────────

def run_inference(
    model: "YOLO | OpenVINODetector",
    frame: np.ndarray,
    config: dict,
    conf_threshold: float,
) -> sv.Detections:
    if isinstance(model, OpenVINODetector):
        return _run_openvino(model, frame, config, conf_threshold)
    return _run_yolo(model, frame, config, conf_threshold)


# ── YOLOv8 backend ────────────────────────────────────────

def _run_yolo(
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


# ── OpenVINO backend ──────────────────────────────────────

def _run_openvino(
    detector: OpenVINODetector,
    frame: np.ndarray,
    config: dict,
    conf_threshold: float,
) -> sv.Detections:
    if config.get("sahi"):
        all_xyxy: list[np.ndarray] = []
        all_confs: list[np.ndarray] = []
        for tile, x_off, y_off in slice_tiles(frame, config):
            dets = _infer_openvino_single(detector, tile, conf_threshold, config)
            if len(dets) == 0:
                continue
            boxes = dets.xyxy.copy()
            boxes[:, [0, 2]] += x_off
            boxes[:, [1, 3]] += y_off
            all_xyxy.append(boxes)
            all_confs.append(dets.confidence)

        if all_xyxy:
            xyxy = np.concatenate(all_xyxy).astype(np.float32)
            confs = np.concatenate(all_confs).astype(np.float32)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confs = np.empty(0, dtype=np.float32)
        dets = sv.Detections(xyxy=xyxy, confidence=confs)
        return dets.with_nms(threshold=config["iou_threshold"], class_agnostic=True)

    return _infer_openvino_single(detector, frame, conf_threshold, config)


def _infer_openvino_single(
    detector: OpenVINODetector,
    frame: np.ndarray,
    conf_threshold: float,
    config: dict,
) -> sv.Detections:
    """One forward pass through the OpenVINO model.

    Expects YOLOv8-style output: [1, 4+num_classes, num_anchors]
    where the 4 box values are cx, cy, w, h in input-image pixel coords.
    """
    frame_h, frame_w = frame.shape[:2]
    input_h, input_w = detector.input_h, detector.input_w

    # Resize + normalise → [1, 3, H, W]  (BGR→RGB via swapRB)
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1 / 255.0,
        size=(input_w, input_h),
        swapRB=True, crop=False,
    )

    result = detector.compiled([blob])[detector.output_layer]  # [1, 4+nc, na]
    pred = result[0]  # [4+nc, na]

    # Normalise to [na, 4+nc] — YOLOv8 outputs num_classes+4 rows, na columns
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # → [na, 4+nc]

    boxes_raw = pred[:, :4]        # cx, cy, w, h  in input-px coords
    class_scores = pred[:, 4:]     # [na, nc]

    # Filter by confidence
    detect_classes = config.get("detect_classes")
    if detect_classes is not None:
        confs = class_scores[:, detect_classes].max(axis=1)
    else:
        confs = class_scores.max(axis=1)

    mask = confs >= conf_threshold
    if not mask.any():
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
        )

    boxes_raw = boxes_raw[mask]
    confs = confs[mask]

    # cx,cy,w,h (in 640×640 space) → x1,y1,x2,y2 (in original frame space)
    scale_x = frame_w / input_w
    scale_y = frame_h / input_h
    cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = (cx - bw / 2) * scale_x
    y1 = (cy - bh / 2) * scale_y
    x2 = (cx + bw / 2) * scale_x
    y2 = (cy + bh / 2) * scale_y
    xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    dets = sv.Detections(xyxy=xyxy, confidence=confs.astype(np.float32))
    return dets.with_nms(threshold=config["iou_threshold"], class_agnostic=True)
