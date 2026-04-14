"""
Train YOLOv8s on Mac (Apple Silicon / MPS).

Differences from quick_train.py (Jetson):
- No jetson_compat shim needed
- device="mps" (Metal Performance Shaders) or "cpu" fallback
- amp=False  — MPS autocast is unstable on some torch builds
- mosaic/mixup re-enabled — no Jetson OOM constraints
- batch=8    — Mac has enough RAM; tune up/down as needed
- workers=8  — more CPU cores available
- Export to CoreML (.mlpackage) instead of TensorRT

Usage:
    python training/mac_train.py
    python training/mac_train.py --data dataset/data.yaml --epochs 100 --batch 8
"""

import argparse
import os

# Project root = one level above this file (training/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA = os.path.join(_PROJECT_ROOT, "dataset", "data.yaml")

import torch
from ultralytics import YOLO

IMGSZ = 640


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    print("MPS not available, falling back to CPU")
    return "cpu"


def train(data_yaml: str, epochs: int = 50, batch: int = 8) -> None:
    device = get_device()
    print(f"Using device: {device}")

    model = YOLO("yolov8s.pt")

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=IMGSZ,
        device=device,
        batch=batch,
        workers=8,
        amp=False,          # MPS autocast unstable on some torch versions
        cache=False,
        name="fish_detector_mac",
        patience=20,
        save=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        augment=True,
        mosaic=1.0,         # safe to enable on Mac (no Jetson OOM risk)
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.3,
        fliplr=0.5,
    )

    best_pt = "runs/detect/fish_detector_mac/weights/best.pt"
    if not os.path.exists(best_pt):
        print(f"Warning: {best_pt} not found — skipping export")
        return

    best_model = YOLO(best_pt)

    # CoreML export — runs natively on Apple Silicon via ANE / GPU
    best_model.export(format="coreml", imgsz=IMGSZ, nms=True)
    print("\nBest model exported to CoreML (.mlpackage)")
    print(f"Weights also available as PyTorch: {best_pt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8s on Mac (MPS)")
    parser.add_argument(
        "--data",
        default=_DEFAULT_DATA,
        help="Path to data.yaml (default: dataset/data.yaml relative to project root)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size; reduce if you see memory pressure (Activity Monitor)",
    )
    args = parser.parse_args()

    train(args.data, epochs=args.epochs, batch=args.batch)
