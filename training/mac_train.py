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
import glob
import os
import tempfile

# Project root = one level above this file (training/)
_PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA     = os.path.join(_PROJECT_ROOT, "dataset", "data.yaml")
_DEFAULT_USERDATA = os.path.join(_PROJECT_ROOT, "dataset", "user_recorded")

import torch
import yaml
from ultralytics import YOLO

IMGSZ = 640


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    print("MPS not available, falling back to CPU")
    return "cpu"


def _has_user_images(user_dir: str) -> bool:
    images_dir = os.path.join(user_dir, "images")
    if not os.path.isdir(images_dir):
        return False
    return bool(glob.glob(os.path.join(images_dir, "*.jpg")) or
                glob.glob(os.path.join(images_dir, "*.png")))


def maybe_merge_user_data(orig_yaml: str, user_dir: str) -> str:
    """
    If user_dir has labeled images, return a path to a tmp data.yaml that
    appends them to the train list. Otherwise return orig_yaml unchanged.
    Validation always uses the original yaml's val split (user labels are noisier).
    """
    if not _has_user_images(user_dir):
        return orig_yaml

    with open(orig_yaml) as f:
        data = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(orig_yaml))

    def _abs(p):
        if os.path.isabs(p):
            return p
        cand = os.path.normpath(os.path.join(base, p))
        if os.path.exists(cand):
            return cand
        # strip leading ../
        return os.path.normpath(os.path.join(base, *[s for s in p.split(os.sep) if s != ".."]))

    orig_train = _abs(data.get("train", "train/images"))
    orig_val   = _abs(data.get("val", "valid/images"))
    user_train = os.path.abspath(os.path.join(user_dir, "images"))

    tmp_dir = tempfile.mkdtemp(prefix="aquascope_user_")
    merged_yaml = os.path.join(tmp_dir, "merged_user_data.yaml")
    with open(merged_yaml, "w") as f:
        yaml.safe_dump({
            "train": [user_train, orig_train],
            "val":   orig_val,
            "nc":    data["nc"],
            "names": data["names"],
        }, f)
    print(f"[DATA] Merging user_recorded into training set:")
    print(f"       user train : {user_train}")
    print(f"       orig train : {orig_train}")
    print(f"       orig val   : {orig_val}")
    return merged_yaml


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
    parser.add_argument(
        "--user-data",
        default=_DEFAULT_USERDATA,
        help=f"User-recorded labels from the dashboard (default: {_DEFAULT_USERDATA}). "
             "Silently skipped if missing or empty.",
    )
    parser.add_argument(
        "--no-user",
        action="store_true",
        help="Skip dataset/user_recorded/ even if it has labeled images",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size; reduce if you see memory pressure (Activity Monitor)",
    )
    args = parser.parse_args()

    data_path = args.data
    if not args.no_user:
        data_path = maybe_merge_user_data(args.data, args.user_data)

    train(data_path, epochs=args.epochs, batch=args.batch)
