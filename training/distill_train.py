"""
Train a YOLOv8n student on DINO-generated pseudo-labels, then export to OpenVINO FP16.

This is the second step of the distillation pipeline:
  1. generate_labels_dino.py  ← teacher labels frames
  2. distill_train.py         ← student learns from those labels  ← YOU ARE HERE

By default the script merges the pseudo-labeled distillation set with the
original Roboflow-labeled dataset so the student benefits from both.
Pass --no-merge to train on pseudo-labels only.

The final export produces:
  runs/detect/<name>/weights/best_openvino_model/   ← FP16 IR ready for model.py

To run on the Jetson, just point config["model_path"] at that directory:
    python app/fish_tracker.py --model runs/detect/.../best_openvino_model/

Usage:
    python training/distill_train.py
    python training/distill_train.py --distill-data distillation/ --epochs 100
    python training/distill_train.py --no-merge --epochs 50
    python training/distill_train.py --base-model yolov8s.pt  # larger student
"""

import os

# Must be set before torch is imported — PyTorch checks this flag at module load
# time to decide whether to register CPU fallback kernels for MPS.
# Fixes the non-deterministic shape-mismatch crash in Ultralytics' TAL loss
# assigner (tal.py bbox_scores / iou_calculation) on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DISTILL = _PROJECT_ROOT / "distillation"
_DEFAULT_ORIG_DATA = _PROJECT_ROOT / "dataset" / "data.yaml"
_DEFAULT_RUN_NAME = "fish_distilled"

IMGSZ = 640


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"          # CUDA GPU 0
    if torch.backends.mps.is_available():
        return "mps"
    print("[DEVICE] No GPU found, using CPU")
    return "cpu"


# ── Dataset merging ───────────────────────────────────────────────────────────

def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _abs_path(base: Path, rel: str) -> str:
    """
    Resolve *rel* against *base*, falling back to stripping leading '../'
    components and looking inside *base* when the standard resolution doesn't
    exist on disk.

    Handles Roboflow data.yaml files that ship with '../train/images' even
    though train/ sits inside the same directory as data.yaml.
    """
    p = Path(rel)
    if p.is_absolute():
        return str(p.resolve())

    standard = (base / p).resolve()
    if standard.exists():
        return str(standard)

    # Strip every leading '../' and try within base
    parts = p.parts
    stripped = Path(*[pt for pt in parts if pt != ".."])
    local = (base / stripped).resolve()
    if local.exists():
        return str(local)

    return str(standard)  # best guess; YOLO will give a clear error if missing


def build_merged_yaml(distill_dir: Path, orig_yaml: Path, tmp_dir: Path) -> Path:
    """
    Create a temporary data.yaml that points YOLO at both datasets.

    YOLO supports a list under `train:` and `val:` so we can mix sources without
    copying any images.

    Validates that both datasets share the same classes (by name) before merging.
    """
    d_data = _read_yaml(distill_dir / "data.yaml")
    o_data = _read_yaml(orig_yaml)

    # Verify class lists match
    d_names = d_data.get("names", [])
    o_names = o_data.get("names", [])
    if d_names != o_names:
        print(f"[WARN] Class mismatch between datasets:")
        print(f"       distill: {d_names}")
        print(f"       original: {o_names}")
        print(f"       Using original dataset classes.")

    # Resolve all paths to absolute so YOLO never mis-resolves relative components
    distill_images = str((distill_dir / "images").resolve())
    orig_base = orig_yaml.parent.resolve()
    orig_train = _abs_path(orig_base, o_data.get("train", "train/images"))
    orig_val   = _abs_path(orig_base, o_data.get("val",   "valid/images"))

    # Omit 'path' — with fully absolute train/val entries YOLO uses them as-is
    merged = {
        "train": [distill_images, orig_train],
        "val":   orig_val,        # use original labeled set for validation (ground truth)
        "nc":    o_data["nc"],
        "names": o_names,
    }

    merged_yaml = tmp_dir / "merged_data.yaml"
    with open(merged_yaml, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

    print(f"[DATA] Merged dataset:")
    print(f"       distill train : {distill_images}")
    print(f"       orig train    : {orig_train}")
    print(f"       orig val      : {orig_val}")
    print(f"       classes       : {o_names}")
    return merged_yaml


def build_distill_only_yaml(distill_dir: Path, tmp_dir: Path) -> Path:
    """
    Create a data.yaml for pseudo-labels only, with a random 90/10 train/val split.
    Used when --no-merge is set.
    """
    from sklearn.model_selection import train_test_split
    import random

    images_dir = distill_dir / "images"
    all_images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    if not all_images:
        print(f"ERROR: No images found in {images_dir}")
        sys.exit(1)

    train_imgs, val_imgs = train_test_split(all_images, test_size=0.10, random_state=42)

    # Write split list files that YOLO accepts
    train_txt = tmp_dir / "train.txt"
    val_txt   = tmp_dir / "val.txt"
    train_txt.write_text("\n".join(str(p) for p in train_imgs))
    val_txt.write_text("\n".join(str(p) for p in val_imgs))

    d_data = _read_yaml(distill_dir / "data.yaml")
    data_yaml = tmp_dir / "distill_data.yaml"
    config = {
        "path":  str(tmp_dir),
        "train": str(train_txt),
        "val":   str(val_txt),
        "nc":    d_data["nc"],
        "names": d_data["names"],
    }
    with open(data_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[DATA] Distill-only split: {len(train_imgs)} train / {len(val_imgs)} val")
    return data_yaml


# ── Resume helper ─────────────────────────────────────────────────────────────

def find_last_checkpoint(run_name: str) -> Path | None:
    """
    Find the most recent last.pt for *run_name*.

    Ultralytics appends a number when a run name already exists
    (fish_distilled, fish_distilled2, fish_distilled3 …).
    We pick the highest-numbered directory that has a last.pt.
    """
    runs_root = _PROJECT_ROOT / "runs" / "detect"
    if not runs_root.exists():
        # Try relative path (when running from training/ or project root)
        runs_root = Path("runs") / "detect"
    if not runs_root.exists():
        return None

    candidates = sorted(runs_root.glob(f"{run_name}*/weights/last.pt"))
    if not candidates:
        return None
    # Pick the one whose parent run dir has the highest suffix number
    def _sort_key(p: Path) -> int:
        suffix = p.parent.parent.name.replace(run_name, "").strip()
        return int(suffix) if suffix.isdigit() else 0

    return sorted(candidates, key=_sort_key)[-1]


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data_yaml: Path,
    base_model: str,
    epochs: int,
    batch: int,
    device: str,
    run_name: str,
    amp: bool,
    resume: bool = False,
) -> Path:
    from ultralytics import YOLO

    if resume:
        last_pt = find_last_checkpoint(run_name)
        if last_pt is None:
            print(f"[WARN] --resume: no last.pt found for '{run_name}', starting fresh")
            resume = False
        else:
            print(f"[RESUME] Continuing from: {last_pt}")
            base_model = str(last_pt)

    print(f"\n{'='*60}")
    print(f"  Distillation Training {'(RESUME)' if resume else ''}")
    print(f"  Base model : {base_model}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch      : {batch}")
    print(f"  Device     : {device}")
    print(f"  Run name   : {run_name}")
    print(f"{'='*60}\n")

    model = YOLO(base_model)

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=IMGSZ,
        device=device,
        batch=batch,
        workers=8,
        amp=amp,
        cache=False,
        name=run_name,
        resume=resume,
        patience=20,
        save=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        # Augmentation — helps the student generalise beyond teacher artifacts
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.3,
        fliplr=0.5,
        # Label smoothing reduces over-confidence on pseudo-labels
        label_smoothing=0.05,
    )

    best_pt = Path(f"runs/detect/{run_name}/weights/best.pt")
    if not best_pt.exists():
        print(f"[WARN] best.pt not found at {best_pt} — checking last.pt")
        best_pt = Path(f"runs/detect/{run_name}/weights/last.pt")

    if not best_pt.exists():
        print(f"ERROR: No weights found under runs/detect/{run_name}/weights/")
        sys.exit(1)

    print(f"\n[TRAIN] Best weights: {best_pt}")
    return best_pt


# ── Export to OpenVINO FP16 ───────────────────────────────────────────────────

def export_openvino(best_pt: Path) -> Path:
    """
    Export the trained .pt to OpenVINO IR in FP16.

    Produces:  <best_pt_parent>/best_openvino_model/
    That directory can be passed directly to app/fish_tracker.py via --model.
    """
    from ultralytics import YOLO

    print(f"\n[EXPORT] Converting {best_pt.name} → OpenVINO FP16 ...")
    model = YOLO(str(best_pt))

    export_path = model.export(
        format="openvino",
        imgsz=IMGSZ,
        half=True,       # FP16
        dynamic=False,
        simplify=True,
    )

    ov_dir = Path(str(export_path))
    print(f"\n[EXPORT] OpenVINO FP16 model: {ov_dir}")
    print(f"\nTo use in AquaScope:")
    print(f"  python app/fish_tracker.py --model {ov_dir}")
    return ov_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n student on DINO pseudo-labels → export OpenVINO FP16",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--distill-data",
        type=Path,
        default=_DEFAULT_DISTILL,
        help=f"Distillation output directory from generate_labels_dino.py "
             f"(default: {_DEFAULT_DISTILL})",
    )
    parser.add_argument(
        "--orig-data",
        type=Path,
        default=_DEFAULT_ORIG_DATA,
        help=f"Original labeled dataset data.yaml to merge with "
             f"(default: {_DEFAULT_ORIG_DATA})",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Train on pseudo-labels only (skip merging with original dataset)",
    )
    parser.add_argument(
        "--base-model",
        default="yolov8n.pt",
        help="Starting weights for the student (default: yolov8n.pt). "
             "Use yolov8s.pt for a larger student.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size; reduce if OOM (default: 16)",
    )
    parser.add_argument(
        "--name",
        default=_DEFAULT_RUN_NAME,
        help=f"Run name under runs/detect/ (default: {_DEFAULT_RUN_NAME})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest last.pt checkpoint for --name",
    )
    parser.add_argument(
        "--export-only",
        type=Path,
        default=None,
        metavar="WEIGHTS_PT",
        help="Skip training; export an existing .pt directly to OpenVINO FP16",
    )
    args = parser.parse_args()

    # Resolve relative paths to absolute so they survive being embedded in temp YAMLs
    args.distill_data = args.distill_data.resolve()
    args.orig_data    = args.orig_data.resolve()

    # ── Export-only shortcut ──────────────────────────────────────────────────
    if args.export_only is not None:
        if not args.export_only.exists():
            print(f"ERROR: {args.export_only} not found")
            sys.exit(1)
        export_openvino(args.export_only)
        return

    # ── Validate distillation data ────────────────────────────────────────────
    if not args.distill_data.exists():
        print(f"ERROR: Distillation directory not found: {args.distill_data}")
        print(f"  Run first:  python training/generate_labels_dino.py")
        sys.exit(1)

    distill_yaml = args.distill_data / "data.yaml"
    if not distill_yaml.exists():
        print(f"ERROR: data.yaml missing in {args.distill_data}")
        print(f"  Re-run generate_labels_dino.py to regenerate it.")
        sys.exit(1)

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device()
    # MPS: disable amp (unstable on some torch versions)
    amp = device not in ("mps", "cpu")

    # ── Build data.yaml in a temp dir ─────────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="aquascope_distill_"))
    try:
        if args.no_merge:
            print("[DATA] Training on pseudo-labels only (--no-merge)")
            try:
                data_yaml = build_distill_only_yaml(args.distill_data, tmp_dir)
            except ImportError:
                print("[WARN] scikit-learn not installed; using all images for both train and val")
                data_yaml = distill_yaml
        else:
            if not args.orig_data.exists():
                print(f"[WARN] Original data.yaml not found at {args.orig_data}")
                print(f"       Falling back to pseudo-labels only")
                try:
                    data_yaml = build_distill_only_yaml(args.distill_data, tmp_dir)
                except ImportError:
                    data_yaml = distill_yaml
            else:
                data_yaml = build_merged_yaml(args.distill_data, args.orig_data, tmp_dir)

        # ── Train ─────────────────────────────────────────────────────────────
        best_pt = train(
            data_yaml=data_yaml,
            base_model=args.base_model,
            epochs=args.epochs,
            batch=args.batch,
            device=device,
            run_name=args.name,
            amp=amp,
            resume=args.resume,
        )

        # ── Export ────────────────────────────────────────────────────────────
        export_openvino(best_pt)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"  Distillation pipeline complete!")
    print(f"  Student weights : runs/detect/{args.name}/weights/best.pt")
    print(f"  OpenVINO FP16   : runs/detect/{args.name}/weights/best_openvino_model/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
