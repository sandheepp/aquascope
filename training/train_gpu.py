"""
Train YOLOv8n on dataset + distillation data (NVIDIA GPU), then export to
TensorRT FP16 .engine for deployment on the Jetson or any CUDA device.

Pipeline:
  1. Merge dataset/data.yaml (Roboflow labels) + distillation/images
     (DINO pseudo-labels from generate_labels_dino.py).
  2. Train YOLOv8n with full augmentation + AMP.
  3. Export best.pt → best.engine (TensorRT FP16).

Usage (local, GPU available):
    python training/train_gpu.py
    python training/train_gpu.py --epochs 150 --batch 32
    python training/train_gpu.py --no-merge        # pseudo-labels only
    python training/train_gpu.py --export-only runs/detect/fish_gpu/weights/best.pt

Usage (Docker — preferred):
    docker compose -f training/docker-compose.gpu.yml up
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATASET  = _PROJECT_ROOT / "dataset"
_DEFAULT_DISTILL  = _PROJECT_ROOT / "dataset" / "distillation"
_DEFAULT_USERDATA = _PROJECT_ROOT / "dataset" / "user_recorded"
_DEFAULT_RUN_NAME = "fish_gpu"
IMGSZ = 640


def _user_data_train_path(user_dir: Path) -> str | None:
    """Return resolved path to user_recorded/images if it has any images, else None."""
    if not user_dir.exists():
        return None
    images_dir = user_dir / "images"
    if not images_dir.is_dir():
        return None
    has_any = any(images_dir.glob("*.jpg")) or any(images_dir.glob("*.png"))
    if not has_any:
        return None
    return str(images_dir.resolve())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _abs_path(base: Path, rel: str) -> str:
    """Resolve rel against base, stripping leading '../' if needed."""
    p = Path(rel)
    if p.is_absolute():
        return str(p.resolve())
    standard = (base / p).resolve()
    if standard.exists():
        return str(standard)
    stripped = Path(*[pt for pt in p.parts if pt != ".."])
    local = (base / stripped).resolve()
    return str(local) if local.exists() else str(standard)


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_merged_yaml(
    dataset_dir: Path,
    distill_dir: Path | None,
    user_dir: Path | None,
    tmp_dir: Path,
) -> Path:
    """
    Merge Roboflow-labeled dataset with optional DINO pseudo-labels and optional
    user-recorded labels (from the in-dashboard labeling tab). Validation always
    uses the ground-truth Roboflow set — user/distill labels are noisier.

    Pass None for distill_dir or user_dir to skip that source.
    """
    orig_yaml = dataset_dir / "data.yaml"
    o_data    = _read_yaml(orig_yaml)
    orig_base = orig_yaml.parent.resolve()

    orig_train = _abs_path(orig_base, o_data.get("train", "train/images"))
    orig_val   = _abs_path(orig_base, o_data.get("val",   "valid/images"))

    train_sources: list[str] = []
    if distill_dir is not None:
        distill_images = str((distill_dir / "images").resolve())
        train_sources.append(distill_images)
    if user_dir is not None:
        user_images = _user_data_train_path(user_dir)
        if user_images:
            train_sources.append(user_images)
    train_sources.append(orig_train)

    merged = {
        "train": train_sources if len(train_sources) > 1 else train_sources[0],
        "val":   orig_val,
        "nc":    o_data["nc"],
        "names": o_data["names"],
    }

    merged_yaml = tmp_dir / "merged_data.yaml"
    with open(merged_yaml, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

    print("[DATA] Merged dataset:")
    for src in train_sources[:-1]:
        print(f"       extra train  : {src}")
    print(f"       orig train   : {orig_train}")
    print(f"       orig val     : {orig_val}")
    print(f"       classes ({o_data['nc']}) : {o_data['names']}")
    return merged_yaml


def build_user_only_yaml(
    user_dir: Path,
    tmp_dir: Path,
    class_names: list[str],
    val_split: float = 0.10,
) -> Path:
    """90/10 train/val split over user-recorded labels only, with a caller-
    supplied class list (so the trained model only knows the classes the user
    actually cares about, regardless of what the labeling-time inference model
    happened to predict).

    Labels with a class index outside the new range are dropped from the
    label file before training — keeps stale labels from a previous N-class
    inference model from breaking the new run.
    """
    import random

    images_dir = user_dir / "images"
    labels_dir = user_dir / "labels"
    all_images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not all_images:
        print(f"ERROR: no images in {images_dir}")
        sys.exit(1)

    # YOLO finds labels by replacing the LAST '/images/' with '/labels/' in
    # an image's path, so the filtered set must live at tmp/images/ + tmp/labels/.
    images_link_dir = tmp_dir / "images"
    labels_filter_dir = tmp_dir / "labels"
    images_link_dir.mkdir(exist_ok=True)
    labels_filter_dir.mkdir(exist_ok=True)

    nc = len(class_names)
    kept_images: list[Path] = []
    dropped = 0
    for img in all_images:
        lbl_src = labels_dir / (img.stem + ".txt")
        if not lbl_src.exists():
            continue
        kept_lines: list[str] = []
        for line in lbl_src.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if 0 <= cid < nc:
                kept_lines.append(line)
            else:
                dropped += 1
        if not kept_lines:
            continue
        (labels_filter_dir / lbl_src.name).write_text("\n".join(kept_lines) + "\n")
        link = images_link_dir / img.name
        if not link.exists():
            try:
                link.symlink_to(img.resolve())
            except OSError:
                import shutil as _sh
                _sh.copy2(img, link)
        kept_images.append(img)
    if not kept_images:
        print(f"ERROR: no labels in [0, {nc}) range under {labels_dir}")
        sys.exit(1)
    if dropped:
        print(f"[DATA] Dropped {dropped} labels with class_idx outside [0, {nc}).")

    if len(kept_images) < 10:
        train_imgs = val_imgs = kept_images
    else:
        rng = random.Random(42)
        shuffled = list(kept_images)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_split)))
        val_imgs = shuffled[:n_val]
        train_imgs = shuffled[n_val:]

    train_txt = tmp_dir / "train.txt"
    val_txt   = tmp_dir / "val.txt"
    train_txt.write_text("\n".join(str(images_link_dir / p.name) for p in train_imgs))
    val_txt.write_text("\n".join(str(images_link_dir / p.name) for p in val_imgs))

    data_yaml = tmp_dir / "user_only_data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump({
            "path":  str(tmp_dir),
            "train": str(train_txt),
            "val":   str(val_txt),
            "nc":    nc,
            "names": list(class_names),
        }, f, default_flow_style=False, sort_keys=False)

    print(f"[DATA] User-only: {len(train_imgs)} train / {len(val_imgs)} val "
          f"| classes ({nc}): {class_names}")
    return data_yaml


def build_distill_only_yaml(distill_dir: Path, tmp_dir: Path) -> Path:
    """90/10 train/val split over pseudo-labels only."""
    from sklearn.model_selection import train_test_split

    images_dir = distill_dir / "images"
    all_images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not all_images:
        print(f"ERROR: no images in {images_dir}")
        sys.exit(1)

    train_imgs, val_imgs = train_test_split(all_images, test_size=0.10, random_state=42)
    train_txt = tmp_dir / "train.txt"
    val_txt   = tmp_dir / "val.txt"
    train_txt.write_text("\n".join(str(p) for p in train_imgs))
    val_txt.write_text("\n".join(str(p) for p in val_imgs))

    d_data = _read_yaml(distill_dir / "data.yaml")
    data_yaml = tmp_dir / "distill_data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump({
            "path":  str(tmp_dir),
            "train": str(train_txt),
            "val":   str(val_txt),
            "nc":    d_data["nc"],
            "names": d_data["names"],
        }, f, default_flow_style=False, sort_keys=False)

    print(f"[DATA] Distill-only: {len(train_imgs)} train / {len(val_imgs)} val")
    return data_yaml


# ── Resume helper ─────────────────────────────────────────────────────────────

def find_last_checkpoint(run_name: str) -> Path | None:
    runs_root = _PROJECT_ROOT / "runs" / "detect"
    if not runs_root.exists():
        runs_root = Path("runs") / "detect"
    if not runs_root.exists():
        return None
    candidates = sorted(runs_root.glob(f"{run_name}*/weights/last.pt"))
    if not candidates:
        return None

    def _key(p: Path) -> int:
        suffix = p.parent.parent.name.replace(run_name, "").strip()
        return int(suffix) if suffix.isdigit() else 0

    return sorted(candidates, key=_key)[-1]


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data_yaml: Path,
    base_model: str,
    epochs: int,
    batch: int,
    run_name: str,
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
    print(f"  GPU Training {'(RESUME)' if resume else ''}")
    print(f"  Base model : {base_model}")
    print(f"  Epochs     : {epochs}  |  Batch: {batch}  |  imgsz: {IMGSZ}")
    print(f"  Run name   : {run_name}")
    print(f"{'='*60}\n")

    model = YOLO(base_model)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=IMGSZ,
        device=0,           # GPU 0
        batch=batch,
        workers=8,
        amp=True,           # AMP FP16 training (faster on CUDA)
        cache=False,
        name=run_name,
        resume=resume,
        patience=20,
        save=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.3,
        fliplr=0.5,
        label_smoothing=0.05,
    )

    best_pt = Path(f"runs/detect/{run_name}/weights/best.pt")
    if not best_pt.exists():
        best_pt = Path(f"runs/detect/{run_name}/weights/last.pt")
    if not best_pt.exists():
        print(f"ERROR: no weights under runs/detect/{run_name}/weights/")
        sys.exit(1)

    print(f"\n[TRAIN] Best weights: {best_pt}")
    return best_pt


# ── TensorRT export ───────────────────────────────────────────────────────────

def export_engine(best_pt: Path, batch: int = 1) -> Path:
    """
    Export .pt → TensorRT FP16 .engine.
    The .engine file sits alongside best.pt in weights/.

    batch=1 produces a fixed-batch engine optimal for the Jetson inference loop.
    Pass batch=-1 for a dynamic-batch engine if you need variable batch sizes.
    """
    from ultralytics import YOLO

    print(f"\n[EXPORT] {best_pt.name} → TensorRT FP16 .engine ...")
    model = YOLO(str(best_pt))

    export_path = model.export(
        format="engine",
        imgsz=IMGSZ,
        half=True,          # FP16
        device=0,
        batch=batch,
        simplify=True,
        workspace=4,        # TRT workspace GiB (reduce to 2 if OOM)
    )

    engine_path = Path(str(export_path))
    print(f"\n[EXPORT] TensorRT engine: {engine_path}")
    print(f"\nTo use in AquaScope on the Jetson:")
    print(f"  scp {engine_path} jetson:/opt/aquascope/models/best.engine")
    print(f"  python app/fish_tracker.py --model /opt/aquascope/models/best.engine")
    return engine_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n → TensorRT FP16 .engine (NVIDIA GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help=f"Roboflow dataset directory containing data.yaml (default: {_DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--distill-data",
        type=Path,
        default=_DEFAULT_DISTILL,
        help=f"Distillation output from generate_labels_dino.py (default: {_DEFAULT_DISTILL})",
    )
    parser.add_argument(
        "--no-distill",
        action="store_true",
        help="Skip distillation pseudo-labels (do not include them in training)",
    )
    parser.add_argument(
        "--user-data",
        type=Path,
        default=_DEFAULT_USERDATA,
        help=f"User-recorded labels from the dashboard (default: {_DEFAULT_USERDATA}). "
             "Silently skipped if the directory is missing or empty.",
    )
    parser.add_argument(
        "--no-user",
        action="store_true",
        help="Skip dataset/user_recorded/ even if it has labeled images",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Train on distillation pseudo-labels only (skip Roboflow dataset)",
    )
    parser.add_argument(
        "--base-model",
        default="training/yolov8n.pt",
        help="Starting weights (default: training/yolov8n.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Training batch size (default: 32; reduce if OOM)",
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
        "--engine-batch",
        type=int,
        default=1,
        help="TensorRT engine batch size (default: 1 for Jetson; use -1 for dynamic)",
    )
    parser.add_argument(
        "--export-only",
        type=Path,
        default=None,
        metavar="WEIGHTS_PT",
        help="Skip training; export an existing .pt to TensorRT .engine",
    )
    args = parser.parse_args()

    args.dataset      = args.dataset.resolve()
    args.distill_data = args.distill_data.resolve()
    args.user_data    = args.user_data.resolve()

    if args.export_only is not None:
        if not args.export_only.exists():
            print(f"ERROR: {args.export_only} not found")
            sys.exit(1)
        export_engine(args.export_only, batch=args.engine_batch)
        return

    # Validate GPU
    import torch
    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU detected. This script requires an NVIDIA GPU.")
        print("       For CPU/MPS training use training/distill_train.py instead.")
        sys.exit(1)
    print(f"[GPU] {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GiB VRAM)")

    # Check data sources
    if not args.no_merge and not (args.dataset / "data.yaml").exists():
        print(f"[WARN] dataset/data.yaml not found at {args.dataset}; using distillation only")
        args.no_merge = True

    distill_yaml = args.distill_data / "data.yaml"
    distill_available = distill_yaml.exists() and not args.no_distill
    if not distill_yaml.exists() and not args.no_distill:
        print(f"[WARN] distillation data.yaml not found at {args.distill_data}; skipping")

    user_train = _user_data_train_path(args.user_data) if not args.no_user else None
    if user_train:
        print(f"[DATA] user_recorded labels found: {user_train}")

    if args.no_merge and not distill_available:
        print("ERROR: --no-merge given but distillation set is unavailable.")
        print("       Either drop --no-merge or run training/generate_labels_dino.py.")
        sys.exit(1)

    tmp_dir = Path(tempfile.mkdtemp(prefix="aquascope_gpu_"))
    try:
        if args.no_merge:
            try:
                data_yaml = build_distill_only_yaml(args.distill_data, tmp_dir)
            except ImportError:
                data_yaml = distill_yaml
        else:
            data_yaml = build_merged_yaml(
                dataset_dir=args.dataset,
                distill_dir=args.distill_data if distill_available else None,
                user_dir=args.user_data if not args.no_user else None,
                tmp_dir=tmp_dir,
            )

        best_pt = train(
            data_yaml=data_yaml,
            base_model=args.base_model,
            epochs=args.epochs,
            batch=args.batch,
            run_name=args.name,
            resume=args.resume,
        )

        export_engine(best_pt, batch=args.engine_batch)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    engine_path = best_pt.parent / "best.engine"
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Weights : {best_pt}")
    print(f"  Engine  : {engine_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
