"""
Dashboard 'Train Model' subprocess. Cross-platform.

Spawned by app/stream.py. Writes a JSON status file after each epoch so the
dashboard can render progress.

Trains YOLOv8s on dataset/user_recorded/ ONLY (90/10 train/val split) with the
dashboard's two classes (fish, shrimp). Defaults are Jetson-Orin-Nano-friendly
(batch=2, AMP, mosaic/mixup off) — they're conservative enough that the same
script trains fine on a laptop CPU, an Apple Silicon Mac (MPS), or a discrete
NVIDIA GPU.

On CUDA hardware the trained weights are also exported to TensorRT FP16
(models/best_v<N>.engine). On MPS/CPU the export step is skipped and the
versioned .pt becomes the new "current" model — the dashboard dropdown picks
up either format.

Filename kept as train_jetson.py for back-compat with existing scripts; nothing
about the implementation is Jetson-only any more.

Usage:
    python training/train_jetson.py
    python training/train_jetson.py --epochs 30 --batch 4 --status-file /tmp/x.json
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_USERDATA = _PROJECT_ROOT / "dataset" / "user_recorded"


def _detect_device() -> str:
    """Return the best available device: 'cuda', 'mps', or 'cpu'."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


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
                shutil.copy2(img, link)
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


# Dashboard-driven training is locked to the two classes the user actually
# tracks; the labeling tab never produces anything else, so there's no
# benefit to mixing in the original 7-class Roboflow set.
_USER_CLASSES: list[str] = ["fish", "shrimp"]


def _write_status(path: str | None, **kwargs) -> None:
    if not path:
        return
    try:
        # Atomic-ish: write to tmp then rename
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(kwargs, f)
        os.replace(tmp, path)
    except OSError:
        pass


def _next_version(models_dir: Path) -> int:
    """Highest existing engine version + 1, looking at both naming styles
    so the version counter doesn't collide with old engines on disk:
      - new: best_v<N>.engine   (preferred — .engine extension makes it loadable)
      - old: best.engine_v<N>   (legacy)
    """
    import re
    nums: list[int] = []
    new_pat = re.compile(r"^best_v(\d+)\.engine$")
    old_pat = re.compile(r"^best\.engine_v(\d+)$")
    for p in models_dir.iterdir():
        m = new_pat.match(p.name) or old_pat.match(p.name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) if nums else 0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="AquaScope dashboard-triggered training (Jetson)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Total epochs (default: 30; reduce for faster iteration)")
    parser.add_argument("--batch", type=int, default=2,
                        help="Batch size (default: 2 for 8GB Orin Nano unified memory)")
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Train resolution (default: 512; lower → less RAM)")
    parser.add_argument("--export-imgsz", type=int, default=640,
                        help="Engine export resolution — must match the inference imgsz "
                             "(app/config.py imgsz, default 640). Decoupled from --imgsz so "
                             "we can train cheap and still serve full-res inference.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Dataloader workers (default: 0 — avoids RAM duplication on Jetson)")
    parser.add_argument("--mosaic", type=float, default=0.0,
                        help="Mosaic aug prob (default: 0.0 on Jetson — mosaic 4x's per-sample RAM)")
    parser.add_argument("--mixup", type=float, default=0.0,
                        help="MixUp aug prob (default: 0.0 on Jetson)")
    parser.add_argument("--base-model", default="models/best.pt",
                        help="Starting weights — set to current best.pt to fine-tune")
    parser.add_argument("--name", default="fish_train_jetson",
                        help="Run name under runs/detect/")
    parser.add_argument("--status-file", default="",
                        help="JSON file to write progress to (used by the dashboard)")
    args = parser.parse_args()

    models_dir = _PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    version = _next_version(models_dir)

    _write_status(args.status_file,
                  state="starting", version=version,
                  current_epoch=0, total_epochs=args.epochs,
                  message="Setting up data...")

    tmp_dir = Path(tempfile.mkdtemp(prefix="aq_train_"))
    try:
        # Train ONLY on user-recorded labels (90/10 split), with class names
        # fixed to the two classes the dashboard tracks. Roboflow + distillation
        # data is intentionally skipped — the dashboard's labeling tab is the
        # only label source for fish/shrimp, and mixing in 7-class Roboflow
        # data would just inject noise from classes the user doesn't care about.
        data_yaml = build_user_only_yaml(
            user_dir=_DEFAULT_USERDATA,
            tmp_dir=tmp_dir,
            class_names=_USER_CLASSES,
        )

        import torch  # noqa: E402
        device = _detect_device()
        print(f"[TRAIN] Device: {device}")
        if device == "cpu":
            print("[TRAIN] WARNING: training on CPU — expect 5–20× slower epochs "
                  "than a GPU. Use a small dataset / few epochs while you iterate.")

        from ultralytics import YOLO  # noqa: E402

        # Resolve base model: fall back to yolov8s.pt if best.pt doesn't exist yet.
        base_model = args.base_model
        if not Path(base_model).exists():
            print(f"[INFO] {base_model} not found; falling back to yolov8s.pt")
            base_model = "yolov8s.pt"

        model = YOLO(base_model)
        start_t = time.time()
        epoch_start_t = start_t
        epoch_durations: list[float] = []

        def on_train_epoch_start(trainer):  # noqa: ARG001
            nonlocal epoch_start_t
            epoch_start_t = time.time()

        def on_train_epoch_end(trainer):
            nonlocal epoch_start_t
            current = trainer.epoch + 1
            epoch_durations.append(time.time() - epoch_start_t)
            avg = sum(epoch_durations) / len(epoch_durations)
            remaining = max(0, args.epochs - current)
            eta = int(avg * remaining)
            _write_status(args.status_file,
                          state="training", version=version,
                          current_epoch=current, total_epochs=args.epochs,
                          elapsed_sec=int(time.time() - start_t),
                          eta_sec=eta,
                          avg_epoch_sec=int(avg),
                          message=f"Epoch {current}/{args.epochs}")

        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        # AMP only on CUDA: MPS doesn't support it, and CPU training is already
        # slow enough that mixed precision would mostly add overhead.
        use_amp = (device == "cuda")
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=device,
            batch=args.batch,
            workers=args.workers,
            amp=use_amp,
            cache=False,
            project=str(_PROJECT_ROOT / "runs" / "detect"),
            name=args.name,
            exist_ok=True,
            patience=10,
            save=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            mosaic=args.mosaic,
            mixup=args.mixup,
            close_mosaic=0,
            rect=True,                 # rectangular batching — lower padding RAM
        )

        _write_status(args.status_file,
                      state="exporting", version=version,
                      current_epoch=args.epochs, total_epochs=args.epochs,
                      elapsed_sec=int(time.time() - start_t),
                      message=("Exporting TensorRT engine..."
                              if device == "cuda" else
                              "Saving versioned weights..."))

        run_dir = _PROJECT_ROOT / "runs" / "detect" / args.name
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            _write_status(args.status_file, state="failed", version=version,
                          message=f"Training finished but best.pt missing: {best_pt}")
            sys.exit(1)

        # Always keep a versioned .pt — it's portable across machines and the
        # dashboard will list it in the model dropdown even if no engine exists.
        versioned_pt = models_dir / f"best_v{version}.pt"
        shutil.copy2(best_pt, versioned_pt)

        # TensorRT export is CUDA-only (NVIDIA stack). On MPS/CPU the .pt is the
        # final artifact; the dashboard will pick it up and inference will run on
        # whatever device load_model() detects at next reload.
        out_engine: Path | None = None
        if device == "cuda":
            export_model = YOLO(str(best_pt))
            export_path = export_model.export(
                format="engine",
                imgsz=args.export_imgsz,
                half=True,
                device=0,
                batch=1,
                simplify=True,
                workspace=2,           # GiB; conservative for Orin Nano
            )
            out_engine = models_dir / f"best_v{version}.engine"
            try:
                shutil.move(str(export_path), str(out_engine))
            except OSError:
                shutil.copy2(str(export_path), str(out_engine))

        # Only wipe the user-recorded set if a versioned artifact actually
        # landed on disk. On any failure path above, the data stays put so the
        # next run can re-train against it.
        artifact = out_engine if out_engine and out_engine.exists() else versioned_pt
        if artifact.exists() and artifact.stat().st_size > 0:
            for sub in ("images", "labels"):
                target = _DEFAULT_USERDATA / sub
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
            for stale in ("labels.cache",):
                p = _DEFAULT_USERDATA / stale
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            print(f"[CLEANUP] Wiped {_DEFAULT_USERDATA}/images and /labels "
                  f"after successful save of {artifact.name}.")

        _write_status(args.status_file,
                      state="done", version=version,
                      engine_path=str(out_engine) if out_engine else "",
                      pt_path=str(versioned_pt),
                      elapsed_sec=int(time.time() - start_t),
                      message=f"Saved {artifact.name}")
    except Exception as e:                                  # noqa: BLE001
        _write_status(args.status_file,
                      state="failed",
                      message=f"{type(e).__name__}: {e}",
                      trace=traceback.format_exc()[-2000:])
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
