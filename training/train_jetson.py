"""
Jetson-side training launcher for the dashboard 'Train Model' button.

Designed to be spawned as a subprocess by app/stream.py. Writes a JSON status
file after each epoch so the dashboard can render progress.

Trains YOLOv8s on dataset/ + dataset/distillation/ + dataset/user_recorded/ with
Jetson-Orin-Nano-friendly defaults (batch=4, AMP, fewer epochs by default), then
exports the best weights to TensorRT FP16 as models/best.engine_v<N> with N
auto-incremented from existing engines under models/.

Usage:
    python training/train_jetson.py
    python training/train_jetson.py --epochs 30 --batch 4 --status-file /tmp/x.json
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Reuse the merge logic from train_gpu.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gpu import (  # noqa: E402
    _DEFAULT_DATASET,
    _DEFAULT_DISTILL,
    _DEFAULT_USERDATA,
    _PROJECT_ROOT,
    build_merged_yaml,
)


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
    nums = []
    for p in models_dir.glob("best.engine_v*"):
        suffix = p.name.replace("best.engine_v", "")
        try:
            nums.append(int(suffix))
        except ValueError:
            pass
    return (max(nums) if nums else 0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="AquaScope dashboard-triggered training (Jetson)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Total epochs (default: 30; reduce for faster iteration)")
    parser.add_argument("--batch", type=int, default=2,
                        help="Batch size (default: 2 for 8GB Orin Nano unified memory)")
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Train resolution (default: 512; lower → less RAM)")
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

    tmp_dir = Path(tempfile.mkdtemp(prefix="aq_jetson_"))
    try:
        # Build merged dataset (Roboflow + distillation + user_recorded if present).
        distill_yaml = _DEFAULT_DISTILL / "data.yaml"
        data_yaml = build_merged_yaml(
            dataset_dir=_DEFAULT_DATASET,
            distill_dir=_DEFAULT_DISTILL if distill_yaml.exists() else None,
            user_dir=_DEFAULT_USERDATA,
            tmp_dir=tmp_dir,
        )

        import torch  # noqa: E402
        if not torch.cuda.is_available():
            _write_status(args.status_file, state="failed", version=version,
                          message="No CUDA GPU detected. Are you on the Jetson with JetPack?")
            sys.exit(1)

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

        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=0,
            batch=args.batch,
            workers=args.workers,
            amp=True,                  # FP16 — saves memory
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
                      message="Exporting TensorRT engine...")

        run_dir = _PROJECT_ROOT / "runs" / "detect" / args.name
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            _write_status(args.status_file, state="failed", version=version,
                          message=f"Training finished but best.pt missing: {best_pt}")
            sys.exit(1)

        # Keep a versioned .pt copy alongside the engine for safekeeping.
        versioned_pt = models_dir / f"best_v{version}.pt"
        shutil.copy2(best_pt, versioned_pt)

        # Export to TensorRT FP16, then move to versioned name.
        export_model = YOLO(str(best_pt))
        export_path = export_model.export(
            format="engine",
            imgsz=args.imgsz,
            half=True,
            device=0,
            batch=1,
            simplify=True,
            workspace=2,           # GiB; conservative for Orin Nano
        )
        out_engine = models_dir / f"best.engine_v{version}"
        try:
            shutil.move(str(export_path), str(out_engine))
        except OSError:
            shutil.copy2(str(export_path), str(out_engine))

        _write_status(args.status_file,
                      state="done", version=version,
                      engine_path=str(out_engine),
                      pt_path=str(versioned_pt),
                      elapsed_sec=int(time.time() - start_t),
                      message=f"Saved {out_engine.name}")
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
