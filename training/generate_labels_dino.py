"""
Generate YOLO-format pseudo-labels from recordings using a Grounding DINO teacher model.

The 840M DINO model acts as a zero-shot detector — it never saw your fish footage
but can identify fish/jellyfish/etc. by name. Those detections become training labels
for the smaller YOLOv8n student (see distill_train.py).

Input sources (pick one or both):
  --videos   directory of .mp4/.avi recordings  (default: recordings/)
  --images   directory of unlabeled .jpg/.png images

Output (default: distillation/):
  distillation/
    images/     ← sampled frames as .jpg
    labels/     ← YOLO-format .txt files (one per image)
    data.yaml   ← dataset config ready for distill_train.py

Model loading:
  --model accepts either a HuggingFace model ID or a local directory that
  contains config.json + model weights in HuggingFace format.

  Example HF IDs:
    IDEA-Research/grounding-dino-base    (~172 M, fast, good quality)
    IDEA-Research/grounding-dino-tiny    (~172 M, even faster)

  If you have an 840M checkpoint in HuggingFace format (config.json + pytorch_model.bin
  or safetensors), pass the directory path directly:
    --model /path/to/your/dino_840m_hf/

  Raw .pth checkpoints from the original groundingdino repo are NOT directly supported.
  Convert first with:
    python -c "
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    # load with original groundingdino library, save in HF format
    "

Usage:
    python training/generate_labels_dino.py
    python training/generate_labels_dino.py --model IDEA-Research/grounding-dino-base
    python training/generate_labels_dino.py --model /path/to/dino_840m_hf
    python training/generate_labels_dino.py --videos recordings/ --sample-fps 1
    python training/generate_labels_dino.py --images dataset/train/images --out distillation/
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_VIDEOS = _PROJECT_ROOT / "recordings"
_DEFAULT_OUT = _PROJECT_ROOT / "distillation"
_DEFAULT_DATA_YAML = _PROJECT_ROOT / "dataset" / "data.yaml"

# Class names matching the existing dataset — Grounding DINO uses these as text prompts
_DEFAULT_CLASSES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]

# Longer, more descriptive prompts improve zero-shot accuracy
_PROMPT_MAP = {
    "fish":      "fish",
    "jellyfish": "jellyfish",
    "penguin":   "penguin",
    "puffin":    "puffin bird",
    "shark":     "shark",
    "starfish":  "starfish",
    "stingray":  "stingray",
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_grounding_dino(model_id: str, device: str):
    """Load a Grounding DINO model via HuggingFace transformers."""
    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError:
        print("ERROR: transformers not installed.")
        print("  pip install transformers accelerate")
        sys.exit(1)

    print(f"[DINO] Loading model: {model_id}")
    print(f"[DINO] Device: {device}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    import torch
    model = model.to(device)
    model.eval()

    print("[DINO] Model ready")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────

def detect(model, processor, image_bgr: np.ndarray, class_names: list[str],
           conf_threshold: float, device: str) -> list[dict]:
    """
    Run Grounding DINO on one BGR frame.
    Returns list of {"class_id": int, "cx": float, "cy": float, "w": float, "h": float}
    where all coords are normalized [0, 1].
    """
    import torch
    from PIL import Image

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    img_h, img_w = image_bgr.shape[:2]

    # Build text prompt: Grounding DINO expects "class1 . class2 . class3 ."
    prompts = [_PROMPT_MAP.get(c, c) for c in class_names]
    text_prompt = " . ".join(prompts) + " ."

    inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    try:
        # transformers >= 4.38: thresholds accepted as kwargs
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=conf_threshold,
            text_threshold=conf_threshold,
            target_sizes=[(img_h, img_w)],
        )[0]
    except TypeError:
        # Older transformers: apply score threshold manually after post-processing
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            target_sizes=[(img_h, img_w)],
        )[0]
        mask = results["scores"] >= conf_threshold
        results = {
            "boxes":  results["boxes"][mask],
            "scores": results["scores"][mask],
            "labels": [l for l, m in zip(results["labels"], mask.tolist()) if m],
        }

    detections = []
    for box, label_text in zip(results["boxes"].cpu().numpy(),
                                results["labels"]):
        # Map predicted text label back to class index
        label_text = label_text.strip().lower()
        class_id = _match_class(label_text, class_names)
        if class_id is None:
            continue

        x1, y1, x2, y2 = box
        # Clamp to frame bounds
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        if bw > 0 and bh > 0:
            detections.append({"class_id": class_id, "cx": cx, "cy": cy,
                                "w": bw, "h": bh})

    return detections


def _match_class(predicted_label: str, class_names: list[str]) -> int | None:
    """Match Grounding DINO's free-text output back to a class index."""
    # Exact match first
    for i, name in enumerate(class_names):
        if predicted_label == name or predicted_label == _PROMPT_MAP.get(name, name):
            return i
    # Substring match
    for i, name in enumerate(class_names):
        if name in predicted_label or predicted_label in name:
            return i
    return None


# ── Frame extraction from video ───────────────────────────────────────────────

def extract_frames(video_path: Path, out_dir: Path, sample_fps: float,
                   start_idx: int = 0) -> list[tuple[Path, np.ndarray]]:
    """
    Extract frames from a video at *sample_fps* and write them to *out_dir*.
    Returns list of (saved_path, frame_bgr).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {video_path}")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(src_fps / sample_fps))

    frames = []
    frame_idx = 0
    saved_idx = start_idx

    print(f"[VIDEO] {video_path.name}  src={src_fps:.1f}fps  "
          f"total={total_frames}  step={step}  "
          f"→ ~{total_frames // step} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            stem = f"frame_{saved_idx:06d}"
            img_path = out_dir / f"{stem}.jpg"
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames.append((img_path, frame))
            saved_idx += 1
        frame_idx += 1

    cap.release()
    return frames


# ── Image directory loader ────────────────────────────────────────────────────

def load_images_from_dir(src_dir: Path, out_dir: Path,
                         start_idx: int = 0) -> list[tuple[Path, np.ndarray]]:
    """Copy/symlink source images into out_dir for uniform downstream handling."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    src_paths = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in exts)

    frames = []
    for idx, src in enumerate(src_paths):
        dst = out_dir / f"img_{start_idx + idx:06d}{src.suffix}"
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)
        frame = cv2.imread(str(dst))
        if frame is not None:
            frames.append((dst, frame))

    print(f"[IMAGES] {src_dir.name}: loaded {len(frames)} images")
    return frames


# ── Label writer ──────────────────────────────────────────────────────────────

def write_label(label_path: Path, detections: list[dict]) -> None:
    """Write YOLO-format label file. Empty file = no detections (valid for background)."""
    with open(label_path, "w") as f:
        for d in detections:
            f.write(f"{d['class_id']} {d['cx']:.6f} {d['cy']:.6f} "
                    f"{d['w']:.6f} {d['h']:.6f}\n")


# ── Dataset YAML ──────────────────────────────────────────────────────────────

def write_data_yaml(out_dir: Path, class_names: list[str]) -> Path:
    """Write data.yaml for the distillation dataset."""
    yaml_path = out_dir / "data.yaml"
    data = {
        "path": str(out_dir),
        "train": "images",
        "val": "images",   # same split — distill_train.py will split properly
        "nc": len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"[YAML] Written: {yaml_path}")
    return yaml_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels from a Grounding DINO teacher model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="IDEA-Research/grounding-dino-base",
        help="HuggingFace model ID or local HF-format directory "
             "(default: IDEA-Research/grounding-dino-base)",
    )
    parser.add_argument(
        "--videos",
        type=Path,
        default=_DEFAULT_VIDEOS,
        help=f"Directory of .mp4/.avi recordings (default: {_DEFAULT_VIDEOS})",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Directory of unlabeled images (alternative/addition to --videos)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory for distillation dataset (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Frames per second to sample from videos (default: 2.0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold for accepting a pseudo-label (default: 0.30)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=_DEFAULT_CLASSES,
        help="Class names to detect (default: 7 aquarium classes)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device: cuda / mps / cpu (auto-detected if omitted)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap total frames processed (useful for quick test runs)",
    )
    args = parser.parse_args()

    # ── Device auto-detect ────────────────────────────────────────────────────
    if args.device is None:
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"[DEVICE] {args.device}")

    # ── Output dirs ───────────────────────────────────────────────────────────
    images_dir = args.out / "images"
    labels_dir = args.out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect frames ────────────────────────────────────────────────────────
    all_frames: list[tuple[Path, np.ndarray]] = []
    frame_counter = 0

    if args.videos is not None:
        if not args.videos.exists():
            print(f"[WARN] --videos path not found: {args.videos}")
        else:
            video_exts = {".mp4", ".avi", ".mov", ".mkv"}
            video_files = sorted(
                p for p in args.videos.iterdir() if p.suffix.lower() in video_exts
            )
            if not video_files:
                print(f"[WARN] No video files found in {args.videos}")
            for vf in video_files:
                frames = extract_frames(vf, images_dir, args.sample_fps, frame_counter)
                all_frames.extend(frames)
                frame_counter += len(frames)

    if args.images is not None:
        if not args.images.exists():
            print(f"[WARN] --images path not found: {args.images}")
        else:
            frames = load_images_from_dir(args.images, images_dir, frame_counter)
            all_frames.extend(frames)
            frame_counter += len(frames)

    if not all_frames:
        print("ERROR: No frames collected. Check --videos / --images paths.")
        sys.exit(1)

    if args.max_frames is not None:
        all_frames = all_frames[: args.max_frames]
        print(f"[INFO] Capped at {args.max_frames} frames")

    print(f"\n[INFO] Total frames to label: {len(all_frames)}")

    # ── Load teacher model ────────────────────────────────────────────────────
    model, processor = load_grounding_dino(args.model, args.device)

    # ── Label generation ──────────────────────────────────────────────────────
    total_detections = 0
    frames_with_detections = 0

    for i, (img_path, frame_bgr) in enumerate(all_frames):
        label_path = labels_dir / (img_path.stem + ".txt")

        # Skip already-labeled frames (allows resuming interrupted runs)
        if label_path.exists():
            existing = label_path.read_text().strip()
            total_detections += len(existing.splitlines()) if existing else 0
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(all_frames)}] skipped (already labeled)")
            continue

        detections = detect(model, processor, frame_bgr, args.classes,
                            args.conf, args.device)
        write_label(label_path, detections)

        if detections:
            total_detections += len(detections)
            frames_with_detections += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(all_frames):
            print(f"  [{i+1}/{len(all_frames)}]  "
                  f"detections so far: {total_detections}  "
                  f"frames with fish: {frames_with_detections}")

    # ── Write data.yaml ───────────────────────────────────────────────────────
    write_data_yaml(args.out, args.classes)

    print(f"\n{'='*60}")
    print(f"  Pseudo-label generation complete")
    print(f"  Frames labeled : {len(all_frames)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Frames with detections: {frames_with_detections} "
          f"({100*frames_with_detections/max(1,len(all_frames)):.1f}%)")
    print(f"  Output: {args.out}")
    print(f"\nNext step:")
    print(f"  python training/distill_train.py --distill-data {args.out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
