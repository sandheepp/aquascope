#!/usr/bin/env python3
"""
Fine-tune YOLOv8n on an aquarium fish dataset for better accuracy.

Two options:
  A) Use Roboflow's pre-labeled Aquarium dataset (recommended to start)
  B) Label your own fish from C920 captures using Roboflow or CVAT

Run this on a machine with a decent GPU (or the Jetson itself for small datasets).
"""

import argparse
import os


def download_roboflow_dataset(api_key: str, output_dir: str = "datasets"):
    """Download the Aquarium Combined dataset from Roboflow."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow: pip3 install roboflow")
        return None

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("brad-dwyer").project("aquarium-combined")
    dataset = project.version(2).download("yolov8", location=output_dir)
    print(f"✓ Dataset downloaded to {output_dir}")
    return dataset


def train(data_yaml: str, epochs: int = 100, imgsz: int = 640, batch: int = 8):
    """Fine-tune YOLOv8n on the fish dataset."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # Start from pretrained COCO weights

    print(f"\n{'='*50}")
    print(f"  Training YOLOv8n on fish dataset")
    print(f"  Epochs: {epochs} | Image Size: {imgsz} | Batch: {batch}")
    print(f"{'='*50}\n")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="fish_detector",
        patience=20,             # Early stopping
        save=True,
        device=0,                # GPU
        workers=2,               # Jetson has limited CPU cores
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        augment=True,
        mosaic=0.5,              # Mosaic augmentation
        mixup=0.1,
        hsv_h=0.015,             # Hue shift — important for varied fish colors
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.3,              # Fish can be at any orientation
        fliplr=0.5,
    )

    # Export best model to TensorRT
    best_model = YOLO("runs/detect/fish_detector/weights/best.pt")
    best_model.export(format="engine", device=0, half=True, imgsz=imgsz)
    print("\n✓ Best model exported to TensorRT engine")
    print("  Copy best.engine to your fish_tracker directory and update --model flag")

    return results


def capture_training_images(output_dir: str = "captures", num_images: int = 200):
    """Capture images from C920 for manual labeling."""
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Capturing images for labeling. Press SPACE to capture, 'q' to quit.")
    print(f"Target: {num_images} images → {output_dir}/")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f"Captured: {count}/{num_images} | SPACE=capture, Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            path = os.path.join(output_dir, f"fish_{count:04d}.jpg")
            cv2.imwrite(path, frame)
            count += 1
            print(f"  Saved {path}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Captured {count} images")
    print(f"  Upload to https://app.roboflow.com for labeling, then download in YOLOv8 format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fish Model Fine-Tuning")
    sub = parser.add_subparsers(dest="command")

    # Download dataset
    dl = sub.add_parser("download", help="Download Roboflow aquarium dataset")
    dl.add_argument("--api-key", required=True, help="Roboflow API key")

    # Train
    tr = sub.add_parser("train", help="Train YOLOv8n on fish data")
    tr.add_argument("--data", required=True, help="Path to data.yaml")
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--batch", type=int, default=8)

    # Capture images
    cap = sub.add_parser("capture", help="Capture images from C920 for labeling")
    cap.add_argument("--num", type=int, default=200)

    args = parser.parse_args()

    if args.command == "download":
        download_roboflow_dataset(args.api_key)
    elif args.command == "train":
        train(args.data, args.epochs, batch=args.batch)
    elif args.command == "capture":
        capture_training_images(num_images=args.num)
    else:
        parser.print_help()
