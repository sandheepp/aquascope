import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import jetson_compat  # noqa: F401 — must be first to patch torchvision NMS
from ultralytics import YOLO

IMGSZ = 640

model = YOLO('yolov8s.pt')

results = model.train(
    data='/home/jetson/projects/aquascope/datasets/aquarium_dataset/data.yaml',
    epochs=50,
    imgsz=IMGSZ,
    device=0,
    batch=2,        # reduced from 4 — prevents GPU/RAM OOM on Jetson
    workers=0,      # 0 = load in main process — each worker holds a full copy of images in RAM
    amp=False,
    cache=False,    # don't cache dataset in RAM
    name="fish_detector",
    patience=20,             # Early stopping
    save=True,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    augment=True,
    mosaic=0.0,              # disabled — mosaic stitches 4 images, 4x memory spike on Jetson
    mixup=0.0,               # disabled — compounds mosaic memory pressure
    hsv_h=0.015,             # Hue shift — important for varied fish colors
    hsv_s=0.5,
    hsv_v=0.3,
    flipud=0.3,              # Fish can be at any orientation
    fliplr=0.5,
)



# Export best model to TensorRT
best_model = YOLO("runs/detect/fish_detector/weights/best.pt")
best_model.export(format="engine", device=0, half=True, imgsz=IMGSZ)
print("\n✓ Best model exported to TensorRT engine")
print("  Copy best.engine to your fish_tracker directory and update --model flag")
