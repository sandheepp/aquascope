import jetson_compat  # noqa: F401 — must be first to patch torchvision NMS
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(
    data='/home/jetson/projects/aquascope/datasets/aquarium_dataset/data.yaml',
    epochs=50,
    imgsz=640,
    device=0,
    batch=4,        # down from 8
    workers=2,      # down from 8 — each worker holds images in RAM
    amp=False,
    cache=False,    # don't cache dataset in RAM
    optimizer='SGD',
)

# python3 -c "from ultralytics import YOLO; YOLO('/home/jetson/projects/aquascope/runs/detect/train/weights/best.pt').export(format='engine', device=0, half=True, imgsz=640)"
# python3 fish_tracker.py --conf 0.25 --no-display --stream --public --model /home/jetson/projects/aquascope/runs/detect/train/weights/best.engine


# python3 jetson_monitor.py --interval 1 --output training_mem.csv
