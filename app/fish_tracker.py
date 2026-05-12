#!/usr/bin/env python3
"""
AquaScope — CLI entry point.

Jetson Orin Nano + Logitech C920 + YOLOv8s + supervision ByteTrack.

Streaming is on by default; pass --no-stream to disable.

Usage:
    python3 fish_tracker.py                                    # local display + browser stream
    python3 fish_tracker.py --no-display                       # headless + browser stream
    python3 fish_tracker.py --no-display --public              # + public URL
    python3 fish_tracker.py --no-stream                        # local display only, no MJPEG
    python3 fish_tracker.py --model models/best.pt             # override default models/best.engine
    python3 fish_tracker.py --sahi                             # sliced inference (lower FPS)
    python3 fish_tracker.py --sahi --resolution 720p           # SAHI at 4 tiles (~2× faster)
    python3 fish_tracker.py --exposure -6                      # manual exposure
"""

# jetson_compat MUST be imported before torch / torchvision / ultralytics
import jetson_compat  # noqa: F401

import argparse
import os
import warnings

# Upstream noise: Ultralytics + supervision occasionally hit `0/0` and
# `nan * x` inside NMS/IoU on edge-case (empty / degenerate) detection
# sets — visible at the terminal as scalar-divide / invalid-value
# RuntimeWarnings. They're benign (downstream code already guards on
# empty xyxy), so silence the exact two patterns rather than the whole
# numpy RuntimeWarning category.
warnings.filterwarnings(
    "ignore",
    message=r"divide by zero encountered in scalar",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"invalid value encountered in scalar",
    category=RuntimeWarning,
)

from dotenv import load_dotenv
load_dotenv()

from config import DEFAULT_CONFIG
from tracker import FishTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AquaScope — real-time aquarium fish tracker")
    p.add_argument("--camera", type=int, default=0, help="Camera device ID (default 0)")
    p.add_argument("--video", default=None,
                   help="Path to a video file to use instead of a webcam — handy for "
                        "demoing the dashboard on a laptop without an aquarium camera.")
    p.add_argument("--resolution", default="720p", choices=["480p", "720p", "1080p"],
                   help="Capture resolution (default 1080p; ignored when --video is set)")
    p.add_argument("--model", default="models/best.engine",
                   help="YOLOv8 model path (.pt or .engine). Falls back to yolov8n.pt "
                        "if the default path doesn't exist (laptop / fresh-clone friendly).")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--exposure", type=int, default=None,
                   help="Manual V4L2 exposure (e.g. -6). Omit for auto.")
    p.add_argument("--record", action="store_true", help="Record annotated video to file")
    p.add_argument("--no-display", action="store_true", help="Headless mode (no GUI window)")
    p.add_argument("--no-stream", action="store_true",
                   help="Disable browser MJPEG stream (on by default)")
    p.add_argument("--stream-port", type=int, default=8080, help="MJPEG server port (default 8080)")
    p.add_argument("--public", action="store_true",
                   help="Expose stream publicly via Cloudflare tunnel")
    p.add_argument("--sahi", action="store_true",
                   help="Enable SAHI sliced inference (better small-fish recall, lower FPS)")
    p.add_argument("--stream-quality", type=int, default=75, metavar="Q",
                   help="JPEG stream quality 1-100 (default 75; lower = faster, higher = sharper)")
    p.add_argument("--stream-fps", type=int, default=20, metavar="FPS",
                   help="Max stream push rate in fps (default 20; caps JPEG encode overhead)")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = DEFAULT_CONFIG.copy()
    config["camera_id"] = args.camera
    config["video_source"] = args.video
    config["camera_width"], config["camera_height"] = {
        "480p": (640, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
    }[args.resolution]
    config["model_path"] = args.model
    config["confidence_threshold"] = args.conf
    config["imgsz"] = args.imgsz
    config["exposure"] = args.exposure
    config["record"] = args.record
    config["display"] = not args.no_display
    config["stream"] = not args.no_stream
    config["stream_port"] = args.stream_port
    config["public"] = args.public
    config["sahi"] = args.sahi
    config["stream_quality"] = args.stream_quality
    config["stream_fps"] = args.stream_fps

    if config["display"] and not os.environ.get("DISPLAY"):
        print("[INFO] No DISPLAY — switching to headless mode (use --no-display to suppress)")
        config["display"] = False

    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    FishTracker(config).run()


if __name__ == "__main__":
    main()
