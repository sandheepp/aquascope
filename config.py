"""
Default configuration and constants for AquaScope.
"""

DEFAULT_CONFIG: dict = {
    "camera_id": 0,                  # /dev/video0 for C920
    "camera_width": 1920,
    "camera_height": 1080,
    "camera_fps": 60,
    "model_path": "yolov8s.pt",      # Auto-falls back to .pt if .engine missing
    "confidence_threshold": 0.35,
    "iou_threshold": 0.45,
    "imgsz": 640,
    "max_trail_length": 60,          # Trail length in frames
    "log_interval_sec": 60,          # Save JSON stats every N seconds
    "output_dir": "fish_logs",
    "display": True,
    "exposure": None,                # None = auto; int = manual V4L2 value (e.g. -6)
    "record": False,
    "record_path": "fish_recording.mp4",
    "stream": False,
    "stream_port": 8080,
    "public": False,
    # SAHI — sliced inference for small fish detection (off by default for FPS)
    # At 1920×1080 with 640px tiles: 0% overlap→6 tiles, 10%→8 tiles, 20%→12 tiles
    # Tip: use --resolution 720p to halve tile count (4 tiles at 0% overlap)
    "sahi": False,                   # Enable via --sahi flag
    "sahi_slice_height": 640,        # Tile height (px); match imgsz
    "sahi_slice_width": 640,         # Tile width (px); match imgsz
    "sahi_overlap_ratio": 0.1,       # 10% overlap — balance coverage vs tile count
}

# Color palette for fish trails — up to 20 unique fish get distinct colors
TRAIL_COLORS: list[tuple[int, int, int]] = [
    (255, 107, 107), (78, 205, 196),  (69, 183, 209),  (150, 206, 180),
    (255, 238, 173), (255, 154, 162), (199, 206, 234),  (254, 200, 154),
    (255, 183, 178), (181, 234, 215), (224, 187, 228),  (149, 225, 211),
    (253, 253, 150), (174, 198, 207), (179, 158, 181),  (255, 218, 185),
    (119, 221, 231), (203, 153, 201), (162, 217, 206),  (255, 179, 186),
]
