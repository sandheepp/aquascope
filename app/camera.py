"""
Camera / video-source initialisation.

Two sources are supported, picked from `config["video_source"]` (a path) or
`config["camera_id"]` (an int device index, default 0):

  - Webcam: cross-platform — V4L2 on Linux (with the MJPG/buffer tweaks the
    Jetson C920 needs), AVFoundation on macOS, MSMF on Windows.
  - Video file: any container OpenCV can decode. Loops at EOF so demos play
    indefinitely (`tracker.py` honours the `is_video_file` flag in config).

Returns a `cv2.VideoCapture` handle. `init_camera` mutates `config` to set
`is_video_file = True` when a file path was passed, so the tracker loop can
loop the file instead of trying to reconnect a missing webcam.
"""

import os
import platform
import sys
import time

import cv2


def _backend_for_webcam() -> int:
    system = platform.system()
    if system == "Linux":
        return cv2.CAP_V4L2
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Windows":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def _open_video_file(path: str, config: dict) -> cv2.VideoCapture:
    if not os.path.exists(path):
        print(f"[VIDEO] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[VIDEO] Could not open: {path}", file=sys.stderr)
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    config["camera_width"] = w
    config["camera_height"] = h
    config["is_video_file"] = True
    print(f"[VIDEO] Opened file: {path}  ({w}x{h} @ {fps:.0f} FPS) — will loop at EOF")
    return cap


def _open_webcam(config: dict) -> cv2.VideoCapture:
    backend = _backend_for_webcam()
    cap = cv2.VideoCapture(config["camera_id"], backend)
    is_linux = platform.system() == "Linux"

    # V4L2 on Linux: request MJPG to keep CPU off the colour conversion path,
    # and a single-frame driver buffer so we don't get stale frames after a
    # slow inference cycle. These calls are no-ops (or mild errors) on macOS /
    # Windows backends, so they're gated to Linux.
    if is_linux:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera_height"])
    cap.set(cv2.CAP_PROP_FPS, config["camera_fps"])
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    # V4L2 exposure quirks: 1=manual, 3=aperture-priority auto. Outside V4L2
    # those magic ints don't apply, so we leave the OS default in place.
    if is_linux:
        exp = config.get("exposure")
        if exp is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, exp)
            print(f"[CAM] Exposure: manual ({exp})")
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            print("[CAM] Exposure: auto")

    if not cap.isOpened():
        print("[CAM] Not available, retrying in 5 s...")
        time.sleep(5)
        return _open_webcam(config)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAM] Opened webcam {config['camera_id']} on {platform.system()}: "
          f"{w}x{h} @ {fps:.0f} FPS")
    return cap


def init_camera(config: dict) -> cv2.VideoCapture:
    """Open a webcam or a video file (if `config['video_source']` is a path)."""
    src = config.get("video_source")
    if src:
        return _open_video_file(src, config)
    return _open_webcam(config)
