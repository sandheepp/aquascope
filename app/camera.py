"""
Camera initialisation for Logitech C920 on Jetson via V4L2.
"""

import time
import cv2


def init_camera(config: dict) -> cv2.VideoCapture:
    """
    Open the camera with Jetson-friendly V4L2 settings.
    Retries indefinitely if the camera is not available.
    """
    cap = cv2.VideoCapture(config["camera_id"], cv2.CAP_V4L2)

    # MJPG capture + minimal buffer for low latency
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera_height"])
    cap.set(cv2.CAP_PROP_FPS, config["camera_fps"])

    # Autofocus enabled
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    # Exposure (V4L2: 3=aperture-priority auto, 1=manual)
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
        return init_camera(config)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAM] Opened: {w}x{h} @ {fps:.0f} FPS")
    return cap
