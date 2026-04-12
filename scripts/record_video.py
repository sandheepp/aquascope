#!/usr/bin/env python3

# --- Configuration ---
DURATION   = 30       # seconds
FPS        = 30       # C920 max: 30fps at 1080p
WIDTH      = 1920     # C920 max resolution
HEIGHT     = 1080
CAMERA_IDX = 0
# ---------------------

import cv2
import time
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output = f"recording_{timestamp}.mp4"

cap = cv2.VideoCapture(CAMERA_IDX)

# Force MJPEG capture format — C920 outputs MJPEG natively at full res,
# avoiding USB bandwidth limits that degrade YUY2 at 1080p.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

# Enable auto-exposure and autofocus
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)   # 3 = aperture priority auto (V4L2)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)       # enable autofocus

actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera opened at {actual_w}x{actual_h} @ {actual_fps}fps")

# Use mp4v (H.264-compatible container); swap for avc1 if your OpenCV build supports it
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output, fourcc, FPS, (actual_w, actual_h))

print(f"Recording {DURATION}s -> {output}")

start = time.time()
while (time.time() - start) < DURATION:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed — check CAMERA_IDX")
        break
    writer.write(frame)

cap.release()
writer.release()
print(f"Saved: {output}")
