"""
FishTracker — core tracking loop.

Wraps YOLOv8 + ByteTrack inference, draws trails and HUD,
handles recording, streaming, and periodic JSON logging.
"""

import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from camera import init_camera
from config import TRAIL_COLORS
from stream import push_frame, start_public_tunnel, start_stream


class FishTracker:
    """Real-time aquarium fish tracker using YOLOv8 + ByteTrack."""

    def __init__(self, config: dict):
        self.config = config
        self.trails: dict = defaultdict(lambda: deque(maxlen=config["max_trail_length"]))
        self.fish_stats: dict = defaultdict(lambda: {
            "first_seen": None, "last_seen": None,
            "total_distance_px": 0.0, "frame_count": 0,
        })
        self.frame_count = 0
        self.fps_history: deque = deque(maxlen=30)
        self.last_log_time = time.time()

        os.makedirs(config["output_dir"], exist_ok=True)
        self.model = self._load_model()
        self.cap = init_camera(config)
        self.writer = self._init_writer()
        self._start_stream()

    # ── Initialisation helpers ────────────────────────────

    def _load_model(self) -> YOLO:
        path = self.config["model_path"]
        if path.endswith(".engine") and not os.path.exists(path):
            pt = path.replace(".engine", ".pt")
            print(f"[MODEL] Engine not found: {path}")
            print(f"[MODEL] Falling back to: {pt}")
            print(f"[MODEL] Export hint: python3 -c \"from ultralytics import YOLO; "
                  f"YOLO('{pt}').export(format='engine', device=0, half=True, "
                  f"imgsz={self.config['imgsz']})\"")
            path = pt
        print(f"[MODEL] Loading: {path}")
        return YOLO(path)

    def _init_writer(self) -> cv2.VideoWriter | None:
        if not self.config["record"]:
            return None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(
            self.config["record_path"], fourcc,
            self.config["camera_fps"],
            (self.config["camera_width"], self.config["camera_height"]),
        )

    def _start_stream(self) -> None:
        if not self.config["stream"]:
            return
        port = self.config["stream_port"]
        start_stream(port)
        host = os.getenv("JETSON_HOST", "localhost")
        print(f"[STREAM] Local:  http://{host}:{port}/")
        if self.config.get("public"):
            start_public_tunnel(port)

    # ── Drawing ───────────────────────────────────────────

    def _color(self, track_id: int) -> tuple:
        return TRAIL_COLORS[track_id % len(TRAIL_COLORS)]

    def _draw_trails(self, frame: np.ndarray) -> None:
        for track_id, trail in self.trails.items():
            if len(trail) < 2:
                continue
            color = self._color(track_id)
            pts = list(trail)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                cv2.line(frame, pts[i - 1], pts[i], color,
                         max(1, int(3 * alpha)), cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        avg_fps = float(np.mean(self.fps_history)) if self.fps_history else 0.0
        active = sum(
            1 for s in self.fish_stats.values()
            if s["last_seen"] and
            (datetime.now() - datetime.fromisoformat(s["last_seen"])).total_seconds() < 2
        )
        for text, y in [
            (f"FPS: {avg_fps:.1f}", 40),
            (f"Active Fish: {active}", 70),
            (f"Total IDs: {len(self.fish_stats)}", 100),
        ]:
            cv2.putText(frame, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    # ── Tracking state ────────────────────────────────────

    def _update_trail(self, track_id: int, center: tuple) -> None:
        now = datetime.now().isoformat()
        trail = self.trails[track_id]
        stats = self.fish_stats[track_id]
        if trail:
            last = trail[-1]
            stats["total_distance_px"] += np.hypot(
                center[0] - last[0], center[1] - last[1]
            )
        trail.append(center)
        stats["frame_count"] += 1
        stats["last_seen"] = now
        if stats["first_seen"] is None:
            stats["first_seen"] = now

    # ── Logging ───────────────────────────────────────────

    def _log_stats(self) -> None:
        if time.time() - self.last_log_time < self.config["log_interval_sec"]:
            return
        self.last_log_time = time.time()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config["output_dir"], f"fish_stats_{ts}.json")
        with open(path, "w") as f:
            json.dump({
                "timestamp": ts,
                "total_frames": self.frame_count,
                "unique_fish": len(self.fish_stats),
                "fish": {
                    str(tid): {
                        "first_seen": s["first_seen"],
                        "last_seen": s["last_seen"],
                        "total_distance_px": round(s["total_distance_px"], 2),
                        "frame_count": s["frame_count"],
                    }
                    for tid, s in self.fish_stats.items()
                },
            }, f, indent=2)
        print(f"[LOG] Stats → {path}")

    # ── Main loop ─────────────────────────────────────────

    def run(self) -> None:
        print("\n" + "=" * 60)
        print("  AQUARIUM FISH TRACKER — Running")
        print("  Press 'q' to quit | 's' to snapshot | 'r' to reset trails")
        print("=" * 60 + "\n")

        try:
            while True:
                t0 = time.time()
                frame = self._read_frame()
                if frame is None:
                    continue

                self.frame_count += 1
                annotated = self._infer_and_annotate(frame)
                self._draw_trails(annotated)
                self._draw_hud(annotated)

                self.fps_history.append(1.0 / max(time.time() - t0, 1e-6))

                if self.writer:
                    self.writer.write(annotated)

                if self.config["stream"]:
                    _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    push_frame(jpeg.tobytes())

                self._log_stats()
                self._handle_display(annotated)

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            self._cleanup()

    def _read_frame(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            if not self.cap.isOpened():
                print("[CAM] Disconnected — reconnecting...")
                old = self.cap
                self.cap = init_camera(self.config)
                old.release()
            else:
                print("[CAM] Frame capture failed, retrying...")
        return frame if ret else None

    def _infer_and_annotate(self, frame: np.ndarray) -> np.ndarray:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.config["tracker_config"],
            conf=self.config["confidence_threshold"],
            iou=self.config["iou_threshold"],
            imgsz=self.config["imgsz"],
            verbose=False,
            classes=[0],  # COCO class 0 = person; update after fine-tuning on fish
        )

        annotated = frame.copy()
        if not (results and results[0].boxes is not None and results[0].boxes.id is not None):
            return annotated

        boxes = results[0].boxes
        for box, track_id, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color = self._color(track_id)

            self._update_trail(track_id, (cx, cy))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"Fish #{track_id} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 4, color, -1, cv2.LINE_AA)

        return annotated

    def _handle_display(self, frame: np.ndarray) -> None:
        if not self.config["display"]:
            return
        try:
            cv2.imshow("AquaScope", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
            elif key == ord("s"):
                snap = os.path.join(
                    self.config["output_dir"],
                    f"snapshot_{datetime.now().strftime('%H%M%S')}.jpg",
                )
                cv2.imwrite(snap, frame)
                print(f"[SNAP] Saved {snap}")
            elif key == ord("r"):
                self.trails.clear()
                print("[INFO] Trails reset")
        except cv2.error as e:
            print(f"[WARN] Display error: {e} — switching to headless")
            self.config["display"] = False

    def _cleanup(self) -> None:
        self._log_stats()
        if hasattr(self, "cap") and self.cap and self.cap.isOpened():
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Tracker stopped.")
