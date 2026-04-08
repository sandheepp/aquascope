"""
FishTracker — core tracking loop.

Wraps YOLOv8s + SAHI sliced inference + supervision ByteTrack,
draws trails and HUD, handles recording, streaming, and periodic JSON logging.
"""

import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from camera import init_camera
from config import TRAIL_COLORS
from stream import push_frame, start_public_tunnel, start_stream


class FishTracker:
    """Real-time aquarium fish tracker using YOLOv8s + SAHI + supervision ByteTrack."""

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
        self.sv_tracker = sv.ByteTrack()
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
        print(f"[MODEL] Loading: {path} (SAHI sliced inference)")
        return YOLO(path, task="detect")

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
                    _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.config.get("stream_quality", 85)])
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

    def _slice_tiles(self, frame: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
        """Slice frame into overlapping tiles. Returns (tile, x_offset, y_offset)."""
        h, w = frame.shape[:2]
        sh = self.config["sahi_slice_height"]
        sw = self.config["sahi_slice_width"]
        stride_h = int(sh * (1 - self.config["sahi_overlap_ratio"]))
        stride_w = int(sw * (1 - self.config["sahi_overlap_ratio"]))
        tiles = []
        y = 0
        while True:
            y2 = min(y + sh, h)
            y1 = max(y2 - sh, 0)
            x = 0
            while True:
                x2 = min(x + sw, w)
                x1 = max(x2 - sw, 0)
                tiles.append((frame[y1:y2, x1:x2], x1, y1))
                if x2 == w:
                    break
                x += stride_w
            if y2 == h:
                break
            y += stride_h
        return tiles

    def _infer_and_annotate(self, frame: np.ndarray) -> np.ndarray:
        if self.config.get("sahi"):
            # ── SAHI: run predict on each tile, shift coords to frame space ──
            all_xyxy, all_confs = [], []
            for tile, x_off, y_off in self._slice_tiles(frame):
                results = self.model.predict(
                    source=tile,
                    conf=self.config["confidence_threshold"],
                    iou=self.config["iou_threshold"],
                    imgsz=self.config["imgsz"],
                    verbose=False,
                    classes=self.config.get("detect_classes"),  # None = all classes
                    device=0,
                )
                if not (results and results[0].boxes is not None and len(results[0].boxes)):
                    continue
                boxes = results[0].boxes.xyxy.cpu().numpy().copy()
                boxes[:, [0, 2]] += x_off
                boxes[:, [1, 3]] += y_off
                all_xyxy.append(boxes)
                all_confs.append(results[0].boxes.conf.cpu().numpy())

            if all_xyxy:
                xyxy = np.concatenate(all_xyxy, axis=0).astype(np.float32)
                confs = np.concatenate(all_confs, axis=0).astype(np.float32)
            else:
                xyxy = np.empty((0, 4), dtype=np.float32)
                confs = np.empty(0, dtype=np.float32)

            detections = sv.Detections(xyxy=xyxy, confidence=confs)
            detections = detections.with_nms(threshold=self.config["iou_threshold"], class_agnostic=True)
        else:
            # ── Full-frame inference (fast path) ──────────────────────
            results = self.model.predict(
                source=frame,
                conf=self.config["confidence_threshold"],
                iou=self.config["iou_threshold"],
                imgsz=self.config["imgsz"],
                verbose=False,
                classes=[0],  # COCO class 0 = person; update after fine-tuning on fish
                device=0,
            )
            if results and results[0].boxes is not None and len(results[0].boxes):
                xyxy = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
                confs = results[0].boxes.conf.cpu().numpy().astype(np.float32)
            else:
                xyxy = np.empty((0, 4), dtype=np.float32)
                confs = np.empty(0, dtype=np.float32)
            detections = sv.Detections(xyxy=xyxy, confidence=confs)

        tracked = self.sv_tracker.update_with_detections(detections)

        # ── Draw ─────────────────────────────────────────────
        annotated = frame.copy()
        for i in range(len(tracked)):
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            track_id = int(tracked.tracker_id[i])
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
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
