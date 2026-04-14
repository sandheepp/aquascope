"""
FishTracker — core tracking loop.

Wraps YOLOv8s + SAHI sliced inference + supervision ByteTrack,
draws trails and HUD, handles recording, streaming, and periodic JSON logging.
"""

import glob
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
from stream import hat_mode_enabled, push_frame, push_stats, request_reset, start_public_tunnel, start_stream, trails_mode_enabled


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
        self._last_stream_push = 0.0
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._thermal_zones: dict[str, str] = {}   # label → /sys path, built on first call
        self._temps: dict[str, float] = {}          # label → °C, updated each frame

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

    # ── Thermal ───────────────────────────────────────────

    # Priority order: first matching label wins for each slot.
    _TEMP_SLOTS: dict[str, list[str]] = {
        "CPU": ["CPU-therm", "cpu-thermal", "cpu_thermal"],
        "GPU": ["GPU-therm", "gpu-thermal", "gpu_thermal"],
    }

    def _build_thermal_map(self) -> None:
        """Scan /sys/class/thermal once and cache zone paths by label."""
        import glob as _glob
        for zone_type_path in sorted(_glob.glob("/sys/class/thermal/thermal_zone*/type")):
            try:
                zone_label = open(zone_type_path).read().strip()
            except OSError:
                continue
            zone_dir = os.path.dirname(zone_type_path)
            for slot, candidates in self._TEMP_SLOTS.items():
                if slot not in self._thermal_zones and zone_label in candidates:
                    self._thermal_zones[slot] = os.path.join(zone_dir, "temp")

    def _read_jetson_temps(self) -> dict[str, float]:
        """Return {slot: °C} for each mapped thermal zone. Silent on error."""
        if not self._thermal_zones:
            self._build_thermal_map()
        result: dict[str, float] = {}
        for slot, path in self._thermal_zones.items():
            try:
                result[slot] = int(open(path).read().strip()) / 1000.0
            except OSError:
                pass
        return result

    # ── Drawing ───────────────────────────────────────────

    def _draw_hat(self, frame: np.ndarray, x1: int, y1: int, x2: int) -> None:
        """Draw a tiny top hat above the bounding box."""
        w = x2 - x1
        hat_w = max(int(w * 0.7), 14)
        hat_h = max(int(hat_w * 0.6), 8)
        brim_h = max(int(hat_h * 0.3), 3)
        cx = (x1 + x2) // 2

        # Crown
        crown_x1 = cx - hat_w // 2
        crown_y2 = y1 - 2
        crown_y1 = crown_y2 - hat_h
        crown_x2 = cx + hat_w // 2
        cv2.rectangle(frame, (crown_x1, crown_y1), (crown_x2, crown_y2), (30, 30, 30), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (crown_x1, crown_y1), (crown_x2, crown_y2), (200, 180, 50), 1, cv2.LINE_AA)

        # Hat band
        band_y1 = crown_y2 - brim_h - 1
        band_y2 = crown_y2 - 1
        cv2.rectangle(frame, (crown_x1 + 1, band_y1), (crown_x2 - 1, band_y2), (200, 180, 50), -1, cv2.LINE_AA)

        # Brim
        brim_x1 = cx - int(hat_w * 0.65)
        brim_x2 = cx + int(hat_w * 0.65)
        cv2.rectangle(frame, (brim_x1, crown_y2), (brim_x2, crown_y2 + brim_h), (30, 30, 30), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (brim_x1, crown_y2), (brim_x2, crown_y2 + brim_h), (200, 180, 50), 1, cv2.LINE_AA)

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
        h, w = frame.shape[:2]
        now = datetime.now()
        avg_fps = float(np.mean(self.fps_history)) if self.fps_history else 0.0
        active = sum(
            1 for s in self.fish_stats.values()
            if s["last_seen"] and
            (now - datetime.fromisoformat(s["last_seen"])).total_seconds() < 2
        )

        # ── Temperatures ──────────────────────────────────────
        self._temps = self._read_jetson_temps()

        def _temp_color(t: float) -> tuple:
            if t >= 75:
                return (60, 60, 255)    # red — throttle territory
            if t >= 60:
                return (30, 165, 255)   # orange — warm
            return (80, 255, 140)       # green — nominal

        if self._temps:
            temp_parts = "  ".join(
                f"{slot} {t:.0f}°C" for slot, t in sorted(self._temps.items())
            )
            # Use the hottest sensor to pick the row colour
            max_t = max(self._temps.values())
            temp_color = _temp_color(max_t)
        else:
            temp_parts = "Temp  N/A"
            temp_color = (120, 120, 120)

        # ── Top-left panel ────────────────────────────────────
        pad, lh, panel_w = 12, 28, 230
        rows = [
            ("AquaScope", (200, 220, 255), 0.62, 2),
            (f"FPS    {avg_fps:5.1f}", (80, 255, 140), 0.58, 1),
            (f"Active {active:5d}", (80, 220, 255), 0.58, 1),
            (f"Total  {len(self.fish_stats):5d}", (180, 180, 255), 0.58, 1),
            (f"Frame  {self.frame_count:5d}", (120, 120, 120), 0.45, 1),
            (temp_parts, temp_color, 0.45, 1),
        ]
        panel_h = pad * 2 + lh * len(rows)
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        # subtle border
        cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (60, 60, 60), 1, cv2.LINE_AA)

        for i, (text, color, scale, thickness) in enumerate(rows):
            y = 8 + pad + lh * i + int(lh * 0.72)
            cv2.putText(frame, text, (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

        # ── FPS colour bar (green→yellow→red) ─────────────────
        fps_max = 30.0
        bar_x, bar_y, bar_w, bar_h = 8 + panel_w + 6, 8, 8, panel_h
        ratio = min(avg_fps / fps_max, 1.0)
        filled = int(bar_h * ratio)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        bar_color = (
            (0, int(255 * ratio), int(255 * (1 - ratio)))
            if ratio < 0.5
            else (0, 255, 0)
        )
        cv2.rectangle(frame, (bar_x, bar_y + bar_h - filled),
                      (bar_x + bar_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)

        # ── Bottom status bar ─────────────────────────────────
        bar_h2 = 26
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - bar_h2), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
        ts = now.strftime("%Y-%m-%d  %H:%M:%S")
        model_name = self.config["model_path"].split("/")[-1]
        sahi_tag = "  [SAHI]" if self.config.get("sahi") else ""
        status = f"  {ts}    {model_name}{sahi_tag}"
        cv2.putText(frame, status, (8, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
        # right-align keys hint
        hint = "Q quit   S snap   R reset"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(frame, hint, (w - hw - 10, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

    # ── Reset ─────────────────────────────────────────────

    def _reset(self) -> None:
        self.trails.clear()
        self.fish_stats.clear()
        self.sv_tracker = sv.ByteTrack()
        self.frame_count = 0
        print("[INFO] Reset — trails, stats, and tracker cleared")

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

    # ── Stats for stream dashboard ────────────────────────

    def _build_stats(self) -> dict:
        now = datetime.now()
        avg_fps = float(np.mean(self.fps_history)) if self.fps_history else 0.0
        active = sum(
            1 for s in self.fish_stats.values()
            if s["last_seen"] and
            (now - datetime.fromisoformat(s["last_seen"])).total_seconds() < 2
        )
        return {
            "fps": round(avg_fps, 1),
            "active": active,
            "total_ids": len(self.fish_stats),
            "frame": self.frame_count,
            "model": self.config["model_path"].split("/")[-1],
            "sahi": bool(self.config.get("sahi")),
            "temps_c": {slot: round(t, 1) for slot, t in self._temps.items()},
            "fish": {
                str(tid): {
                    "first_seen": s["first_seen"],
                    "last_seen": s["last_seen"],
                    "last_seen_ts": datetime.fromisoformat(s["last_seen"]).timestamp()
                        if s["last_seen"] else 0,
                    "total_distance_px": round(s["total_distance_px"], 1),
                    "frame_count": s["frame_count"],
                }
                for tid, s in self.fish_stats.items()
            },
        }

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
                if request_reset():
                    self._reset()

                t0 = time.time()
                raw = self._read_frame()
                if raw is None:
                    continue

                # Infer on the raw frame so the model sees unprocessed pixels
                self.frame_count += 1
                annotated = self._infer_and_annotate(raw)
                if trails_mode_enabled():
                    self._draw_trails(annotated)

                # Enhance only the display/stream output
                annotated = self._enhance_frame(annotated)

                if self.config["display"]:
                    self._draw_hud(annotated)

                self.fps_history.append(1.0 / max(time.time() - t0, 1e-6))

                if self.writer:
                    self.writer.write(annotated)

                if self.config["stream"]:
                    now_t = time.time()
                    stream_interval = 1.0 / self.config.get("stream_fps", 20)
                    if now_t - self._last_stream_push >= stream_interval:
                        _, jpeg = cv2.imencode(".jpg", annotated,
                                              [cv2.IMWRITE_JPEG_QUALITY, self.config.get("stream_quality", 75)])
                        push_frame(jpeg.tobytes())
                        self._last_stream_push = now_t
                    if self.frame_count % 5 == 0:
                        push_stats(self._build_stats())

                self._log_stats()
                self._handle_display(annotated)

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            self._cleanup()

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance contrast, colour vibrancy, and sharpness — single LAB pass."""
        # 1. CLAHE on L + saturation boost on a/b — one colour-space round-trip
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        # a/b channels are stored as uint8 centred at 128; scale deviation for vibrancy
        ab = lab[:, :, 1:].astype(np.float32)
        lab[:, :, 1:] = np.clip((ab - 128) * 1.4 + 128, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2. Sharpening kernel — single filter pass, no Gaussian+blend overhead
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        frame = cv2.filter2D(frame, -1, kernel)

        return frame

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
            if hat_mode_enabled():
                self._draw_hat(annotated, x1, y1, x2)

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
                self._reset()
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
        self._delete_snapshots()
        print("[INFO] Tracker stopped.")

    def _delete_snapshots(self) -> None:
        snap_dir = os.path.join(self.config["output_dir"], "screenshots")
        files = glob.glob(os.path.join(snap_dir, "snap_*.jpg"))
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
        if files:
            print(f"[INFO] Deleted {len(files)} snapshot(s) from {snap_dir}")
