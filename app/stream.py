"""
MJPEG HTTP streaming server + Cloudflare public tunnel.

Endpoints:
  /                  — dashboard UI
  /stream            — MJPEG video stream
  /stats             — JSON live stats
  /reset             — reset tracker trails
  /screenshot        — save current frame, return JSON {filename}
  /screenshots/<f>   — serve saved screenshot
  /models            — JSON list of model files in the models dir
  /model             — set the active model (?v=<basename>)
  /label/state       — labeling enabled flag + queue depth
  /label/toggle      — toggle (or set ?v=0|1) labeling capture
  /label/queue       — JSON list of pending candidates
  /label/image/<id>  — JPEG of a pending candidate frame
  /label/decision    — accept/reject a candidate (?id=<id>&keep=0|1)
  /train/labels      — current user-recorded label count + min required + ETA
  /train/start       — spawn training subprocess (returns 409 if already running)
  /train/status      — poll training progress (state/epoch/eta/message/version)
  /train/cancel      — terminate the running training subprocess
  /train/log         — tail of training subprocess stdout/stderr (text)
  /train/acknowledge — user dismissed the training modal; resume inference
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote

# ── Shared state ──────────────────────────────────────────
_frame: bytes = b""
_lock = threading.Lock()

_stats: dict = {}
_stats_lock = threading.Lock()

_reset_flag = False
_reset_lock = threading.Lock()

_hat_mode = False
_hat_lock = threading.Lock()

_trails_enabled = False
_trails_lock = threading.Lock()

_enhance_enabled = True
_enhance_lock = threading.Lock()

_conf_threshold = 0.35
_conf_lock = threading.Lock()

_resolution = "1080p"
_resolution_lock = threading.Lock()

_models_dir: str = "models"
_model_path: str = "models/best.engine"
_model_lock = threading.Lock()
_MODEL_EXTS = (".pt", ".engine", ".onnx")

# ── Labeling tab state ────────────────────────────────────
import uuid

_label_enabled: bool = False
_label_lock = threading.Lock()
_label_pending: list[dict] = []                # candidates awaiting decision
_label_pending_dir: str = "/tmp/aquascope_label_pending"
_label_output_dir: str = "dataset/user_recorded"
_label_class_names: list[str] = []             # populated from model.names at startup
_LABEL_THROTTLE_SEC = 2.0                      # min seconds between captures of same track_id
_LABEL_QUEUE_MAX = 50                          # cap pending queue to avoid disk blow-up
_label_last_capture: dict[int, float] = {}     # track_id → last capture ts

# ── Training state ────────────────────────────────────────
_train_proc = None                             # subprocess.Popen | None
_train_lock = threading.Lock()
_train_status_file: str = "/tmp/aquascope_train_status.json"
_train_log_file: str = "/tmp/aquascope_train.log"
_train_log_fh = None                           # open file handle for the running subprocess
_train_log_thread = None                       # daemon thread that tees subprocess output
# Set True when user starts training; cleared when user clicks "Close" on the modal.
# The tracker keeps inference paused while this is True — so the user controls when
# inference resumes (via /train/acknowledge), not the subprocess exit.
_train_unacked: bool = False
_TRAIN_MIN_LABELS: int = 100                   # button stays disabled below this
_TRAIN_DEFAULT_EPOCHS: int = 30
_TRAIN_DEFAULT_BATCH: int = 2                  # Orin Nano unified-memory friendly
_TRAIN_LOG_TAIL_BYTES: int = 16384             # how much of the tail /train/log returns
# Wall-clock anchor for the elapsed-timer the dashboard shows. The subprocess
# only writes elapsed_sec at epoch boundaries / phase changes — between those,
# the modal would show "—" for the slow data-setup + model-load phase. We
# synthesize elapsed in get_train_status() so the timer ticks every second.
_train_started_at: float | None = None
_train_label_count_at_start: int = 0
_train_epochs_used: int = _TRAIN_DEFAULT_EPOCHS

_RESOLUTIONS = {
    "480p":  (854,  480),
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
}


def hat_mode_enabled() -> bool:
    with _hat_lock:
        return _hat_mode


def trails_mode_enabled() -> bool:
    with _trails_lock:
        return _trails_enabled


def enhance_mode_enabled() -> bool:
    with _enhance_lock:
        return _enhance_enabled


def get_conf_threshold() -> float:
    with _conf_lock:
        return _conf_threshold


def get_resolution() -> str:
    with _resolution_lock:
        return _resolution


def set_models_dir(path: str) -> None:
    global _models_dir
    _models_dir = path


def set_model_path(path: str) -> None:
    global _model_path
    with _model_lock:
        _model_path = path


def get_model_path() -> str:
    with _model_lock:
        return _model_path


# ── Labeling: public API used by tracker.py ──────────────
def set_label_output_dir(path: str) -> None:
    global _label_output_dir
    _label_output_dir = path


def set_label_pending_dir(path: str) -> None:
    global _label_pending_dir
    _label_pending_dir = path


def set_label_class_names(names) -> None:
    """Accept list/dict from Ultralytics (`model.names` is a dict id→name)."""
    global _label_class_names
    if isinstance(names, dict):
        _label_class_names = [names[k] for k in sorted(names)]
    else:
        _label_class_names = list(names) if names else []


def label_enabled() -> bool:
    with _label_lock:
        return _label_enabled


def label_should_capture(track_id: int) -> bool:
    """True iff capture is on AND this track_id hasn't been queued recently AND queue isn't full."""
    with _label_lock:
        if not _label_enabled or len(_label_pending) >= _LABEL_QUEUE_MAX:
            return False
        now = time.time()
        last = _label_last_capture.get(int(track_id), 0.0)
        if now - last < _LABEL_THROTTLE_SEC:
            return False
        _label_last_capture[int(track_id)] = now
        return True


def enqueue_label_candidate(jpeg_bytes: bytes, bbox, img_w: int, img_h: int,
                            class_idx: int, track_id: int) -> str:
    """Persist the JPEG and queue a candidate dict. Returns the candidate id."""
    os.makedirs(_label_pending_dir, exist_ok=True)
    cid = uuid.uuid4().hex[:12]
    path = os.path.join(_label_pending_dir, f"{cid}.jpg")
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    candidate = {
        "id": cid,
        "frame_path": path,
        "bbox": [int(v) for v in bbox],
        "img_w": int(img_w),
        "img_h": int(img_h),
        "class_idx": int(class_idx),
        "track_id": int(track_id),
        "ts": time.time(),
    }
    with _label_lock:
        _label_pending.append(candidate)
    return cid


def _label_decide(candidate_id: str, keep: bool) -> dict:
    """Persist (if keep=True) or just discard. Always removes the pending JPEG."""
    with _label_lock:
        idx = next((i for i, c in enumerate(_label_pending) if c["id"] == candidate_id), -1)
        if idx < 0:
            return {"error": "candidate not found", "id": candidate_id}
        candidate = _label_pending.pop(idx)
        out_dir = _label_output_dir

    result: dict = {"saved": False}
    if keep:
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{ts}_{candidate['id']}"
        img_dst = os.path.join(out_dir, "images", f"{base}.jpg")
        lbl_dst = os.path.join(out_dir, "labels", f"{base}.txt")
        with open(candidate["frame_path"], "rb") as f:
            data = f.read()
        with open(img_dst, "wb") as f:
            f.write(data)
        x1, y1, x2, y2 = candidate["bbox"]
        w, h = candidate["img_w"], candidate["img_h"]
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        with open(lbl_dst, "w") as f:
            f.write(f"{candidate['class_idx']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        result = {"saved": True, "image": img_dst, "label": lbl_dst}

    try:
        os.remove(candidate["frame_path"])
    except OSError:
        pass

    with _label_lock:
        result["queued"] = len(_label_pending)
    return result


# ── Training: public API used by tracker.py ──────────────
def train_running() -> bool:
    with _train_lock:
        return _train_proc is not None and _train_proc.poll() is None


def inference_should_pause() -> bool:
    """Tracker pauses while training is running OR while the finished-modal hasn't been
    acknowledged by the user (Close button). This keeps inference off the GPU until the
    user explicitly hands control back."""
    with _train_lock:
        if _train_proc is not None and _train_proc.poll() is None:
            return True
        return _train_unacked


def acknowledge_training() -> dict:
    """Called by the dashboard when user clicks Close on the training modal.
    Clears the 'unacknowledged' flag so the tracker resumes inference."""
    global _train_unacked
    with _train_lock:
        was = _train_unacked
        _train_unacked = False
    return {"acknowledged": True, "was_pending": was}


def get_train_status() -> dict:
    """Read the JSON status file the subprocess writes; reconcile with proc state."""
    status: dict = {"state": "idle"}
    try:
        with open(_train_status_file) as f:
            status = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    with _train_lock:
        if _train_proc is not None:
            rc = _train_proc.poll()
            if rc is not None and status.get("state") not in ("done", "failed"):
                # Process exited but status file wasn't updated.
                status["state"] = "done" if rc == 0 else "failed"
                status.setdefault("message", f"subprocess exited rc={rc}")
        # Synthesize elapsed_sec from the wall clock so the timer ticks every
        # second from the moment the user clicks Train, not just at epoch
        # boundaries (the subprocess can spend 30+s loading torch + the model
        # before its first status write). Take max() so a per-epoch value from
        # the subprocess can never go backwards.
        active = status.get("state") in ("starting", "training", "exporting")
        if _train_started_at is not None and active:
            wall = int(time.time() - _train_started_at)
            status["elapsed_sec"] = max(int(status.get("elapsed_sec", 0)), wall)
        # ETA fallback for the "starting" phase: the subprocess can't compute a
        # per-epoch ETA until epoch 1 finishes, so use the initial heuristic.
        if status.get("state") == "starting" and not status.get("eta_sec"):
            est = estimate_training_minutes(_train_label_count_at_start, _train_epochs_used)
            est_total_sec = ((est["low_min"] + est["high_min"]) / 2.0) * 60.0
            status["eta_sec"] = max(0, int(est_total_sec - status.get("elapsed_sec", 0)))
    return status


def count_user_labels() -> int:
    labels_dir = os.path.join(_label_output_dir, "labels")
    if not os.path.isdir(labels_dir):
        return 0
    return sum(1 for f in os.listdir(labels_dir) if f.endswith(".txt"))


def estimate_training_minutes(label_count: int, epochs: int = _TRAIN_DEFAULT_EPOCHS) -> dict:
    """
    Rough ETA for the dashboard to show on the confirm dialog.
    Real numbers come from the subprocess after epoch 1 (avg_epoch_sec).
    Heuristic: ~1.5 min/epoch on Orin Nano + ~5ms per extra training sample/epoch.
    """
    base_min = epochs * 1.5
    extra_min = (label_count * epochs * 0.005) / 60.0
    total = base_min + extra_min
    return {"low_min": max(15, int(total - 10)),
            "high_min": int(total + 15),
            "epochs": epochs}


def latest_engine_version() -> tuple[int, str | None]:
    """Return (version, path) for the highest models/best.engine_v<N>, or (0, None)."""
    if not os.path.isdir(_models_dir):
        return 0, None
    best_n = 0
    best_path: str | None = None
    for name in os.listdir(_models_dir):
        if not name.startswith("best.engine_v"):
            continue
        try:
            n = int(name[len("best.engine_v"):])
        except ValueError:
            continue
        if n > best_n:
            best_n = n
            best_path = os.path.join(_models_dir, name)
    return best_n, best_path


def _tee_train_output(proc: subprocess.Popen, log_fh) -> None:
    """Background pump: copy each subprocess output line to BOTH the log file
    (for /train/log polling) AND the parent's stdout (so the operator sees the
    same stream in the terminal that launched the dashboard)."""
    try:
        for raw in iter(proc.stdout.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                line = repr(raw)
            try:
                log_fh.write(line)
                log_fh.flush()
            except (OSError, ValueError):
                pass
            try:
                sys.stdout.write("[TRAIN] " + line)
                sys.stdout.flush()
            except (OSError, ValueError):
                pass
    finally:
        try:
            proc.stdout.close()
        except (OSError, ValueError):
            pass
        try:
            log_fh.close()
        except (OSError, ValueError):
            pass


def start_training(epochs: int = _TRAIN_DEFAULT_EPOCHS) -> dict:
    """Spawn the training subprocess if not already running."""
    global _train_proc, _train_log_fh, _train_log_thread, _train_unacked
    global _train_started_at, _train_label_count_at_start, _train_epochs_used
    with _train_lock:
        if _train_proc is not None and _train_proc.poll() is None:
            return {"error": "training already in progress"}
        _train_unacked = True
        # Clear stale status + log so the dashboard doesn't show old data.
        for path in (_train_status_file, _train_log_file):
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            _train_log_fh = open(_train_log_file, "w", buffering=1)
        except OSError as e:
            return {"error": f"could not open log file: {e}"}
        cmd = [
            sys.executable, "-u", "training/train_jetson.py",
            "--status-file", _train_status_file,
            "--epochs", str(epochs),
            "--batch",  str(_TRAIN_DEFAULT_BATCH),
        ]
        try:
            _train_proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                bufsize=0,
            )
        except OSError as e:
            try:
                _train_log_fh.close()
            except OSError:
                pass
            _train_log_fh = None
            return {"error": str(e)}
        _train_log_thread = threading.Thread(
            target=_tee_train_output,
            args=(_train_proc, _train_log_fh),
            daemon=True,
        )
        _train_log_thread.start()
        _train_started_at = time.time()
        _train_label_count_at_start = count_user_labels()
        _train_epochs_used = epochs
        return {"started": True, "pid": _train_proc.pid}


def cancel_training() -> dict:
    global _train_proc, _train_log_fh
    with _train_lock:
        if _train_proc is None or _train_proc.poll() is not None:
            return {"error": "no training in progress"}
        _train_proc.terminate()
        try:
            _train_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _train_proc.kill()
        if _train_log_fh is not None:
            try:
                _train_log_fh.close()
            except OSError:
                pass
            _train_log_fh = None
    return {"cancelled": True}


def get_train_log(max_bytes: int = _TRAIN_LOG_TAIL_BYTES) -> str:
    """Return the tail of the training subprocess log (max_bytes from the end)."""
    try:
        size = os.path.getsize(_train_log_file)
    except OSError:
        return ""
    try:
        with open(_train_log_file, "rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, 2)
                f.readline()  # discard partial first line
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""

_screenshots: list[dict] = []   # [{filename, ts, label}]
_screenshots_lock = threading.Lock()
_screenshot_dir = "fish_logs/screenshots"


def push_frame(jpeg_bytes: bytes) -> None:
    global _frame
    with _lock:
        _frame = jpeg_bytes


def push_stats(data: dict) -> None:
    global _stats
    with _stats_lock:
        _stats = data


def request_reset() -> bool:
    global _reset_flag
    with _reset_lock:
        if _reset_flag:
            _reset_flag = False
            return True
    return False


# Per-connection MJPEG session cap. Viewers must refresh the page after this
# elapses to start a new connection.
_STREAM_SESSION_LIMIT_SEC = 180


# ── HTTP handler ──────────────────────────────────────────
class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A002
        pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            self._serve_index()
        elif p == "/stream":
            self._serve_stream()
        elif p == "/stats":
            self._serve_json(_stats, _stats_lock)
        elif p == "/reset":
            self._serve_reset()
        elif p == "/hat":
            self._serve_hat()
        elif p == "/trails":
            self._serve_trails()
        elif p == "/enhance":
            self._serve_enhance()
        elif p == "/conf":
            self._serve_conf()
        elif p == "/resolution":
            self._serve_resolution()
        elif p == "/models":
            self._serve_models_list()
        elif p == "/model":
            self._serve_model_select()
        elif p == "/label/state":
            self._serve_label_state()
        elif p == "/label/toggle":
            self._serve_label_toggle()
        elif p == "/label/queue":
            self._serve_label_queue()
        elif p.startswith("/label/image/"):
            self._serve_label_image(p[len("/label/image/"):])
        elif p == "/label/decision":
            self._serve_label_decision()
        elif p == "/train/labels":
            self._serve_train_labels()
        elif p == "/train/start":
            self._serve_train_start()
        elif p == "/train/status":
            self._serve_train_status()
        elif p == "/train/cancel":
            self._serve_train_cancel()
        elif p == "/train/log":
            self._serve_train_log()
        elif p == "/train/acknowledge":
            self._serve_train_acknowledge()
        elif p == "/screenshot":
            self._serve_screenshot()
        elif p == "/screenshots":
            self._serve_screenshot_list()
        elif p.startswith("/screenshots/"):
            self._serve_screenshot_file(p[len("/screenshots/"):])
        else:
            self.send_response(404)
            self.end_headers()

    # ── helpers ───────────────────────────────────────────

    def _json_response(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, obj, lock):
        with lock:
            body = json.dumps(obj).encode()
        self._json_response(body)

    def _serve_reset(self):
        global _reset_flag
        with _reset_lock:
            _reset_flag = True
        self._json_response(b'{"status":"reset requested"}')

    def _serve_hat(self):
        global _hat_mode
        with _hat_lock:
            _hat_mode = not _hat_mode
            state = _hat_mode
        self._json_response(json.dumps({"hat": state}).encode())

    def _serve_trails(self):
        global _trails_enabled
        with _trails_lock:
            _trails_enabled = not _trails_enabled
            state = _trails_enabled
        self._json_response(json.dumps({"trails": state}).encode())

    def _serve_enhance(self):
        global _enhance_enabled
        with _enhance_lock:
            _enhance_enabled = not _enhance_enabled
            state = _enhance_enabled
        self._json_response(json.dumps({"enhance": state}).encode())

    def _serve_resolution(self):
        global _resolution
        qs = self.path.split("?", 1)[1] if "?" in self.path else ""
        params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
        val = params.get("v", "")
        if val not in _RESOLUTIONS:
            self._json_response(b'{"error":"invalid resolution"}')
            return
        with _resolution_lock:
            _resolution = val
        self._json_response(json.dumps({"resolution": val}).encode())

    def _serve_conf(self):
        global _conf_threshold
        qs = self.path.split("?", 1)[1] if "?" in self.path else ""
        params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
        try:
            val = float(params["v"])
            val = max(0.05, min(0.95, val))
        except (KeyError, ValueError):
            with _conf_lock:
                val = _conf_threshold
        with _conf_lock:
            _conf_threshold = val
        self._json_response(json.dumps({"conf": round(val, 2)}).encode())

    def _serve_models_list(self):
        models_dir = _models_dir
        models: list[str] = []
        if os.path.isdir(models_dir):
            for name in sorted(os.listdir(models_dir)):
                if name.endswith(_MODEL_EXTS):
                    models.append(os.path.join(models_dir, name))
        body = json.dumps({"models": models, "current": get_model_path()}).encode()
        self._json_response(body)

    def _serve_model_select(self):
        qs = self.path.split("?", 1)[1] if "?" in self.path else ""
        params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
        raw = unquote(params.get("v", ""))
        # Reject path traversal: only the basename is honored, resolved under _models_dir.
        name = os.path.basename(raw)
        if not name or not name.endswith(_MODEL_EXTS):
            self._json_response(b'{"error":"invalid model"}')
            return
        full = os.path.join(_models_dir, name)
        if not os.path.isfile(full):
            self._json_response(json.dumps({"error": "not found", "path": full}).encode())
            return
        set_model_path(full)
        self._json_response(json.dumps({"model": full}).encode())

    # ── Labeling tab handlers ─────────────────────────────

    def _serve_label_state(self):
        with _label_lock:
            body = json.dumps({"enabled": _label_enabled, "queued": len(_label_pending)}).encode()
        self._json_response(body)

    def _serve_label_toggle(self):
        global _label_enabled
        qs = self.path.split("?", 1)[1] if "?" in self.path else ""
        params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
        val = params.get("v")
        with _label_lock:
            if val == "1":
                _label_enabled = True
            elif val == "0":
                _label_enabled = False
            else:
                _label_enabled = not _label_enabled
            state = _label_enabled
        self._json_response(json.dumps({"enabled": state}).encode())

    def _serve_label_queue(self):
        with _label_lock:
            pub = []
            for c in _label_pending:
                cls = (_label_class_names[c["class_idx"]]
                       if 0 <= c["class_idx"] < len(_label_class_names)
                       else f"class_{c['class_idx']}")
                pub.append({
                    "id": c["id"],
                    "bbox": c["bbox"],
                    "img_w": c["img_w"],
                    "img_h": c["img_h"],
                    "class_idx": c["class_idx"],
                    "class_name": cls,
                    "track_id": c["track_id"],
                    "image_url": f"/label/image/{c['id']}",
                })
        self._json_response(json.dumps({"queue": pub, "count": len(pub)}).encode())

    def _serve_label_image(self, candidate_id: str):
        candidate_id = os.path.basename(candidate_id)   # safety
        with _label_lock:
            cand = next((c for c in _label_pending if c["id"] == candidate_id), None)
        if cand is None or not os.path.exists(cand["frame_path"]):
            self.send_response(404)
            self.end_headers()
            return
        with open(cand["frame_path"], "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ── Training tab handlers ─────────────────────────────

    def _serve_train_labels(self):
        count = count_user_labels()
        body = json.dumps({
            "count": count,
            "min_required": _TRAIN_MIN_LABELS,
            "ready": count >= _TRAIN_MIN_LABELS,
            "estimate": estimate_training_minutes(count),
        }).encode()
        self._json_response(body)

    def _serve_train_start(self):
        count = count_user_labels()
        if count < _TRAIN_MIN_LABELS:
            self._json_response(json.dumps({
                "error": f"need at least {_TRAIN_MIN_LABELS} labels (have {count})",
                "count": count,
                "min_required": _TRAIN_MIN_LABELS,
            }).encode())
            return
        self._json_response(json.dumps(start_training()).encode())

    def _serve_train_status(self):
        status = get_train_status()
        # Pin the running flag from proc state so the UI doesn't trust a stale file.
        status["running"] = train_running()
        ver, path = latest_engine_version()
        status["latest_version"] = ver
        status["latest_engine"] = path
        self._json_response(json.dumps(status).encode())

    def _serve_train_cancel(self):
        self._json_response(json.dumps(cancel_training()).encode())

    def _serve_train_log(self):
        log_text = get_train_log()
        self._json_response(json.dumps({"log": log_text}).encode())

    def _serve_train_acknowledge(self):
        self._json_response(json.dumps(acknowledge_training()).encode())

    def _serve_label_decision(self):
        qs = self.path.split("?", 1)[1] if "?" in self.path else ""
        params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
        cid = unquote(params.get("id", ""))
        keep = params.get("keep") == "1"
        if not cid:
            self._json_response(b'{"error":"missing id"}')
            return
        result = _label_decide(cid, keep)
        self._json_response(json.dumps(result).encode())

    def _serve_screenshot(self):
        with _lock:
            frame = _frame
        if not frame:
            self._json_response(b'{"error":"no frame yet"}')
            return
        os.makedirs(_screenshot_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"snap_{ts}.jpg"
        fpath = os.path.join(_screenshot_dir, fname)
        with open(fpath, "wb") as f:
            f.write(frame)
        entry = {"filename": fname, "ts": ts, "label": f"Snap {ts[9:]}"}
        with _screenshots_lock:
            _screenshots.insert(0, entry)
            if len(_screenshots) > 50:
                _screenshots.pop()
        self._json_response(json.dumps({"filename": fname}).encode())

    def _serve_screenshot_list(self):
        with _screenshots_lock:
            body = json.dumps(_screenshots).encode()
        self._json_response(body)

    def _serve_screenshot_file(self, filename: str):
        # Safety: no path traversal
        filename = os.path.basename(filename)
        fpath = os.path.join(_screenshot_dir, filename)
        if not os.path.exists(fpath):
            self.send_response(404)
            self.end_headers()
            return
        with open(fpath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        deadline = time.monotonic() + _STREAM_SESSION_LIMIT_SEC
        try:
            while time.monotonic() < deadline:
                with _lock:
                    frame = _frame
                if frame:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_index(self):
        html = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AquaScope — Neural Tank Monitor</title>
<style>
/* ============================================================
   AquaScope — Cyber/Lab Design System
   Dark, neon aquatic data-viz. Teal/cyan on near-black.
   ============================================================ */

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  /* ── Surfaces ── */
  --bg-0:      #05090d;   /* deepest backdrop */
  --bg-1:      #070d12;   /* app background */
  --panel:     #0a141b;   /* card surface */
  --panel-2:   #0d1c25;   /* raised surface */
  --panel-3:   #102530;   /* hover / active surface */
  --inset:     #04080b;   /* recessed wells (feed, charts) */

  /* ── Lines ── */
  --border:    #143038;
  --border-2:  #1c424d;
  --grid:      rgba(0, 229, 204, 0.05);

  /* ── Text ── */
  --text:      #d2ecea;
  --text-2:    #8fb3b3;
  --dim:       #5a7d80;
  --faint:     #3c585c;

  /* ── Accents (aquatic) ── */
  --teal:      #00e5cc;   /* primary */
  --teal-2:    #19f0d6;
  --cyan:      #2ad4ff;   /* data viz secondary */
  --aqua:      #57f5b6;   /* data viz tertiary */
  --deep:      #4aa8ff;   /* cool blue */

  /* ── Status ── */
  --good:      #45e0a8;
  --warn:      #ffc857;
  --alert:     #ff6f6f;
  --crit:      #ff4d6d;

  /* ── Glows ── */
  --glow-teal: 0 0 0 1px rgba(0,229,204,.35), 0 0 18px -2px rgba(0,229,204,.45);
  --glow-soft: 0 0 24px -6px rgba(0,229,204,.30);

  --font:  'Space Grotesk', system-ui, sans-serif;
  --mono:  'JetBrains Mono', ui-monospace, monospace;

  --r-sm: 6px;
  --r:    10px;
  --r-lg: 14px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

html, body { height: 100%; }

body {
  background: var(--bg-1);
  color: var(--text);
  font-family: var(--font);
  font-size: 14px;
  line-height: 1.45;
  -webkit-font-smoothing: antialiased;
  overflow: hidden;
}

/* ambient grid + radial wash behind everything */
#app-bg {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(900px 600px at 78% -8%, rgba(0,229,204,.08), transparent 60%),
    radial-gradient(700px 500px at 8% 108%, rgba(74,168,255,.07), transparent 60%),
    linear-gradient(180deg, #070f15 0%, #05090d 100%);
}
#app-bg::after {
  content: ""; position: absolute; inset: 0;
  background-image:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: 44px 44px;
  mask-image: radial-gradient(120% 120% at 50% 0%, #000 40%, transparent 100%);
}

::selection { background: rgba(0,229,204,.3); color: #fff; }

/* scrollbars */
* { scrollbar-width: thin; scrollbar-color: var(--border-2) transparent; }
*::-webkit-scrollbar { width: 9px; height: 9px; }
*::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 8px; border: 2px solid transparent; background-clip: content-box; }
*::-webkit-scrollbar-thumb:hover { background: #2a5b66; background-clip: content-box; }

/* ── typographic helpers ── */
.mono { font-family: var(--mono); }
.eyebrow {
  font-family: var(--mono);
  font-size: 10px; letter-spacing: .22em; text-transform: uppercase;
  color: var(--dim);
}
.tabular { font-variant-numeric: tabular-nums; font-family: var(--mono); }

/* ── App layout ── */
#root { position: relative; z-index: 1; height: 100%; }

.shell {
  display: grid;
  grid-template-columns: 232px 1fr;
  height: 100vh;
}

/* ============================================================
   Sidebar
   ============================================================ */
.sidebar {
  background: linear-gradient(180deg, #08121a 0%, #060c11 100%);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  padding: 18px 0 14px;
  position: relative;
  z-index: 30;
}
.brand {
  display: flex; align-items: center; gap: 11px;
  padding: 4px 18px 18px;
}
.brand-mark {
  width: 34px; height: 34px; flex: none;
  display: grid; place-items: center;
  border-radius: 9px;
  background: radial-gradient(circle at 30% 25%, #0c2730, #07151c);
  border: 1px solid var(--border-2);
  box-shadow: inset 0 0 12px rgba(0,229,204,.18);
  position: relative;
}
.brand-mark svg { width: 21px; height: 21px; }
.brand-mark::after {
  content:""; position:absolute; inset:0; border-radius:9px;
  box-shadow: 0 0 16px -4px rgba(0,229,204,.6);
}
.brand-name {
  font-weight: 700; font-size: 17px; letter-spacing: .02em;
  line-height: 1;
}
.brand-name b { color: var(--teal); font-weight: 700; }
.brand-sub {
  font-family: var(--mono); font-size: 9px; letter-spacing: .2em;
  color: var(--dim); text-transform: uppercase; margin-top: 3px; white-space: nowrap;
}

.nav-group { padding: 6px 12px; }
.nav-label {
  font-family: var(--mono); font-size: 9px; letter-spacing: .2em;
  text-transform: uppercase; color: var(--faint);
  padding: 12px 8px 7px;
}
.nav-item {
  display: flex; align-items: center; gap: 11px;
  padding: 9px 10px; border-radius: var(--r-sm);
  color: var(--text-2); cursor: pointer;
  font-size: 13.5px; font-weight: 500;
  position: relative; transition: background .14s, color .14s;
  user-select: none;
}
.nav-item span { white-space: nowrap; }
.nav-item .ni-icon { width: 18px; height: 18px; flex: none; opacity: .75; }
.nav-item:hover { background: rgba(255,255,255,.03); color: var(--text); }
.nav-item:hover .ni-icon { opacity: 1; }
.nav-item.active {
  color: var(--teal); background: linear-gradient(90deg, rgba(0,229,204,.12), rgba(0,229,204,.02));
}
.nav-item.active .ni-icon { opacity: 1; color: var(--teal); }
.nav-item.active::before {
  content: ""; position: absolute; left: -12px; top: 6px; bottom: 6px;
  width: 3px; border-radius: 0 3px 3px 0; background: var(--teal);
  box-shadow: 0 0 10px var(--teal);
}
.nav-badge {
  margin-left: auto; font-family: var(--mono); font-size: 10px;
  background: rgba(0,229,204,.14); color: var(--teal);
  padding: 1px 6px; border-radius: 20px; border: 1px solid rgba(0,229,204,.25);
}
.nav-badge.alert { background: rgba(255,111,111,.14); color: var(--alert); border-color: rgba(255,111,111,.3); }

.nav-spacer { flex: 1; }

.dev-card {
  margin: 8px 14px 0; padding: 11px 12px;
  border: 1px solid var(--border); border-radius: var(--r);
  background: var(--panel);
}
.dev-row { display: flex; align-items: center; justify-content: space-between; font-size: 11px; }
.dev-row + .dev-row { margin-top: 7px; }
.dev-row .lbl { color: var(--dim); font-family: var(--mono); font-size: 10px; letter-spacing: .06em; }
.dev-row .val { font-family: var(--mono); color: var(--text); white-space: nowrap; }
.dev-bar { height: 4px; border-radius: 3px; background: #0a181d; overflow: hidden; margin-top: 4px; }
.dev-bar > i { display: block; height: 100%; border-radius: 3px; }

.conn {
  display: flex; align-items: center; gap: 8px;
  padding: 12px 20px 2px; font-size: 11px; color: var(--text-2);
  font-family: var(--mono);
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--good); box-shadow: 0 0 8px var(--good); }
.dot.live { animation: pulse 1.8s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .45; transform: scale(.8); } }

/* ============================================================
   Main column
   ============================================================ */
.main { display: flex; flex-direction: column; min-width: 0; height: 100vh; }

.topbar {
  height: 58px; flex: none;
  display: flex; align-items: center; gap: 16px;
  padding: 0 22px;
  border-bottom: 1px solid var(--border);
  background: rgba(7,13,18,.72);
  backdrop-filter: blur(10px);
  position: relative; z-index: 20;
}
.tb-title { font-size: 17px; font-weight: 600; letter-spacing: -.01em; white-space: nowrap; }
.tb-sub { font-family: var(--mono); font-size: 11px; color: var(--dim); margin-top: 1px; white-space: nowrap; }
.live-pill {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--mono); font-size: 10.5px; letter-spacing: .12em;
  color: var(--teal); padding: 4px 9px; border-radius: 20px; white-space: nowrap; flex: none;
  border: 1px solid rgba(0,229,204,.3); background: rgba(0,229,204,.07);
}
.tb-spacer { flex: 1; }
.tb-right { display: flex; align-items: center; gap: 14px; }
.clock { font-family: var(--mono); font-size: 13px; color: var(--cyan); }
.clock .date { color: var(--dim); font-size: 11px; }

.btn {
  font-family: var(--font); font-size: 13px; font-weight: 500;
  display: inline-flex; align-items: center; gap: 7px;
  padding: 7px 13px; border-radius: var(--r-sm); cursor: pointer;
  border: 1px solid var(--border-2); background: var(--panel-2); color: var(--text);
  transition: all .14s;
}
.btn:hover { background: var(--panel-3); border-color: #2a5b66; }
.btn svg { width: 15px; height: 15px; }
.btn-primary {
  background: linear-gradient(135deg, var(--teal), #00bfab);
  color: #022; border: none; font-weight: 600;
  box-shadow: 0 0 0 1px rgba(0,229,204,.3), 0 6px 18px -8px rgba(0,229,204,.7);
}
.btn-primary:hover { filter: brightness(1.08); }
.btn-ghost { background: transparent; border-color: var(--border); }
.btn-icon { padding: 7px; }

.content {
  flex: 1; overflow-y: auto; overflow-x: hidden;
  padding: 22px;
  position: relative;
}

/* ============================================================
   Cards / panels
   ============================================================ */
.card {
  background: linear-gradient(160deg, var(--panel-2) 0%, var(--panel) 100%);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  position: relative;
}
.card.pad { padding: 16px; }
.card-h {
  display: flex; align-items: center; gap: 10px; min-width: 0;
  padding: 13px 16px; border-bottom: 1px solid var(--border);
}
.card-h .ch-title { font-size: 13px; font-weight: 600; letter-spacing: .01em; white-space: nowrap; flex: none; }
.card-h .ch-sub { font-family: var(--mono); font-size: 10px; color: var(--dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; text-align: right; }
.card-h .ch-spacer { flex: 1; }

.kpi { padding: 15px 16px 16px; }
.kpi-top { display: flex; align-items: center; justify-content: space-between; }
.kpi-label { font-family: var(--mono); font-size: 9.5px; letter-spacing: .11em; text-transform: uppercase; color: var(--dim); white-space: nowrap; }
.kpi-val { font-family: var(--mono); font-size: 30px; font-weight: 600; line-height: 1; margin-top: 12px; letter-spacing: -.02em; }
.kpi-val .unit { font-size: 14px; color: var(--dim); margin-left: 4px; }
.kpi-delta { font-family: var(--mono); font-size: 11px; margin-top: 7px; display: flex; align-items: center; gap: 5px; }
.up { color: var(--good); }
.down { color: var(--alert); }
.flat { color: var(--dim); }

.chip {
  display: inline-flex; align-items: center; gap: 5px; white-space: nowrap;
  font-family: var(--mono); font-size: 10px; letter-spacing: .06em;
  padding: 3px 8px; border-radius: 20px;
  border: 1px solid var(--border-2); color: var(--text-2);
}
.chip.on { color: var(--teal); border-color: rgba(0,229,204,.35); background: rgba(0,229,204,.08); }
.chip.warn { color: var(--warn); border-color: rgba(255,200,87,.3); background: rgba(255,200,87,.07); }
.chip.alert { color: var(--alert); border-color: rgba(255,111,111,.3); background: rgba(255,111,111,.07); }

/* grids */
.grid { display: grid; gap: 14px; }
.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }

/* toggles */
.toggle {
  width: 38px; height: 21px; border-radius: 20px; flex: none;
  background: #0c1c22; border: 1px solid var(--border-2);
  position: relative; cursor: pointer; transition: all .18s;
}
.toggle > i {
  position: absolute; top: 2px; left: 2px; width: 15px; height: 15px;
  border-radius: 50%; background: var(--dim); transition: all .18s;
}
.toggle.on { background: rgba(0,229,204,.2); border-color: var(--teal); }
.toggle.on > i { left: 19px; background: var(--teal); box-shadow: 0 0 8px var(--teal); }

/* range slider */
input[type=range].rng {
  -webkit-appearance: none; appearance: none; width: 100%; height: 4px;
  border-radius: 3px; outline: none; cursor: pointer; background: #0e2228;
}
input[type=range].rng::-webkit-slider-thumb {
  -webkit-appearance: none; width: 15px; height: 15px; border-radius: 50%;
  background: var(--teal); border: 2px solid #04141a; box-shadow: 0 0 8px rgba(0,229,204,.7); cursor: pointer;
}
input[type=range].rng::-moz-range-thumb {
  width: 15px; height: 15px; border-radius: 50%; background: var(--teal);
  border: 2px solid #04141a; cursor: pointer;
}

select.sel {
  width: 100%; padding: 8px 30px 8px 11px; border-radius: var(--r-sm);
  border: 1px solid var(--border-2); background: var(--panel); color: var(--text);
  font-family: var(--mono); font-size: 12px; cursor: pointer; outline: none;
  appearance: none; -webkit-appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%235a7d80'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: right 11px center;
}
select.sel:focus { border-color: var(--teal); }

/* corner brackets */
.bracket { position: absolute; width: 14px; height: 14px; border-color: var(--teal); border-style: solid; opacity: .55; pointer-events: none; }
.bracket.tl { top: 9px; left: 9px; border-width: 1.5px 0 0 1.5px; }
.bracket.tr { top: 9px; right: 9px; border-width: 1.5px 1.5px 0 0; }
.bracket.bl { bottom: 9px; left: 9px; border-width: 0 0 1.5px 1.5px; }
.bracket.br { bottom: 9px; right: 9px; border-width: 0 1.5px 1.5px 0; }

/* fade-in (capture-safe: never animates opacity to 0, so DOM-clone screenshots stay visible) */
.view { animation: viewIn .35s ease; }
@keyframes viewIn { from { transform: translateY(7px); } to { transform: none; } }

.section-title { font-size: 12px; font-family: var(--mono); letter-spacing: .14em; text-transform: uppercase; color: var(--dim); margin: 4px 2px 2px; }

/* mobile nav (hidden on desktop) */
.mobile-bar { display: none; }
.hamburger { display: none; }
.scrim { display: none; }

/* ============================================================
   Responsive
   ============================================================ */
@media (max-width: 1180px) {
  .cols-4 { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 900px) {
  .shell { grid-template-columns: 1fr; }
  .sidebar {
    position: fixed; top: 0; bottom: 0; left: 0; width: 232px;
    transform: translateX(-100%); transition: transform .25s ease; z-index: 60;
  }
  .sidebar.open { transform: none; box-shadow: 0 0 60px rgba(0,0,0,.7); }
  .scrim.show { display: block; position: fixed; inset: 0; background: rgba(0,0,0,.55); z-index: 55; }
  .hamburger { display: inline-flex; }
  .content { padding: 16px; padding-bottom: 80px; }
  .cols-3 { grid-template-columns: 1fr; }
  .cols-2 { grid-template-columns: 1fr; }
  .clock .date { display: none; }
}
@media (max-width: 560px) {
  .cols-4 { grid-template-columns: 1fr 1fr; }
  .tb-sub { display: none; }
}

/* ============================================================
   AquaScope — View-specific styles
   ============================================================ */

.row-between { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
.dimc { color: var(--dim); }

/* segmented control */
.seg { display: inline-flex; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 3px; gap: 2px; }
.seg-btn {
  font-family: var(--mono); font-size: 11px; padding: 5px 12px; border-radius: 6px;
  border: none; background: transparent; color: var(--dim); cursor: pointer; transition: all .14s;
}
.seg-btn:hover { color: var(--text-2); }
.seg-btn.on { background: rgba(0,229,204,.14); color: var(--teal); }

/* shared confidence bar */
.conf-bar { display: inline-block; width: 54px; height: 4px; border-radius: 3px; background: #0e2228; overflow: hidden; vertical-align: middle; margin-right: 7px; }
.conf-bar > i { display: block; height: 100%; border-radius: 3px; }

/* ============================================================
   LIVE VIEW
   ============================================================ */
.live-grid { display: grid; grid-template-columns: 1fr 340px; gap: 16px; align-items: start; }
.feed-wrap {
  position: relative; aspect-ratio: 16 / 9; width: 100%;
  background: var(--inset); border: 1px solid var(--border-2); border-radius: var(--r-lg);
  overflow: hidden; box-shadow: var(--glow-soft);
}
.scanlines { position: absolute; inset: 0; pointer-events: none; background: repeating-linear-gradient(to bottom, transparent 0 3px, rgba(0,0,0,.10) 3px 4px); }
.feed-top { position: absolute; top: 12px; left: 12px; right: 12px; display: flex; align-items: center; gap: 10px; pointer-events: none; }
.rec { display: inline-flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 10.5px; letter-spacing: .14em; color: var(--alert); background: rgba(8,12,16,.7); padding: 4px 9px; border-radius: 20px; border: 1px solid rgba(255,111,111,.3); }
.rec > i { width: 7px; height: 7px; border-radius: 50%; background: var(--alert); box-shadow: 0 0 8px var(--alert); animation: pulse 1.6s infinite; }
.feed-tag { margin-left: auto; font-size: 10px; color: var(--teal); background: rgba(8,12,16,.7); padding: 4px 9px; border-radius: 20px; border: 1px solid var(--border-2); }
.snap-btn {
  position: absolute; bottom: 14px; right: 14px;
  display: inline-flex; align-items: center; gap: 7px; cursor: pointer;
  font-family: var(--font); font-size: 12.5px; font-weight: 600; color: #022;
  padding: 8px 13px; border-radius: 8px; border: none;
  background: linear-gradient(135deg, var(--teal), #00bfab);
  box-shadow: 0 6px 18px -8px rgba(0,229,204,.8);
}
.snap-btn:hover { filter: brightness(1.08); }
.snap-btn svg { width: 15px; height: 15px; }

.filmstrip { display: flex; gap: 9px; overflow-x: auto; padding-bottom: 4px; }
.film-thumb { flex: none; width: 116px; }
.film-img { height: 66px; border-radius: 7px; border: 1px solid var(--border-2); display: grid; place-items: center; overflow: hidden; }
.film-cap { display: block; font-size: 9.5px; color: var(--dim); margin-top: 5px; }

.fish-list { padding: 8px; overflow-y: auto; display: flex; flex-direction: column; gap: 5px; max-height: 320px; }
.fish-row { display: flex; align-items: center; gap: 10px; padding: 8px 9px; border-radius: 8px; border: 1px solid var(--border); background: var(--panel); transition: border-color .15s; }
.fish-row:hover { border-color: var(--border-2); }
.fish-swatch { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.fish-name { font-size: 13px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fish-meta { font-size: 10.5px; color: var(--dim); margin-top: 1px; }
.fish-conf { margin-left: auto; font-size: 12px; color: var(--teal); }
.ctl-row { display: flex; align-items: center; justify-content: space-between; padding: 7px 4px; cursor: pointer; font-size: 13px; border-radius: 6px; }
.ctl-row:hover { background: rgba(255,255,255,.02); }

/* ============================================================
   ANALYTICS — per-fish table
   ============================================================ */
.heat-legend { display: flex; align-items: center; gap: 8px; margin-top: 12px; font-size: 10px; color: var(--dim); }
.heat-bar { flex: none; width: 110px; height: 7px; border-radius: 4px; background: linear-gradient(90deg, rgba(0,229,204,.06), rgba(0,229,204,.9)); }

.ptable { padding: 6px 8px 10px; }
.pt-head, .pt-row { display: grid; grid-template-columns: 1.6fr 1.2fr 1.1fr 1fr .9fr 1.2fr; gap: 10px; align-items: center; padding: 9px 10px; }
.pt-head { font-family: var(--mono); font-size: 9.5px; letter-spacing: .12em; text-transform: uppercase; color: var(--faint); border-bottom: 1px solid var(--border); }
.pt-row { border-bottom: 1px solid rgba(20,48,56,.5); font-size: 12.5px; }
.pt-row:hover { background: rgba(0,229,204,.03); }
.pt-row:last-child { border-bottom: none; }
.pt-name { display: flex; align-items: center; gap: 9px; font-weight: 500; }
.pt-name em { color: var(--dim); font-style: normal; font-size: 11px; }
.pt-row .mono { font-size: 12px; }

/* ============================================================
   AI INSIGHTS
   ============================================================ */
.insights-grid { display: grid; grid-template-columns: 1fr 380px; gap: 16px; align-items: stretch; }
.ai-orb { width: 18px; height: 18px; border-radius: 50%; flex: none; background: radial-gradient(circle at 35% 30%, #7CFFCB, var(--teal) 55%, #007a6c); box-shadow: 0 0 10px rgba(0,229,204,.8); position: relative; }
.ai-orb::after { content: ""; position: absolute; inset: -3px; border-radius: 50%; border: 1px solid rgba(0,229,204,.4); animation: pulse 2s infinite; }

.health-card { padding: 16px; display: flex; flex-direction: column; align-items: center; }
.health-sub { width: 100%; margin-top: 14px; display: flex; flex-direction: column; gap: 8px; }
.hs-row { display: grid; grid-template-columns: 64px 1fr 26px; gap: 8px; align-items: center; font-size: 11px; }
.hs-bar { height: 5px; border-radius: 3px; background: #0e2228; overflow: hidden; }
.hs-bar > i { display: block; height: 100%; border-radius: 3px; transition: width .6s; }

.summary-card { display: flex; flex-direction: column; }
.summary-body { padding: 14px 16px; font-size: 13.5px; line-height: 1.6; color: var(--text-2); }
.summary-body b { color: var(--text); }
.summary-tags { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 14px; }

.feed-list { padding: 8px; display: flex; flex-direction: column; gap: 8px; }
.insight { display: flex; gap: 12px; padding: 13px; border-radius: 10px; border: 1px solid var(--border); background: var(--panel); position: relative; overflow: hidden; }
.insight::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; }
.insight.alert::before { background: var(--alert); } .insight.warn::before { background: var(--warn); }
.insight.good::before { background: var(--good); } .insight.info::before { background: var(--cyan); }
.ins-icon { width: 34px; height: 34px; flex: none; border-radius: 9px; border: 1px solid; display: grid; place-items: center; background: var(--inset); }
.ins-top { display: flex; align-items: center; gap: 9px; }
.ins-title { font-size: 13.5px; font-weight: 600; }
.ins-tag { font-size: 9.5px; color: var(--dim); border: 1px solid var(--border-2); padding: 1px 6px; border-radius: 20px; }
.ins-time { margin-left: auto; font-size: 10.5px; color: var(--faint); }
.ins-body { font-size: 12.5px; color: var(--text-2); line-height: 1.55; margin-top: 5px; }
.ins-conf { display: flex; align-items: center; gap: 8px; margin-top: 9px; font-size: 10px; }

/* Ask your tank */
.ask-card { display: flex; flex-direction: column; min-height: 520px; }
.ask-body { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.bubble { max-width: 88%; font-size: 13px; line-height: 1.55; padding: 11px 13px; border-radius: 13px; }
.bubble.ai { align-self: flex-start; background: var(--panel); border: 1px solid var(--border-2); border-bottom-left-radius: 4px; color: var(--text); }
.bubble.user { align-self: flex-end; background: linear-gradient(135deg, rgba(0,229,204,.18), rgba(0,229,204,.08)); border: 1px solid rgba(0,229,204,.3); border-bottom-right-radius: 4px; color: var(--text); }
.caret { color: var(--teal); animation: blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity: 0; } }
.ask-sugg { display: flex; flex-wrap: wrap; gap: 7px; padding: 0 14px 12px; }
.sugg { font-family: var(--font); font-size: 11.5px; color: var(--text-2); background: var(--panel); border: 1px solid var(--border-2); padding: 6px 11px; border-radius: 20px; cursor: pointer; transition: all .14s; }
.sugg:hover { border-color: var(--teal); color: var(--teal); }
.ask-input { display: flex; gap: 8px; padding: 12px 14px; border-top: 1px solid var(--border); }
.ask-input input { flex: 1; background: var(--inset); border: 1px solid var(--border-2); border-radius: 8px; padding: 9px 13px; color: var(--text); font-family: var(--font); font-size: 13px; outline: none; }
.ask-input input:focus { border-color: var(--teal); }
.ask-input button svg { width: 16px; height: 16px; }

/* ============================================================
   TRAINING
   ============================================================ */
.training-grid { display: grid; grid-template-columns: 1fr 360px; gap: 16px; align-items: start; }
.triage-body { padding: 16px; display: flex; flex-direction: column; align-items: center; gap: 16px; }
.crop-frame { position: relative; width: 100%; aspect-ratio: 4/3; max-width: 560px; border-radius: 10px; overflow: hidden; border: 1px solid var(--border-2); }
.crop-box { position: absolute; border: 1.6px solid; border-radius: 3px; }
.crop-label { position: absolute; top: -18px; left: -1px; font-size: 10px; color: #04141a; padding: 1px 6px; border-radius: 3px 3px 0 0; white-space: nowrap; }
.crop-conf { position: absolute; bottom: 8px; right: 10px; font-size: 10px; color: var(--teal); background: rgba(8,12,16,.7); padding: 3px 7px; border-radius: 20px; }
.triage-q { font-size: 16px; font-weight: 600; }
.triage-actions { display: flex; gap: 14px; }
.tri-btn { display: inline-flex; align-items: center; gap: 9px; font-family: var(--font); font-size: 15px; font-weight: 600; padding: 12px 30px; border-radius: 10px; cursor: pointer; border: 1px solid; transition: all .14s; }
.tri-btn svg { width: 19px; height: 19px; }
.tri-btn em { font-style: normal; font-size: 10px; opacity: .7; border: 1px solid currentColor; border-radius: 4px; padding: 1px 5px; }
.tri-btn.accept { color: var(--good); border-color: rgba(69,224,168,.4); background: rgba(69,224,168,.08); }
.tri-btn.accept:hover { background: rgba(69,224,168,.16); }
.tri-btn.reject { color: var(--alert); border-color: rgba(255,111,111,.4); background: rgba(255,111,111,.08); }
.tri-btn.reject:hover { background: rgba(255,111,111,.16); }
.triage-empty { padding: 50px 30px; color: var(--dim); text-align: center; font-size: 13px; }

.mini-stat { display: flex; flex-direction: column; gap: 5px; padding: 10px 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--panel); }
.mini-stat .tabular { font-size: 18px; font-weight: 600; }

.model-list { padding: 8px; display: flex; flex-direction: column; gap: 6px; }
.model-row { display: flex; align-items: center; gap: 12px; padding: 11px 12px; border-radius: 9px; border: 1px solid var(--border); background: var(--panel); }
.model-row.active { border-color: rgba(0,229,204,.35); background: linear-gradient(90deg, rgba(0,229,204,.06), var(--panel)); }
.model-name { font-size: 12.5px; display: flex; align-items: center; }
.model-meta { font-size: 10px; color: var(--dim); margin-top: 3px; }
.model-stats { margin-left: auto; display: flex; gap: 16px; text-align: right; }
.model-stats > div { display: flex; flex-direction: column; gap: 2px; }
.model-stats .tabular { font-size: 14px; font-weight: 600; }

/* train modal */
.modal-scrim { position: fixed; inset: 0; background: rgba(2,6,9,.8); backdrop-filter: blur(4px); z-index: 200; display: grid; place-items: center; padding: 20px; animation: viewIn .2s; }
.train-modal { width: min(620px, 96vw); max-height: 92vh; overflow-y: auto; box-shadow: 0 24px 70px rgba(0,0,0,.7); }
.tm-stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 14px; }
.tm-big { font-size: 24px; font-weight: 700; margin-top: 4px; }
.tm-bar { height: 8px; border-radius: 5px; background: #0e2228; overflow: hidden; }
.tm-bar > i { display: block; height: 100%; border-radius: 5px; background: linear-gradient(90deg, var(--cyan), var(--teal)); transition: width .3s; box-shadow: 0 0 10px var(--teal); }
.tm-log { background: var(--inset); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; font-family: var(--mono); font-size: 11px; line-height: 1.6; color: var(--text-2); height: 150px; overflow-y: auto; }
.tm-log .log-ok { color: var(--teal); }

/* ============================================================
   DEVICE HEALTH
   ============================================================ */
.device-banner { display: flex; align-items: center; gap: 16px; }
.dev-icon { width: 52px; height: 52px; flex: none; border-radius: 12px; display: grid; place-items: center; background: var(--inset); border: 1px solid var(--border-2); box-shadow: inset 0 0 14px rgba(0,229,204,.15); }
.dev-icon svg { width: 28px; height: 28px; }
.db-stats { display: flex; gap: 28px; }
.db-stats > div { display: flex; flex-direction: column; gap: 4px; }
.db-stats .tabular { font-size: 16px; font-weight: 600; white-space: nowrap; }
.gauge-card { padding: 16px 12px 14px; display: grid; place-items: center; }
.kv-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid rgba(20,48,56,.5); font-size: 12.5px; }
.kv-row > span { white-space: nowrap; }
.kv-row:last-child { border-bottom: none; }

/* ============================================================
   SETTINGS
   ============================================================ */
.set-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 0; border-bottom: 1px solid rgba(20,48,56,.5); }
.set-row:last-child { border-bottom: none; }
.set-label { font-size: 13.5px; font-weight: 500; }
.set-sub { font-size: 11px; color: var(--dim); margin-top: 3px; }
.set-control { flex: none; }

/* ============================================================
   MOBILE
   ============================================================ */
@media (max-width: 900px) {
  .live-grid, .insights-grid, .training-grid { grid-template-columns: 1fr; }
  .ask-card { min-height: 440px; }
  .db-stats { gap: 18px; flex-wrap: wrap; }
  .pt-head { display: none; }
  .pt-row { grid-template-columns: 1fr 1fr; gap: 6px 10px; padding: 12px 10px; border: 1px solid var(--border); border-radius: 8px; margin-bottom: 8px; }
  .pt-row > span:nth-child(6) { grid-column: 1 / -1; }
  .mobile-bar {
    display: flex; position: fixed; bottom: 0; left: 0; right: 0; height: 60px; z-index: 50;
    background: rgba(7,13,18,.94); backdrop-filter: blur(12px); border-top: 1px solid var(--border);
    align-items: center; justify-content: space-around; padding: 0 8px;
  }
  .mb-item { background: none; border: none; color: var(--dim); cursor: pointer; padding: 10px 16px; border-radius: 9px; }
  .mb-item .ni-icon { width: 22px; height: 22px; }
  .mb-item.on { color: var(--teal); background: rgba(0,229,204,.1); }
}
@media (max-width: 560px) {
  .tm-stat-row { grid-template-columns: repeat(2, 1fr); }
  .db-stats { width: 100%; justify-content: space-between; }
}
</style>
</head>
<body>
<div id="app-bg"></div>
<div id="root"></div>
<script>
/* ============================================================
   AquaScope — Neural Tank Monitor (vanilla, self-contained)
   Cyber/lab dashboard. Wired to the real stream.py endpoints
   where they exist; Analytics + AI Insights are simulated.
   ============================================================ */
(function () {
'use strict';

/* ── tiny hyperscript (HTML + SVG) ── */
var SVG_NS = 'http://www.w3.org/2000/svg';
var SVG_TAGS = new Set(['svg','path','circle','rect','line','text','g','defs',
  'linearGradient','radialGradient','stop','ellipse','polyline','polygon','animate']);

function h(tag, props) {
  var kids = Array.prototype.slice.call(arguments, 2);
  var isSvg = SVG_TAGS.has(tag);
  var el = isSvg ? document.createElementNS(SVG_NS, tag) : document.createElement(tag);
  if (props) {
    for (var k in props) {
      var v = props[k];
      if (v == null || v === false) continue;
      if (k === 'class') el.setAttribute('class', v);
      else if (k === 'html') el.innerHTML = v;
      else if (k === 'style') {
        if (typeof v === 'object') Object.assign(el.style, v);
        else el.setAttribute('style', v);
      } else if (k.slice(0, 2) === 'on' && typeof v === 'function') {
        el.addEventListener(k.slice(2).toLowerCase(), v);
      } else if (!isSvg && k in el) {
        try { el[k] = v; } catch (e) { el.setAttribute(k, v); }
      } else {
        el.setAttribute(k, v);
      }
    }
  }
  append(el, kids);
  return el;
}
function append(el, kids) {
  for (var i = 0; i < kids.length; i++) {
    var c = kids[i];
    if (c == null || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else el.appendChild(c.nodeType ? c : document.createTextNode(String(c)));
  }
}
function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
function $(id) { return document.getElementById(id); }

/* ── utils ── */
function rand(a, b) { return a + Math.random() * (b - a); }
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function pad2(n) { return String(n).padStart(2, '0'); }
function seriesNoise(n, base, amp, seed) {
  seed = seed || 1; var out = [], v = base;
  for (var i = 0; i < n; i++) {
    v += (Math.sin((i + seed) * 0.7) + Math.sin((i + seed) * 0.23)) * amp * 0.18;
    v += rand(-amp, amp) * 0.5; out.push(v);
  }
  return out;
}
function fmtClock(d) { return pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds()); }
function fmtDate(d) { return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' }); }
function fmtUptime(s) {
  var hh = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return hh + 'h ' + pad2(m) + 'm ' + pad2(sec) + 's';
}
function fmtSec(s) {
  if (s == null || isNaN(s)) return '—';
  s = Math.max(0, Math.round(s)); var m = Math.floor(s / 60), r = s % 60;
  return m ? (m + 'm ' + r + 's') : (r + 's');
}

/* ── simulated data (Analytics / AI Insights only) ── */
var FISH_COLORS = ['#00e5cc', '#2ad4ff', '#57f5b6', '#4aa8ff', '#ffc857', '#ff9ecb', '#b388ff', '#7CFFCB'];
function idColor(id) { return FISH_COLORS[Math.abs(parseInt(id, 10) || 0) % FISH_COLORS.length]; }

var FISH = [
  { id: 7,  name: 'Tank Sinatra', species: 'Angelfish',      color: '#00e5cc', conf: 0.97, active: 4218, status: 'alive' },
  { id: 3,  name: 'Finn Diesel',  species: 'Betta',          color: '#2ad4ff', conf: 0.94, active: 3902, status: 'alive' },
  { id: 12, name: 'Bubbles',      species: 'Guppy',          color: '#57f5b6', conf: 0.91, active: 5114, status: 'alive' },
  { id: 5,  name: 'Sushi',        species: 'Neon Tetra',     color: '#4aa8ff', conf: 0.89, active: 2740, status: 'alive' },
  { id: 9,  name: 'Gilly',        species: 'Molly',          color: '#ffc857', conf: 0.86, active: 1980, status: 'alive' },
  { id: 18, name: 'Marigold',     species: 'Cardinal Tetra', color: '#ff9ecb', conf: 0.82, active: 3320, status: 'alive' },
  { id: 21, name: 'Sir Swims',    species: 'Zebra Danio',    color: '#b388ff', conf: 0.79, active: 1240, status: 'idle' },
  { id: 2,  name: 'Coral',        species: 'Corydoras',      color: '#7CFFCB', conf: 0.74, active: 640,  status: 'idle' }
];
var HOURS = Array.from({ length: 24 }, function (_, i) { return i; });
function circadian(seed) {
  seed = seed || 0;
  return HOURS.map(function (hr) {
    var day = Math.exp(-Math.pow((hr - 13) / 5, 2)) * 70;
    var feed = Math.exp(-Math.pow((hr - 8) / 0.8, 2)) * 55 + Math.exp(-Math.pow((hr - 18) / 0.8, 2)) * 60;
    return clamp(day + feed + 10 + Math.sin(hr + seed) * 6, 4, 100);
  });
}
var ANALYTICS = {
  countOverTime: seriesNoise(48, 7.4, 1.1, 3).map(function (v) { return clamp(Math.round(v), 4, 9); }),
  activity: seriesNoise(48, 58, 22, 8).map(function (v) { return clamp(v, 8, 100); }),
  confidence: seriesNoise(48, 0.88, 0.05, 11).map(function (v) { return clamp(v, 0.6, 0.99); }),
  circadian: circadian(2),
  heatmap: Array.from({ length: 8 }, function (_, r) {
    return Array.from({ length: 12 }, function (_, c) {
      var d = Math.hypot(c - 6, r - 3.5);
      return clamp(Math.exp(-d * d / 9) * 100 + rand(-12, 18), 0, 100);
    });
  })
};
var INSIGHTS = [
  { id: 1, sev: 'alert', icon: 'alert', title: 'Reduced activity — Sir Swims',
    body: 'Track #21 has moved 61% less than its 7-day baseline over the last 3 hours. Could indicate stress or early illness. Recommend a closer look.',
    time: '14m ago', conf: 0.88, tag: 'Behavior' },
  { id: 2, sev: 'warn', icon: 'temp', title: 'GPU thermals trending up',
    body: 'Jetson GPU has held 71–74°C for 25 min. Still nominal, but consider checking enclosure airflow before the afternoon light cycle.',
    time: '38m ago', conf: 0.79, tag: 'System' },
  { id: 3, sev: 'good', icon: 'spark', title: 'Feeding response looks healthy',
    body: 'All 6 active fish converged to the upper-left feeding zone within 9s of the 08:00 feed window. Strong, synchronized response.',
    time: '6h ago', conf: 0.95, tag: 'Behavior' },
  { id: 4, sev: 'info', icon: 'eye', title: 'New individual detected',
    body: 'Track #2 ("Coral", Corydoras) appeared for the first time at 11:42. Likely a bottom-dweller that surfaced into frame.',
    time: '2h ago', conf: 0.74, tag: 'Detection' },
  { id: 5, sev: 'good', icon: 'model', title: 'Model v4 improved recall',
    body: 'Since switching to best.engine_v4, small-fish recall is up ~12% and ID switches dropped from 8/hr to 3/hr.',
    time: '1d ago', conf: 0.91, tag: 'Model' }
];
var SUGGESTED_Q = [
  'How active was the tank today?',
  'Is any fish behaving unusually?',
  'When were the fish most active?',
  'Summarize the last 24 hours'
];
var ASK_ANSWERS = {
  'how active was the tank today?': "Today's mean activity index is 64/100 — about 8% above the weekly average. Peak activity hit 94 around the 18:00 feed window. Bubbles (#12) logged the most movement (5,114px), while Coral (#2) stayed mostly in the lower-left substrate zone.",
  'is any fish behaving unusually?': "One flag: Sir Swims (#21) is 61% below its movement baseline over the last 3 hours and is holding near the surface-right corner. Everyone else is within normal range. I'd keep an eye on #21 through the next feed cycle.",
  'when were the fish most active?': "Two clear peaks, both feeding-driven: 08:00 (morning feed, index ~88) and 18:00 (evening feed, index ~94). Midday holds a gentle plateau around 60. Activity bottoms out 02:00–05:00 during the dark cycle.",
  'summarize the last 24 hours': "6–8 fish tracked continuously at 22.4 FPS avg. Two healthy feeding responses, no ID losses for >30min. One behavioral flag (#21 low activity). System nominal: GPU peaked 74°C, model best.engine_v4. Overall tank health score: 92/100 — Thriving."
};

/* ── global app state ── */
var App = {
  page: 'live',
  timers: [],          // per-view intervals, cleared on nav
  observers: [],       // per-view ResizeObservers
  viewLive: null,      // current view's live-update callback
  conn: true,
  live: {
    fps: 0, active: 0, total: 0, frame: 0,
    cpu: 56, gpu: 70, gpuUtil: 80, ram: 5.4,
    realTemps: false, model: '—', resolution: '1080p', fish: {}
  }
};
function every(ms, fn) { var id = setInterval(fn, ms); App.timers.push(id); return id; }
function observe(el, cb) {
  var ro = new ResizeObserver(function (es) { cb((es[0].contentRect.width) || 600); });
  ro.observe(el); App.observers.push(ro);
  requestAnimationFrame(function () { cb(el.clientWidth || 600); });
  return ro;
}
function clearViewState() {
  App.timers.forEach(clearInterval); App.timers = [];
  App.observers.forEach(function (o) { o.disconnect(); }); App.observers = [];
  App.viewLive = null;
}

/* ============================================================
   Charts (neon SVG)
   ============================================================ */
function linePath(vals, w, hgt, pad, lo, hi) {
  pad = pad || 0;
  lo = lo == null ? Math.min.apply(null, vals) : lo;
  hi = hi == null ? Math.max.apply(null, vals) : hi;
  var span = (hi - lo) || 1, iW = w - pad * 2, iH = hgt - pad * 2;
  return vals.map(function (v, i) {
    var x = pad + (i / (vals.length - 1)) * iW;
    var y = pad + iH - ((v - lo) / span) * iH;
    return (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
  }).join(' ');
}
var _gid = 0;
function gid(p) { return p + (++_gid); }

function Sparkline(opts) {
  var vals = opts.vals, color = opts.color || 'var(--teal)';
  var w = opts.w || 120, hgt = opts.h || 34, fill = opts.fill !== false;
  var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
  var p = linePath(vals, w, hgt, 3, lo, hi);
  var area = p + ' L' + (w - 3) + ',' + (hgt - 3) + ' L3,' + (hgt - 3) + ' Z';
  var id = gid('sg');
  return h('svg', { width: w, height: hgt, viewBox: '0 0 ' + w + ' ' + hgt, style: { display: 'block' } },
    h('defs', null, h('linearGradient', { id: id, x1: '0', y1: '0', x2: '0', y2: '1' },
      h('stop', { offset: '0', 'stop-color': color, 'stop-opacity': '0.32' }),
      h('stop', { offset: '1', 'stop-color': color, 'stop-opacity': '0' }))),
    fill && h('path', { d: area, fill: 'url(#' + id + ')' }),
    h('path', { d: p, fill: 'none', stroke: color, 'stroke-width': '1.8', 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
}

function AreaChart(opts) {
  var height = opts.height || 200, color = opts.color || 'var(--teal)';
  var wrap = h('div', { style: { width: '100%', overflow: 'hidden' } });
  var cur = opts.vals.slice();
  function build(w) {
    var padL = 38, padB = 22, padT = 10, padR = 8, hgt = height;
    var lo = opts.min == null ? Math.min.apply(null, cur) : opts.min;
    var hi = opts.max == null ? Math.max.apply(null, cur) : opts.max;
    var span = (hi - lo) || 1, iW = w - padL - padR, iH = hgt - padT - padB;
    var pts = cur.map(function (v, i) {
      return [padL + (i / (cur.length - 1)) * iW, padT + iH - ((v - lo) / span) * iH];
    });
    var line = pts.map(function (p, i) { return (i ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join(' ');
    var area = line + ' L' + (padL + iW) + ',' + (padT + iH) + ' L' + padL + ',' + (padT + iH) + ' Z';
    var id = gid('ag'), ticks = 4, last = pts[pts.length - 1];
    var grid = [];
    for (var i = 0; i <= ticks; i++) {
      var y = padT + (i / ticks) * iH, val = hi - (i / ticks) * span;
      grid.push(h('g', null,
        h('line', { x1: padL, y1: y, x2: w - padR, y2: y, stroke: 'var(--border)', 'stroke-width': '1', 'stroke-dasharray': '2 4', opacity: '0.6' }),
        h('text', { x: padL - 7, y: y + 3, 'text-anchor': 'end', 'font-size': '9', 'font-family': 'var(--mono)', fill: 'var(--faint)' },
          opts.fmtY ? opts.fmtY(val) : Math.round(val))));
    }
    var svg = h('svg', { width: w, height: hgt, viewBox: '0 0 ' + w + ' ' + hgt, style: { display: 'block' } },
      h('defs', null, h('linearGradient', { id: id, x1: '0', y1: '0', x2: '0', y2: '1' },
        h('stop', { offset: '0', 'stop-color': color, 'stop-opacity': '0.30' }),
        h('stop', { offset: '1', 'stop-color': color, 'stop-opacity': '0' }))),
      grid,
      h('path', { d: area, fill: 'url(#' + id + ')' }),
      h('path', { d: line, fill: 'none', stroke: color, 'stroke-width': '2', 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }),
      opts.live && h('circle', { cx: last[0], cy: last[1], r: '3.5', fill: color },
        h('animate', { attributeName: 'opacity', values: '1;.3;1', dur: '1.4s', repeatCount: 'indefinite' })));
    clear(wrap); wrap.appendChild(svg);
  }
  observe(wrap, build);
  return { node: wrap, update: function (v) { cur = v; build(wrap.clientWidth || 600); } };
}

function BarChart(opts) {
  var height = opts.height || 180, color = opts.color || 'var(--cyan)', highlight = opts.highlight || [];
  var wrap = h('div', { style: { width: '100%', overflow: 'hidden' } });
  var vals = opts.vals;
  function build(w) {
    var padB = 18, padT = 6, padL = 4, padR = 4, hgt = height;
    var max = Math.max.apply(null, vals) || 1, iW = w - padL - padR, iH = hgt - padT - padB, bw = iW / vals.length;
    var bars = vals.map(function (v, i) {
      var bh = (v / max) * iH, x = padL + i * bw, y = padT + iH - bh, hot = highlight.indexOf(i) >= 0;
      return h('g', null,
        h('rect', { x: x + bw * 0.18, y: y, width: bw * 0.64, height: bh, rx: '2',
          fill: hot ? 'var(--teal)' : color, opacity: hot ? 1 : 0.55,
          style: hot ? { filter: 'drop-shadow(0 0 5px var(--teal))' } : null }),
        (opts.labels && i % 3 === 0) && h('text', { x: x + bw / 2, y: hgt - 5, 'text-anchor': 'middle',
          'font-size': '8.5', 'font-family': 'var(--mono)', fill: 'var(--faint)' }, opts.labels[i]));
    });
    var svg = h('svg', { width: w, height: hgt, viewBox: '0 0 ' + w + ' ' + hgt, style: { display: 'block' } }, bars);
    clear(wrap); wrap.appendChild(svg);
  }
  observe(wrap, build);
  return { node: wrap };
}

function Ring(opts) {
  var value = opts.value, max = opts.max || 100, size = opts.size || 132, stroke = opts.stroke || 11;
  var color = opts.color || 'var(--teal)', track = opts.track || '#0e2228';
  var r = (size - stroke) / 2, c = 2 * Math.PI * r, off = c - (clamp(value, 0, max) / max) * c;
  var box = h('div', { style: { position: 'relative', width: size + 'px', height: size + 'px' } },
    h('svg', { width: size, height: size, style: { transform: 'rotate(-90deg)' } },
      h('circle', { cx: size / 2, cy: size / 2, r: r, fill: 'none', stroke: track, 'stroke-width': stroke }),
      h('circle', { cx: size / 2, cy: size / 2, r: r, fill: 'none', stroke: color, 'stroke-width': stroke,
        'stroke-dasharray': c, 'stroke-dashoffset': off, 'stroke-linecap': 'round',
        style: { transition: 'stroke-dashoffset .8s cubic-bezier(.4,0,.2,1)', filter: 'drop-shadow(0 0 6px ' + color + ')' } })),
    h('div', { style: { position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', textAlign: 'center' } }, opts.children));
  return box;
}

function Gauge(opts) {
  var value = opts.value, max = opts.max || 100, color = opts.color || 'var(--teal)', size = opts.size || 150;
  var w = size, hgt = size * 0.62, sw = 10, r = (w - sw) / 2, cx = w / 2, cy = hgt - 4;
  var startA = Math.PI, endA = 0, pct = clamp(value / max, 0, 1), ang = startA + (endA - startA) * pct;
  function arc(a0, a1) {
    var x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0);
    var x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
    var large = Math.abs(a1 - a0) > Math.PI ? 1 : 0;
    return 'M' + x0 + ',' + y0 + ' A' + r + ',' + r + ' 0 ' + large + ' 1 ' + x1 + ',' + y1;
  }
  var nx = cx + r * Math.cos(ang), ny = cy + r * Math.sin(ang);
  return h('div', { style: { textAlign: 'center' } },
    h('svg', { width: w, height: hgt + 6, viewBox: '0 0 ' + w + ' ' + (hgt + 6) },
      h('path', { d: arc(startA, endA), fill: 'none', stroke: '#0e2228', 'stroke-width': sw, 'stroke-linecap': 'round' }),
      h('path', { d: arc(startA, ang), fill: 'none', stroke: color, 'stroke-width': sw, 'stroke-linecap': 'round',
        style: { filter: 'drop-shadow(0 0 5px ' + color + ')', transition: 'all .6s' } }),
      h('circle', { cx: nx, cy: ny, r: '4.5', fill: '#04141a', stroke: color, 'stroke-width': '2' })),
    h('div', { style: { marginTop: '-6px' } },
      h('div', { class: 'tabular', style: { fontSize: '24px', fontWeight: 600, color: 'var(--text)' } },
        value, h('span', { style: { fontSize: '12px', color: 'var(--dim)' } }, opts.unit || '')),
      h('div', { class: 'eyebrow', style: { marginTop: '2px' } }, opts.label)));
}

function Heatmap(opts) {
  var grid = opts.grid, color = opts.color || '0,229,204';
  var wrap = h('div', { style: { width: '100%', overflow: 'hidden' } });
  function build(w) {
    var rows = grid.length, cols = grid[0].length, cell = w / cols, hgt = cell * rows;
    var rects = [];
    grid.forEach(function (row, r) {
      row.forEach(function (v, c) {
        rects.push(h('rect', { x: c * cell + 1.5, y: r * cell + 1.5, width: cell - 3, height: cell - 3, rx: '2',
          fill: 'rgba(' + color + ',' + (v / 100 * 0.85 + 0.04).toFixed(2) + ')' }));
      });
    });
    var svg = h('svg', { width: w, height: hgt, viewBox: '0 0 ' + w + ' ' + hgt, style: { display: 'block', borderRadius: '8px' } }, rects);
    clear(wrap); wrap.appendChild(svg);
  }
  observe(wrap, build);
  return wrap;
}

/* ── shared bits ── */
function eyebrow(text, style) { return h('div', { class: 'eyebrow', style: style }, text); }
function confBar(pct, color, width) {
  return h('div', { class: 'conf-bar', style: width ? { width: width + 'px' } : null },
    h('i', { style: { width: pct + '%', background: color } }));
}
function chip(text, kind) { return h('span', { class: 'chip' + (kind ? ' ' + kind : '') }, text); }
function liveDot(cls) { return h('span', { class: 'dot live' + (cls ? ' ' + cls : '') }); }

/* ============================================================
   Nav + shell
   ============================================================ */
var ICONS = {
  live: 'M2 7a2 2 0 012-2h9a2 2 0 012 2v10a2 2 0 01-2 2H4a2 2 0 01-2-2z M15 9l5-3v12l-5-3',
  chart: 'M3 3v18h18 M7 14l3-4 3 3 5-7',
  ai: 'M12 3a4 4 0 014 4v0a4 4 0 010 8 4 4 0 11-8 0 4 4 0 010-8v0a4 4 0 014-4z M12 7v8 M8.5 11h7',
  train: 'M4 7h16 M4 12h16 M4 17h10 M18 15l3 2-3 2',
  device: 'M5 4h14a1 1 0 011 1v11a1 1 0 01-1 1H5a1 1 0 01-1-1V5a1 1 0 011-1z M8 21h8 M12 17v4',
  gear: 'M12 9a3 3 0 100 6 3 3 0 000-6z M19.4 13a7.9 7.9 0 000-2l2-1.5-2-3.5-2.4 1a8 8 0 00-1.7-1L14 0h-4l-.3 2.5a8 8 0 00-1.7 1l-2.4-1-2 3.5L3.6 11a7.9 7.9 0 000 2l-2 1.5 2 3.5 2.4-1a8 8 0 001.7 1L10 24h4l.3-2.5a8 8 0 001.7-1l2.4 1 2-3.5z'
};
function navIcon(name, transform) {
  return h('svg', { class: 'ni-icon', viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor',
    'stroke-width': '1.7', 'stroke-linecap': 'round', 'stroke-linejoin': 'round' },
    h('path', { d: ICONS[name], transform: transform }));
}
var NAV = [
  { group: 'Monitor', items: [
    { id: 'live', label: 'Live Feed', icon: 'live', badge: 'LIVE' },
    { id: 'analytics', label: 'Analytics', icon: 'chart' },
    { id: 'insights', label: 'AI Insights', icon: 'ai', badge: '1', alert: true }
  ] },
  { group: 'Training', items: [{ id: 'training', label: 'Label & Train', icon: 'train' }] },
  { group: 'System', items: [
    { id: 'system', label: 'Device Health', icon: 'device' },
    { id: 'settings', label: 'Settings', icon: 'gear' }
  ] }
];
var TITLES = {
  live: ['Live Feed', 'Real-time detection & tracking'],
  analytics: ['Analytics', 'Behavioral & detection metrics'],
  insights: ['AI Insights', 'Anomalies, health & natural-language'],
  training: ['Label & Train', 'Active-learning data loop'],
  system: ['Device Health', 'Jetson Orin Nano telemetry'],
  settings: ['Settings', 'Pipeline & camera configuration']
};

function buildSidebar() {
  var devVals = {};
  function bar(v, c) { var i = h('i', { style: { width: clamp(v, 0, 100) + '%', background: c, boxShadow: '0 0 6px ' + c } }); return { node: h('div', { class: 'dev-bar' }, i), i: i }; }
  var gpuV = h('span', { class: 'val' }), ramV = h('span', { class: 'val' }), fpsV = h('span', { class: 'val', style: { color: 'var(--teal)' } });
  var gpuBar = bar(0, 'var(--teal)'), ramBar = bar(0, 'var(--cyan)');
  var connEl = h('span', null, 'Connecting…');
  App.updateSidebar = function () {
    var L = App.live;
    gpuV.textContent = L.gpuUtil + '% · ' + Math.round(L.gpu) + '°C';
    gpuBar.i.style.width = clamp(L.gpuUtil, 0, 100) + '%';
    ramV.textContent = L.ram.toFixed(1) + '/8GB';
    ramBar.i.style.width = clamp(L.ram / 8 * 100, 0, 100) + '%';
    fpsV.textContent = L.fps.toFixed(1);
    connEl.textContent = App.conn ? 'Connected · live' : 'Reconnecting…';
  };
  var nav = NAV.map(function (g) {
    return h('div', { class: 'nav-group' },
      h('div', { class: 'nav-label' }, g.group),
      g.items.map(function (it) {
        return h('div', { class: 'nav-item' + (App.page === it.id ? ' active' : ''), 'data-nav': it.id,
          onclick: function () { go(it.id); closeNav(); } },
          navIcon(it.icon),
          h('span', null, it.label),
          it.badge && h('span', { class: 'nav-badge' + (it.alert ? ' alert' : '') }, it.badge));
      }));
  });
  var aside = h('aside', { class: 'sidebar', id: 'sidebar' },
    h('div', { class: 'brand' },
      h('div', { class: 'brand-mark' },
        h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'var(--teal)', 'stroke-width': '1.7', 'stroke-linejoin': 'round' },
          h('ellipse', { cx: '13', cy: '12', rx: '7', ry: '4.5', fill: 'rgba(0,229,204,.12)' }),
          h('path', { d: 'M6 12L2 8.5v7z' }),
          h('circle', { cx: '16', cy: '11', r: '1', fill: 'var(--teal)', stroke: 'none' }))),
      h('div', null,
        h('div', { class: 'brand-name', html: 'Aqua<b>Scope</b>' }),
        h('div', { class: 'brand-sub' }, 'Neural Monitor'))),
    nav,
    h('div', { class: 'nav-spacer' }),
    h('div', { class: 'dev-card' },
      h('div', { class: 'dev-row' }, h('span', { class: 'lbl' }, 'GPU'), gpuV), gpuBar.node,
      h('div', { class: 'dev-row', style: { marginTop: '9px' } }, h('span', { class: 'lbl' }, 'RAM'), ramV), ramBar.node,
      h('div', { class: 'dev-row', style: { marginTop: '9px' } }, h('span', { class: 'lbl' }, 'FPS'), fpsV)),
    h('div', { class: 'conn' }, liveDot(), connEl));
  App.updateSidebar();
  return aside;
}

function buildTopbar() {
  var clock = h('div', { class: 'clock' });
  var titleEl = h('div', { class: 'tb-title' }), subEl = h('div', { class: 'tb-sub' });
  var pillSlot = h('span', { class: 'tb-spacer' });
  App.updateTopbar = function () {
    var t = TITLES[App.page]; titleEl.textContent = t[0]; subEl.textContent = t[1];
  };
  function tickClock() {
    var now = new Date();
    clear(clock); clock.appendChild(document.createTextNode(fmtClock(now)));
    clock.appendChild(h('div', { class: 'date' }, fmtDate(now)));
  }
  setInterval(tickClock, 1000); tickClock();
  App.updateTopbar();
  return h('header', { class: 'topbar' },
    h('button', { class: 'btn btn-icon hamburger', onclick: openNav },
      h('svg', { width: '18', height: '18', viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '2' },
        h('path', { d: 'M3 6h18M3 12h18M3 18h18' }))),
    h('div', null, titleEl, subEl),
    App.page === 'live' && h('span', { class: 'live-pill' }, liveDot(), 'LIVE · CAM_01'),
    pillSlot,
    h('div', { class: 'tb-right' }, clock));
}

function openNav() { var s = $('sidebar'); if (s) s.classList.add('open'); $('scrim').classList.add('show'); }
function closeNav() { var s = $('sidebar'); if (s) s.classList.remove('open'); $('scrim').classList.remove('show'); }

var VIEWS;
function go(page) {
  if (!VIEWS[page]) return;
  clearViewState();
  App.page = page;
  // refresh sidebar active + topbar
  document.querySelectorAll('[data-nav]').forEach(function (n) {
    n.classList.toggle('active', n.getAttribute('data-nav') === page);
  });
  document.querySelectorAll('[data-mb]').forEach(function (n) {
    n.classList.toggle('on', n.getAttribute('data-mb') === page);
  });
  // rebuild topbar (live pill depends on page)
  var main = $('main-col');
  var oldTop = main.querySelector('.topbar');
  var newTop = buildTopbar();
  main.replaceChild(newTop, oldTop);
  var content = $('content');
  clear(content);
  content.appendChild(VIEWS[page]());
}

/* ============================================================
   Global live polling (drives sidebar + live view + system)
   ============================================================ */
function startLivePolling() {
  function poll() {
    fetch('/stats').then(function (r) { return r.json(); }).then(function (d) {
      App.conn = true;
      var L = App.live;
      L.fps = d.fps != null ? d.fps : 0;
      L.active = d.active != null ? d.active : 0;
      L.total = d.total_ids != null ? d.total_ids : 0;
      L.frame = d.frame != null ? d.frame : 0;
      L.fish = d.fish || {};
      if (d.resolution) L.resolution = d.resolution;
      if (d.model) L.model = d.model;
      var temps = d.temps_c || {};
      var hasT = false;
      if (temps.CPU != null) { L.cpu = temps.CPU; hasT = true; }
      if (temps.GPU != null) { L.gpu = temps.GPU; hasT = true; }
      L.realTemps = hasT;
      if (App.updateSidebar) App.updateSidebar();
      if (App.viewLive) App.viewLive();
    }).catch(function () {
      App.conn = false;
      if (App.updateSidebar) App.updateSidebar();
    });
  }
  // simulate util/ram (no backend) + temp fallback when off-Jetson
  function walk() {
    var L = App.live;
    L.gpuUtil = Math.round(clamp(L.gpuUtil + rand(-4, 4), 62, 96));
    L.ram = clamp(L.ram + rand(-0.18, 0.18), 4.8, 6.4);
    if (!L.realTemps) {
      L.gpu = clamp(L.gpu + rand(-1.6, 1.6), 64, 78);
      L.cpu = clamp(L.cpu + rand(-1.4, 1.4), 50, 66);
    }
    if (App.updateSidebar) App.updateSidebar();
  }
  setInterval(poll, 1000); poll();
  setInterval(walk, 1500);
}

/* ============================================================
   LIVE VIEW — wired to real /stream, /stats, snapshots, controls
   ============================================================ */
var STREAM_LIMIT_MS = 180000;

function MiniKpi(opts) {
  var valEl = h('span', null, opts.value);
  var node = h('div', { class: 'card', style: { padding: '12px 14px' } },
    h('div', { class: 'kpi-label' }, opts.label),
    h('div', { style: { display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', marginTop: '8px' } },
      h('div', { class: 'tabular', style: { fontSize: '24px', fontWeight: 600, color: opts.color || 'var(--text)' } },
        valEl, opts.unit && h('span', { style: { fontSize: '12px', color: 'var(--dim)', marginLeft: '3px' } }, opts.unit)),
      opts.sparkWrap));
  return { node: node, setValue: function (v) { valEl.textContent = v; } };
}

function LiveView() {
  var streamExpired = false;
  // feed
  var feed = h('img', { id: 'feed', src: '/stream', alt: 'live', style: { width: '100%', height: '100%', objectFit: 'contain', display: 'block' } });
  feed.onerror = function () {
    if (streamExpired) return;
    setTimeout(function () { if (!streamExpired) feed.src = '/stream?t=' + Date.now(); }, 2000);
  };
  var modelTag = h('span', { class: 'feed-tag mono' }, App.live.model || 'model');
  var expiredOverlay = h('div', { style: { display: 'none', position: 'absolute', inset: 0, zIndex: 6, background: 'rgba(2,6,9,.82)', backdropFilter: 'blur(2px)', alignItems: 'center', justifyContent: 'center' } },
    h('div', { class: 'card pad', style: { textAlign: 'center', maxWidth: '280px' } },
      h('div', { style: { fontSize: '15px', fontWeight: 700, color: 'var(--teal)', marginBottom: '6px' } }, '⏸ Stream paused'),
      h('div', { style: { color: 'var(--dim)', fontSize: '11px', marginBottom: '14px', lineHeight: 1.5 } }, '3-minute session limit reached. Refresh to keep streaming.'),
      h('button', { class: 'btn btn-primary', onclick: function () { location.reload(); } }, '↻ Refresh')));
  var pausedOverlay = h('div', { style: { display: 'none', position: 'absolute', inset: 0, zIndex: 5, background: 'rgba(2,6,9,.78)', alignItems: 'center', justifyContent: 'center' } },
    h('div', { class: 'card pad', style: { textAlign: 'center', maxWidth: '300px' } },
      h('div', { style: { fontSize: '15px', fontWeight: 700, color: 'var(--teal)', marginBottom: '6px' } }, '⏸ Inference paused'),
      h('div', { style: { color: 'var(--dim)', fontSize: '11px', lineHeight: 1.5 } }, 'Training is using the GPU. The live feed resumes automatically when training completes.')));
  App.setFeedPaused = function (p) { pausedOverlay.style.display = p ? 'flex' : 'none'; };

  var feedWrap = h('div', { class: 'feed-wrap', style: { cursor: 'pointer' } },
    feed,
    h('span', { class: 'bracket tl' }), h('span', { class: 'bracket tr' }), h('span', { class: 'bracket bl' }), h('span', { class: 'bracket br' }),
    h('div', { class: 'scanlines' }),
    h('div', { class: 'feed-top' }, h('span', { class: 'rec' }, h('i', null), 'REC'), modelTag),
    pausedOverlay, expiredOverlay);
  feedWrap.addEventListener('click', function () { if (!streamExpired) openFullscreen(); });
  setTimeout(function () {
    streamExpired = true; feed.src = ''; expiredOverlay.style.display = 'flex';
  }, STREAM_LIMIT_MS);

  // snapshot button on the feed
  var snapBtn = h('button', { class: 'snap-btn', onclick: function (e) { e.stopPropagation(); takeSnap(snapBtn); } },
    h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '2' },
      h('circle', { cx: '12', cy: '13', r: '4' }), h('path', { d: 'M5 7h2l1-2h8l1 2h2a1 1 0 011 1v9a1 1 0 01-1 1H5a1 1 0 01-1-1V8a1 1 0 011-1z' })),
    'Snapshot');
  feedWrap.appendChild(snapBtn);

  // filmstrip
  var filmRow = h('div', { class: 'filmstrip' });
  var filmCount = h('span', { class: 'mono', style: { marginLeft: 'auto', color: 'var(--teal)', fontSize: '11px' } }, '0');
  var filmEmpty = h('div', { class: 'film-cap mono', style: { padding: '18px 4px', color: 'var(--dim)' } }, 'No snapshots yet — press 📷');
  filmRow.appendChild(filmEmpty);
  var snapState = { list: [] };
  App.snapState = snapState; App.filmRow = filmRow; App.filmCount = filmCount; App.filmEmpty = filmEmpty;

  // right-rail KPIs with rolling sparklines
  var fpsRoll = Array.from({ length: 40 }, function () { return App.live.fps || 22; });
  var actRoll = Array.from({ length: 40 }, function () { return App.live.active || 6; });
  var fpsSpark = h('span', null), actSpark = h('span', null);
  function redrawSparks() {
    clear(fpsSpark); fpsSpark.appendChild(Sparkline({ vals: fpsRoll, color: 'var(--teal)', w: 72, h: 30 }));
    clear(actSpark); actSpark.appendChild(Sparkline({ vals: actRoll, color: 'var(--cyan)', w: 72, h: 30 }));
  }
  redrawSparks();
  var kFps = MiniKpi({ label: 'FPS', value: '0.0', color: 'var(--teal)', sparkWrap: fpsSpark });
  var kAct = MiniKpi({ label: 'Active', value: '0', color: 'var(--cyan)', sparkWrap: actSpark });
  var kTot = MiniKpi({ label: 'Tracked IDs', value: '0', color: 'var(--aqua)' });
  var kGpu = MiniKpi({ label: 'GPU', value: '0', unit: '°C', color: 'var(--warn)' });

  var fishList = h('div', { class: 'fish-list' });
  var liveChip = chip('', 'on'); liveChip.appendChild(liveDot());
  var liveChipText = document.createTextNode('0 live'); liveChip.appendChild(liveChipText);

  function renderFish() {
    var fish = App.live.fish || {};
    var entries = Object.keys(fish).map(function (id) { return [id, fish[id]]; });
    entries.sort(function (a, b) { return (b[1].last_seen_ts || 0) - (a[1].last_seen_ts || 0); });
    var now = Date.now() / 1000;
    clear(fishList);
    if (!entries.length) {
      fishList.appendChild(h('div', { class: 'fish-meta mono', style: { padding: '14px 6px' } }, 'No fish detected yet.'));
      return;
    }
    entries.slice(0, 12).forEach(function (pair) {
      var id = pair[0], f = pair[1], col = idColor(id);
      var alive = (now - (f.last_seen_ts || 0)) < 2;
      fishList.appendChild(h('div', { class: 'fish-row' },
        h('span', { class: 'fish-swatch', style: { background: col, boxShadow: '0 0 8px ' + col, opacity: alive ? 1 : 0.4 } }),
        h('div', { style: { minWidth: 0 } },
          h('div', { class: 'fish-name' }, 'Fish ', h('span', { class: 'mono', style: { color: 'var(--dim)', fontSize: '11px' } }, '#' + id)),
          h('div', { class: 'fish-meta mono' }, Math.round(f.total_distance_px || 0).toLocaleString() + 'px · ' + (f.frame_count || 0) + ' frames')),
        h('span', { class: 'fish-conf mono' }, alive ? 'live' : 'idle')));
    });
  }

  App.viewLive = function () {
    var L = App.live;
    kFps.setValue(L.fps.toFixed(1)); kAct.setValue(L.active); kTot.setValue(L.total); kGpu.setValue(Math.round(L.gpu));
    modelTag.textContent = L.model || 'model';
    fpsRoll = fpsRoll.slice(1).concat(L.fps || 0);
    actRoll = actRoll.slice(1).concat(L.active || 0);
    redrawSparks();
    var live = L.active;
    liveChipText.textContent = live + ' live';
    renderFish();
  };

  var rail = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '14px' } },
    h('div', { class: 'grid cols-2', style: { gap: '12px' } }, kFps.node, kAct.node, kTot.node, kGpu.node),
    h('div', { class: 'card', style: { flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 } },
      h('div', { class: 'card-h' }, h('span', { class: 'ch-title' }, 'Fish Activity'), h('span', { class: 'ch-spacer' }), liveChip),
      fishList),
    FeedControls());

  // initial paint
  App.viewLive();
  loadSnapshots();
  every(2000, loadSnapshots);

  return h('div', { class: 'view live-grid' },
    h('div', null,
      feedWrap,
      h('div', { class: 'card', style: { marginTop: '14px', padding: '12px' } },
        h('div', { style: { display: 'flex', alignItems: 'center', marginBottom: '9px' } }, eyebrow('Snapshots'), filmCount),
        filmRow)),
    rail);
}

function FeedControls() {
  // real toggle states (server defaults: trails OFF, enhance ON, hats OFF)
  var state = { trails: false, enhance: true, hats: false };
  function toggleRow(key, label, endpoint, respKey) {
    var tog = h('div', { class: 'toggle' + (state[key] ? ' on' : '') }, h('i', null));
    var lbl = h('span', { style: { color: state[key] ? 'var(--text)' : 'var(--dim)' } }, label);
    var row = h('div', { class: 'ctl-row', onclick: function () {
      fetch(endpoint).then(function (r) { return r.json(); }).then(function (d) {
        state[key] = !!d[respKey];
        tog.classList.toggle('on', state[key]);
        lbl.style.color = state[key] ? 'var(--text)' : 'var(--dim)';
      });
    } }, lbl, tog);
    return row;
  }
  var confVal = h('span', { class: 'tabular', style: { color: 'var(--teal)', fontSize: '12px', width: '32px' } }, '35%');
  var confDebounce = null;
  var confSlider = h('input', { class: 'rng', type: 'range', min: '5', max: '95', step: '5', value: '35',
    oninput: function (e) {
      var pct = parseInt(e.target.value, 10); confVal.textContent = pct + '%';
      clearTimeout(confDebounce);
      confDebounce = setTimeout(function () { fetch('/conf?v=' + (pct / 100).toFixed(2)); }, 120);
    } });
  var modelSel = h('select', { class: 'sel', onchange: function (e) { fetch('/model?v=' + encodeURIComponent(e.target.value)); } },
    h('option', null, 'loading…'));
  loadModelOptions(modelSel);
  var resSel = h('select', { class: 'sel', value: App.live.resolution, onchange: function (e) { fetch('/resolution?v=' + encodeURIComponent(e.target.value)); } },
    ['480p', '720p', '1080p'].map(function (r) { return h('option', { value: r, selected: r === App.live.resolution }, r); }));

  return h('div', { class: 'card', style: { padding: '14px' } },
    eyebrow('Detection Controls', { marginBottom: '12px' }),
    h('div', { style: { display: 'flex', flexDirection: 'column', gap: '4px' } },
      toggleRow('trails', 'Motion trails', '/trails', 'trails'),
      toggleRow('enhance', 'Image enhance', '/enhance', 'enhance'),
      toggleRow('hats', 'Party hats 🎉', '/hat', 'hat')),
    h('div', { style: { height: '1px', background: 'var(--border)', margin: '13px 0' } }),
    eyebrow('Confidence', { marginBottom: '8px' }),
    h('div', { style: { display: 'flex', alignItems: 'center', gap: '10px' } }, confSlider, confVal),
    eyebrow('Model', { margin: '13px 0 6px' }), modelSel,
    eyebrow('Resolution', { margin: '13px 0 6px' }), resSel,
    h('button', { class: 'btn btn-ghost', style: { width: '100%', marginTop: '13px', justifyContent: 'center', color: 'var(--alert)', borderColor: 'rgba(255,111,111,.3)' },
      onclick: function () { fetch('/reset'); } }, '↺ Reset Trails'));
}

function loadModelOptions(sel) {
  fetch('/models').then(function (r) { return r.json(); }).then(function (d) {
    clear(sel);
    var models = d.models || [];
    if (!models.length) { sel.appendChild(h('option', { disabled: true }, '(no models found)')); return; }
    var curBase = (d.current || '').split('/').pop();
    models.forEach(function (m) {
      var base = m.split('/').pop();
      sel.appendChild(h('option', { value: base, selected: base === curBase }, base));
    });
  }).catch(function () {});
}

/* snapshots */
function takeSnap(btn) {
  btn.disabled = true;
  fetch('/screenshot').then(function (r) { return r.json(); }).then(function (d) {
    if (d.filename) {
      var s = { filename: d.filename, label: 'Snap ' + d.filename.slice(9, 15) };
      App.snapState.list.unshift(s); addThumb(s, true);
    }
    setTimeout(function () { btn.disabled = false; }, 800);
  }).catch(function () { btn.disabled = false; });
}
function addThumb(s, prepend) {
  if (App.filmEmpty && App.filmEmpty.parentNode) App.filmEmpty.parentNode.removeChild(App.filmEmpty);
  var url = '/screenshots/' + s.filename;
  var thumb = h('div', { class: 'film-thumb', 'data-file': s.filename, style: { cursor: 'pointer' } },
    h('div', { class: 'film-img', style: { background: '#04161d' } }, h('img', { src: url, loading: 'lazy', style: { width: '100%', height: '100%', objectFit: 'cover' } })),
    h('span', { class: 'film-cap mono' }, s.label));
  thumb.onclick = function () { openModal(url); };
  if (prepend && App.filmRow.firstChild) App.filmRow.insertBefore(thumb, App.filmRow.firstChild);
  else App.filmRow.appendChild(thumb);
  App.filmCount.textContent = App.snapState.list.length;
}
function loadSnapshots() {
  if (!App.snapState) return;
  fetch('/screenshots').then(function (r) { return r.json(); }).then(function (list) {
    var existing = new Set(App.snapState.list.map(function (s) { return s.filename; }));
    list.forEach(function (s) {
      if (!existing.has(s.filename)) { App.snapState.list.push(s); addThumb(s, false); }
    });
  }).catch(function () {});
}

/* modal lightbox + fullscreen feed */
function openModal(url) {
  var img = h('img', { src: url, style: { maxWidth: '90vw', maxHeight: '85vh', borderRadius: '8px', border: '1px solid var(--border)' } });
  var modal = h('div', { style: { position: 'fixed', inset: 0, background: 'rgba(2,6,9,.85)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' } },
    h('button', { style: { position: 'absolute', top: '20px', right: '24px', background: 'none', border: 'none', color: 'var(--text)', fontSize: '24px', cursor: 'pointer' }, onclick: function () { document.body.removeChild(modal); } }, '✕'),
    img);
  modal.addEventListener('click', function (e) { if (e.target === modal) document.body.removeChild(modal); });
  document.body.appendChild(modal);
}
function openFullscreen() {
  var img = h('img', { src: '/stream?t=' + Date.now(), style: { width: '100%', height: '100%', objectFit: 'contain' } });
  var overlay = h('div', { style: { position: 'fixed', inset: 0, background: '#000', zIndex: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' } },
    h('button', { style: { position: 'absolute', top: '14px', right: '14px', background: 'rgba(0,0,0,.65)', color: '#fff', border: '1px solid rgba(255,255,255,.25)', borderRadius: '50%', width: '38px', height: '38px', fontSize: '18px', cursor: 'pointer', zIndex: 301 },
      onclick: function () { close(); } }, '✕'),
    img);
  function close() {
    img.src = ''; if (overlay.parentNode) document.body.removeChild(overlay);
    if (document.fullscreenElement) document.exitFullscreen().catch(function () {});
  }
  overlay.addEventListener('click', function (e) { if (e.target === overlay) close(); });
  document.body.appendChild(overlay);
  if (overlay.requestFullscreen) overlay.requestFullscreen().catch(function () {});
}

/* ============================================================
   ANALYTICS VIEW (simulated)
   ============================================================ */
function KpiCard(opts) {
  return h('div', { class: 'card kpi' },
    h('div', { class: 'kpi-top' },
      h('span', { class: 'kpi-label' }, opts.label),
      opts.vals && Sparkline({ vals: opts.vals, color: opts.color || 'var(--teal)', w: 66, h: 26, fill: false })),
    h('div', { class: 'kpi-val' }, opts.value, opts.unit && h('span', { class: 'unit' }, opts.unit)),
    opts.delta && h('div', { class: 'kpi-delta ' + (opts.dir || 'flat') },
      (opts.dir === 'up' ? '▲' : opts.dir === 'down' ? '▼' : '–') + ' ' + opts.delta,
      h('span', { style: { color: 'var(--faint)' } }, ' vs 7d avg')));
}
function card(title, sub, body, padStyle) {
  return h('div', { class: 'card' },
    h('div', { class: 'card-h' }, h('span', { class: 'ch-title' }, title), h('span', { class: 'ch-spacer' }),
      sub === 'live' ? chip('live', 'on') : (sub && h('span', { class: 'ch-sub' }, sub))),
    h('div', { style: padStyle || { padding: '14px 14px 10px' } }, body));
}
function AnalyticsView() {
  var range = '24h';
  var countChart = AreaChart({ vals: ANALYTICS.countOverTime, color: 'var(--teal)', height: 210, min: 0, max: 10, live: true });
  var actChart = AreaChart({ vals: ANALYTICS.activity, color: 'var(--cyan)', height: 210, min: 0, max: 100, fmtY: function (v) { return Math.round(v); }, live: true });
  // live ticking of the two area charts
  every(1500, function () {
    ANALYTICS.countOverTime = ANALYTICS.countOverTime.slice(1).concat(clamp(Math.round(ANALYTICS.countOverTime[ANALYTICS.countOverTime.length - 1] + rand(-1, 1)), 4, 9));
    ANALYTICS.activity = ANALYTICS.activity.slice(1).concat(clamp(ANALYTICS.activity[ANALYTICS.activity.length - 1] + rand(-8, 8), 8, 100));
    countChart.update(ANALYTICS.countOverTime); actChart.update(ANALYTICS.activity);
  });
  var hourLabels = HOURS.map(pad2);
  var seg = h('div', { class: 'seg' }, ['6h', '24h', '7d', '30d'].map(function (r) {
    return h('button', { class: 'seg-btn' + (r === range ? ' on' : ''), onclick: function (e) {
      range = r; seg.querySelectorAll('.seg-btn').forEach(function (b) { b.classList.remove('on'); }); e.target.classList.add('on');
    } }, r);
  }));

  var liveChip = chip('live', 'on'); liveChip.insertBefore(liveDot(), liveChip.firstChild);
  var countCardEl = h('div', { class: 'card' },
    h('div', { class: 'card-h' }, h('span', { class: 'ch-title' }, 'Fish Count Over Time'), h('span', { class: 'ch-spacer' }), liveChip),
    h('div', { style: { padding: '14px 14px 10px' } }, countChart.node));

  return h('div', { class: 'view', style: { display: 'flex', flexDirection: 'column', gap: '16px' } },
    h('div', { class: 'row-between' },
      h('div', null,
        h('div', { class: 'section-title' }, 'Tank Analytics'),
        h('div', { class: 'mono', style: { color: 'var(--dim)', fontSize: '12px', marginTop: '2px' } }, 'Last 24 hours · auto-refreshing · simulated history')),
      seg),
    h('div', { class: 'grid cols-4' },
      KpiCard({ label: 'Avg Fish Count', value: '7.2', delta: '+0.4', dir: 'up', color: 'var(--teal)', vals: ANALYTICS.countOverTime.slice(-16) }),
      KpiCard({ label: 'Activity Index', value: '64', unit: '/100', delta: '+8%', dir: 'up', color: 'var(--cyan)', vals: ANALYTICS.activity.slice(-16) }),
      KpiCard({ label: 'Mean Confidence', value: '88', unit: '%', delta: '+2%', dir: 'up', color: 'var(--aqua)', vals: ANALYTICS.confidence.slice(-16).map(function (v) { return v * 100; }) }),
      KpiCard({ label: 'ID Switches / hr', value: '3.1', delta: '-5', dir: 'down', color: 'var(--good)', vals: seriesNoise(16, 4, 1.5, 4) })),
    h('div', { class: 'grid cols-2' },
      countCardEl,
      card('Activity Level', 'distance swum / interval', actChart.node)),
    h('div', { class: 'grid cols-2' },
      card('Circadian Rhythm', 'activity by hour · feeds ▮',
        BarChart({ vals: ANALYTICS.circadian, labels: hourLabels, color: 'var(--deep)', height: 200, highlight: [8, 18] }).node,
        { padding: '16px 14px 6px' }),
      card('Spatial Heatmap', 'where fish dwell',
        h('div', null, Heatmap({ grid: ANALYTICS.heatmap }),
          h('div', { class: 'heat-legend' },
            h('span', { class: 'mono' }, 'low'), h('div', { class: 'heat-bar' }), h('span', { class: 'mono' }, 'high'),
            h('span', { class: 'mono', style: { marginLeft: 'auto', color: 'var(--dim)' } }, 'upper-center = feeding zone'))),
        { padding: '16px' })),
    PerFishTable());
}
function PerFishTable() {
  return h('div', { class: 'card' },
    h('div', { class: 'card-h' }, h('span', { class: 'ch-title' }, 'Per-Fish Breakdown'), h('span', { class: 'ch-spacer' }), h('span', { class: 'ch-sub' }, FISH.length + ' individuals tracked')),
    h('div', { class: 'ptable' },
      h('div', { class: 'pt-head' },
        h('span', null, 'Individual'), h('span', null, 'Species'), h('span', null, 'Confidence'),
        h('span', null, 'Activity (px)'), h('span', null, 'Status'), h('span', null, '7-day trend')),
      FISH.map(function (f, i) {
        return h('div', { class: 'pt-row' },
          h('span', { class: 'pt-name' }, h('span', { class: 'fish-swatch', style: { background: f.color, boxShadow: '0 0 6px ' + f.color } }), f.name + ' ', h('em', { class: 'mono' }, '#' + f.id)),
          h('span', { class: 'mono dimc' }, f.species),
          h('span', { class: 'mono' }, confBar(f.conf * 100, f.color), Math.round(f.conf * 100) + '%'),
          h('span', { class: 'mono' }, f.active.toLocaleString()),
          h('span', null, chip(f.status === 'alive' ? 'active' : 'idle', f.status === 'alive' ? 'on' : '')),
          h('span', null, Sparkline({ vals: seriesNoise(20, f.active / 60, f.active / 220, i + 1), color: f.color, w: 110, h: 26 })));
      })));
}

/* ============================================================
   AI INSIGHTS VIEW (simulated)
   ============================================================ */
var SEV_COLOR = { alert: 'var(--alert)', warn: 'var(--warn)', good: 'var(--good)', info: 'var(--cyan)' };
function insightIconPaths(kind) {
  switch (kind) {
    case 'alert': return [h('path', { d: 'M12 3l9 16H3z M12 10v4 M12 17.5v.1' })];
    case 'temp': return [h('path', { d: 'M10 4a2 2 0 014 0v9a4 4 0 11-4 0z' })];
    case 'eye': return [h('circle', { cx: '12', cy: '12', r: '3' }), h('path', { d: 'M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z' })];
    case 'model': return [h('rect', { x: '4', y: '4', width: '16', height: '16', rx: '2' }), h('path', { d: 'M9 9h6v6H9z' })];
    default: return [h('path', { d: 'M12 3v4 M12 17v4 M3 12h4 M17 12h4 M6 6l2.5 2.5 M15.5 15.5L18 18' })];
  }
}
function insightIcon(kind, color) {
  return h('svg', { viewBox: '0 0 24 24', width: 16, height: 16, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, insightIconPaths(kind));
}
function aiOrb() { return h('span', { class: 'ai-orb' }); }

function TypeOut(text) {
  var span = h('span', null);
  var caret = h('span', { class: 'caret' }, '▍');
  var n = 0;
  span.appendChild(caret);
  var id = setInterval(function () {
    n += 2;
    if (n >= text.length) { clearInterval(id); span.textContent = text; return; }
    span.textContent = text.slice(0, n); span.appendChild(caret);
  }, 14);
  App.timers.push(id);
  return span;
}

function AskTank() {
  var body = h('div', { class: 'ask-body' });
  var log = [{ role: 'ai', text: "Hi! I'm watching your tank in real time. Ask me anything — activity, individual fish, feeding, or system health." }];
  function render() {
    clear(body);
    log.forEach(function (m, i) {
      var bubble = h('div', { class: 'bubble ' + m.role });
      if (m.role === 'ai' && i === log.length - 1 && i !== 0) bubble.appendChild(TypeOut(m.text));
      else bubble.textContent = m.text;
      body.appendChild(bubble);
    });
    body.scrollTop = body.scrollHeight;
  }
  function ask(text) {
    var key = text.trim().toLowerCase();
    var ans = ASK_ANSWERS[key] || "I analyzed the last 24h of tracking data. Activity and detection metrics are within normal range, with 6–8 fish tracked continuously. Try one of the suggested questions for a detailed read-out.";
    log.push({ role: 'user', text: text }); log.push({ role: 'ai', text: ans });
    render(); input.value = '';
  }
  var input = h('input', { placeholder: 'Ask about your fish…' });
  render();
  return h('div', { class: 'card ask-card' },
    h('div', { class: 'card-h' }, aiOrb(), h('span', { class: 'ch-title' }, 'Ask Your Tank'), h('span', { class: 'ch-spacer' }), h('span', { class: 'ch-sub' }, 'grounded in live detections')),
    body,
    h('div', { class: 'ask-sugg' }, SUGGESTED_Q.map(function (s) { return h('button', { class: 'sugg', onclick: function () { ask(s); } }, s); })),
    h('form', { class: 'ask-input', onsubmit: function (e) { e.preventDefault(); if (input.value.trim()) ask(input.value); } },
      input,
      h('button', { type: 'submit', class: 'btn btn-primary', style: { padding: '8px 14px' } },
        h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '2' }, h('path', { d: 'M22 2L11 13 M22 2l-7 20-4-9-9-4z' })))));
}

function InsightsView() {
  var health = 92;
  var sub = [
    { k: 'Activity', v: 88, c: 'var(--teal)' }, { k: 'Detection', v: 95, c: 'var(--cyan)' },
    { k: 'Behavior', v: 84, c: 'var(--aqua)' }, { k: 'System', v: 90, c: 'var(--deep)' }
  ];
  var healthCard = h('div', { class: 'card health-card' },
    h('span', { class: 'bracket tl' }), h('span', { class: 'bracket br' }),
    eyebrow('Tank Health', { marginBottom: '4px' }),
    Ring({ value: health, size: 148, stroke: 12, color: 'var(--teal)', children: h('div', null,
      h('div', { class: 'tabular', style: { fontSize: '40px', fontWeight: 700, lineHeight: 1, color: 'var(--teal)' } }, health),
      eyebrow('Thriving', { marginTop: '2px' })) }),
    h('div', { class: 'health-sub' }, sub.map(function (s) {
      return h('div', { class: 'hs-row' },
        h('span', { class: 'mono dimc' }, s.k),
        h('div', { class: 'hs-bar' }, h('i', { style: { width: s.v + '%', background: s.c } })),
        h('span', { class: 'mono', style: { color: s.c } }, s.v));
    })));
  var summaryCard = h('div', { class: 'card summary-card' },
    h('div', { class: 'card-h' }, aiOrb(), h('span', { class: 'ch-title' }, 'Daily AI Summary'), h('span', { class: 'ch-spacer' }), h('span', { class: 'ch-sub' }, 'generated 06:00')),
    h('div', { class: 'summary-body' },
      h('p', null, h('b', { style: { color: 'var(--teal)' } }, 'Your tank had a calm, healthy day.'), ' ' + ASK_ANSWERS['summarize the last 24 hours']),
      h('div', { class: 'summary-tags' },
        chip('2 feeding responses', 'on'), chip('0 ID losses >30m'), chip('1 behavior flag', 'warn'), chip('92 health', 'on'))));
  var feedList = h('div', { class: 'feed-list' }, INSIGHTS.map(function (it) {
    return h('div', { class: 'insight ' + it.sev },
      h('div', { class: 'ins-icon', style: { borderColor: SEV_COLOR[it.sev] } }, insightIcon(it.icon, SEV_COLOR[it.sev])),
      h('div', { style: { minWidth: 0, flex: 1 } },
        h('div', { class: 'ins-top' },
          h('span', { class: 'ins-title' }, it.title), h('span', { class: 'ins-tag mono' }, it.tag), h('span', { class: 'ins-time mono' }, it.time)),
        h('div', { class: 'ins-body' }, it.body),
        h('div', { class: 'ins-conf' },
          h('span', { class: 'mono dimc' }, 'AI confidence'),
          confBar(it.conf * 100, SEV_COLOR[it.sev], 80),
          h('span', { class: 'mono', style: { color: SEV_COLOR[it.sev] } }, Math.round(it.conf * 100) + '%'))));
  }));

  return h('div', { class: 'view insights-grid' },
    h('div', { style: { display: 'flex', flexDirection: 'column', gap: '16px', minWidth: 0 } },
      h('div', { class: 'grid cols-2', style: { gridTemplateColumns: '260px 1fr' } }, healthCard, summaryCard),
      h('div', { class: 'card', style: { flex: 1 } },
        h('div', { class: 'card-h' }, h('span', { class: 'ch-title' }, 'Insight Feed'), h('span', { class: 'ch-spacer' }), chip('1 needs attention', 'alert')),
        feedList)),
    AskTank());
}

/* ============================================================
   LABEL & TRAIN VIEW — wired to real label/train endpoints
   ============================================================ */
function TrainingView() {
  var current = null;
  var capturing = false;

  // ---- triage crop area ----
  var triageBody = h('div', { class: 'triage-body' });
  function renderTriage() {
    clear(triageBody);
    if (!current) {
      triageBody.appendChild(h('div', { class: 'triage-empty' }, 'Queue empty — enable capture to collect detections from the live feed.'));
      return;
    }
    var c = current, col = idColor(c.track_id);
    var bb = c.bbox, w = c.img_w || 1, hh = c.img_h || 1;
    var box = h('div', { class: 'crop-box', style: {
      left: (bb[0] / w * 100) + '%', top: (bb[1] / hh * 100) + '%',
      width: ((bb[2] - bb[0]) / w * 100) + '%', height: ((bb[3] - bb[1]) / hh * 100) + '%',
      borderColor: col, boxShadow: '0 0 10px ' + col + '55' } },
      h('span', { class: 'crop-label mono', style: { background: col } }, c.class_name + ' #' + c.track_id));
    var frame = h('div', { class: 'crop-frame' },
      h('img', { src: c.image_url + '?t=' + Date.now(), style: { position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' } }),
      box,
      h('span', { class: 'crop-conf mono' }, c.class_name));
    triageBody.appendChild(frame);
    triageBody.appendChild(h('div', { class: 'triage-q' }, 'Is this a fish?'));
    triageBody.appendChild(h('div', { class: 'triage-actions' },
      h('button', { class: 'tri-btn reject', onclick: function () { decide(0); } },
        h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '2.5' }, h('path', { d: 'M6 6l12 12M18 6L6 18' })),
        'No ', h('em', { class: 'mono' }, 'n')),
      h('button', { class: 'tri-btn accept', onclick: function () { decide(1); } },
        h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '2.5' }, h('path', { d: 'M4 12l5 5L20 6' })),
        'Yes ', h('em', { class: 'mono' }, 'y'))));
  }
  function decide(keep) {
    if (!current) return;
    var cid = current.id; current = null;
    fetch('/label/decision?id=' + encodeURIComponent(cid) + '&keep=' + keep)
      .then(function (r) { return r.json(); }).then(function () { pollQueue(); refreshLabels(); });
  }
  var queueCountEl = h('span', { class: 'tabular' }, '0');
  function pollQueue() {
    fetch('/label/queue').then(function (r) { return r.json(); }).then(function (d) {
      queueCountEl.textContent = d.count || 0;
      if (!d.count) { current = null; renderTriage(); return; }
      var next = d.queue[0];
      if (current && current.id === next.id) return;
      current = next; renderTriage();
    }).catch(function () {});
  }

  var capToggle = h('div', { class: 'toggle', onclick: function () {
    fetch('/label/toggle').then(function (r) { return r.json(); }).then(function (d) {
      capturing = !!d.enabled; capToggle.classList.toggle('on', capturing); capLabel.textContent = capturing ? 'capturing' : 'paused';
      capLabel.style.color = capturing ? 'var(--teal)' : 'var(--dim)';
    });
  } }, h('i', null));
  var capLabel = h('span', { class: 'mono', style: { fontSize: '11px', color: 'var(--dim)' } }, 'paused');
  // sync capture state on load
  fetch('/label/state').then(function (r) { return r.json(); }).then(function (d) {
    capturing = !!d.enabled; capToggle.classList.toggle('on', capturing);
    capLabel.textContent = capturing ? 'capturing' : 'paused'; capLabel.style.color = capturing ? 'var(--teal)' : 'var(--dim)';
  }).catch(function () {});

  // ---- dataset progress + train ----
  var need = 100, saved = 0, estimate = null;
  var savedEl = h('div', { class: 'tabular', style: { fontSize: '34px', fontWeight: 700 } }, '0', h('span', { style: { fontSize: '15px', color: 'var(--dim)' } }, '/' + need));
  var progBar = h('i', { style: { width: '0%' } });
  var estEl = h('span', { class: 'tabular' }, '~16m');
  var trainBtn = h('button', { class: 'btn btn-primary', disabled: true, style: { width: '100%', marginTop: '14px', justifyContent: 'center', opacity: 0.45 }, onclick: confirmTraining },
    h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '2' }, h('path', { d: 'M12 2a4 4 0 014 4v1a4 4 0 010 8 4 4 0 11-8 0 4 4 0 010-8V6a4 4 0 014-4z' })),
    'Need labels');
  function refreshLabels() {
    fetch('/train/labels').then(function (r) { return r.json(); }).then(function (d) {
      saved = d.count; need = d.min_required; estimate = d.estimate;
      clear(savedEl); savedEl.appendChild(document.createTextNode(saved));
      savedEl.appendChild(h('span', { style: { fontSize: '15px', color: 'var(--dim)' } }, '/' + need));
      savedEl.style.color = d.ready ? 'var(--teal)' : 'var(--text)';
      progBar.style.width = Math.min(saved / need * 100, 100) + '%';
      if (d.estimate) estEl.textContent = '~' + Math.round((d.estimate.low_min + d.estimate.high_min) / 2) + 'm';
      trainBtn.disabled = !d.ready || App.trainModalOpen;
      trainBtn.style.opacity = (d.ready && !App.trainModalOpen) ? 1 : 0.45;
      clear(trainBtn); trainBtn.appendChild(h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '2' }, h('path', { d: 'M12 2a4 4 0 014 4v1a4 4 0 010 8 4 4 0 11-8 0 4 4 0 010-8V6a4 4 0 014-4z' })));
      trainBtn.appendChild(document.createTextNode(d.ready ? 'Train new model' : ('Need ' + Math.max(0, need - saved) + ' more labels')));
    }).catch(function () {});
  }
  function confirmTraining() {
    if (!estimate) return;
    var e = estimate;
    var ok = window.confirm('Train a new model on your labeled data?\n\n' +
      '• Estimated time: ~' + e.low_min + '–' + e.high_min + ' minutes (' + e.epochs + ' epochs)\n' +
      '• Inference will pause while the GPU is in use\n' +
      '• On success a new models/best.engine_v<N> appears in the model list\n\nContinue?');
    if (!ok) return;
    fetch('/train/start').then(function (r) { return r.json(); }).then(function (d) {
      if (d.error) { alert('Could not start training: ' + d.error); return; }
      openTrainModal(); startTrainPolling();
    });
  }

  // models list (real /models)
  var modelList = h('div', { class: 'model-list' });
  var modelCountEl = h('span', { class: 'ch-sub' }, '0');
  function loadModels() {
    fetch('/models').then(function (r) { return r.json(); }).then(function (d) {
      clear(modelList);
      var models = d.models || [], curBase = (d.current || '').split('/').pop();
      modelCountEl.textContent = models.length;
      if (!models.length) { modelList.appendChild(h('div', { class: 'model-meta mono', style: { padding: '10px' } }, 'No model files found.')); return; }
      models.forEach(function (m) {
        var base = m.split('/').pop(), active = base === curBase;
        modelList.appendChild(h('div', { class: 'model-row' + (active ? ' active' : ''), style: { cursor: 'pointer' },
          onclick: function () { fetch('/model?v=' + encodeURIComponent(base)).then(loadModels); } },
          h('div', { style: { minWidth: 0 } },
            h('div', { class: 'model-name mono' }, base, active && chip('active', 'on')),
            h('div', { class: 'model-meta mono' }, m)),
          active && h('div', { class: 'model-stats' }, h('div', null, h('span', { class: 'kpi-label' }, 'in use'), h('span', { class: 'tabular', style: { color: 'var(--teal)' } }, '●')))));
      });
    }).catch(function () {});
  }

  // ---- triage polling + shortcuts ----
  pollQueue(); refreshLabels(); loadModels();
  every(1000, pollQueue);
  every(3000, refreshLabels);
  var keyHandler = function (e) {
    if (App.page !== 'training' || App.trainModalOpen) return;
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
    if (e.key === 'y' || e.key === 'Y' || e.key === 'ArrowRight') decide(1);
    if (e.key === 'n' || e.key === 'N' || e.key === 'ArrowLeft') decide(0);
  };
  document.addEventListener('keydown', keyHandler);
  App.observers.push({ disconnect: function () { document.removeEventListener('keydown', keyHandler); } });

  // resume an in-progress run if user reloaded mid-train
  fetch('/train/status').then(function (r) { return r.json(); }).then(function (s) {
    if (s.running || s.state === 'training' || s.state === 'starting' || s.state === 'exporting') { openTrainModal(); startTrainPolling(); }
  }).catch(function () {});

  return h('div', { class: 'view training-grid' },
    h('div', { style: { display: 'flex', flexDirection: 'column', gap: '16px', minWidth: 0 } },
      h('div', { class: 'card' },
        h('div', { class: 'card-h' },
          h('span', { class: 'ch-title' }, 'Active Learning · Label Triage'), h('span', { class: 'ch-spacer' }), capToggle, capLabel),
        triageBody)),
    h('div', { style: { display: 'flex', flexDirection: 'column', gap: '16px' } },
      h('div', { class: 'card pad' },
        eyebrow('Dataset Progress', { marginBottom: '12px' }),
        h('div', { class: 'row-between', style: { alignItems: 'flex-end' } }, savedEl, h('span', { class: 'mono', style: { fontSize: '11px', color: 'var(--dim)' } }, 'labels to retrain')),
        h('div', { class: 'tm-bar', style: { marginTop: '10px' } }, progBar),
        h('div', { class: 'grid cols-2', style: { gap: '10px', marginTop: '14px' } },
          h('div', { class: 'mini-stat' }, h('span', { class: 'kpi-label' }, 'In queue'), queueCountEl),
          h('div', { class: 'mini-stat' }, h('span', { class: 'kpi-label' }, 'Est. train'), estEl)),
        trainBtn,
        h('div', { class: 'mono', style: { fontSize: '10px', color: 'var(--faint)', marginTop: '8px', textAlign: 'center' } }, 'Tip: press y / n to triage fast')),
      h('div', { class: 'card' },
        h('div', { class: 'card-h' }, h('span', { class: 'ch-title' }, 'Model Versions'), h('span', { class: 'ch-spacer' }), modelCountEl),
        modelList)));
}

/* ---- real training modal (status/log polling) ---- */
function openTrainModal() {
  if (App.trainModal) return;
  App.trainModalOpen = true;
  var stateLine = h('div', { class: 'mono', style: { color: 'var(--text-2)', fontSize: '13px' } }, 'starting…');
  var bar = h('i', { style: { width: '0%' } });
  var epochEl = h('div', { class: 'tabular tm-big' }, '—');
  var elapsedEl = h('div', { class: 'tabular tm-big' }, '—');
  var etaEl = h('div', { class: 'tabular tm-big' }, '—');
  var verEl = h('div', { class: 'tabular tm-big' }, '—');
  var statusChip = chip('GPU in use · inference paused', 'warn');
  var logBox = h('div', { class: 'tm-log', style: { height: '180px' } }, 'waiting for training subprocess output…');
  var titleEl = h('span', { class: 'ch-title' }, 'Training model');
  var cancelBtn = h('button', { class: 'btn btn-ghost', style: { color: 'var(--alert)', borderColor: 'rgba(255,111,111,.4)' }, onclick: cancelTraining }, 'Cancel');
  var closeBtn = h('button', { class: 'btn btn-primary', style: { display: 'none' }, onclick: closeTrainModal }, 'Close & resume');
  var actions = h('div', { style: { display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '14px' } }, cancelBtn, closeBtn);

  var modal = h('div', { class: 'modal-scrim', onclick: function (e) { if (e.target === modal && closeBtn.style.display !== 'none') closeTrainModal(); } },
    h('div', { class: 'card train-modal' },
      h('div', { class: 'card-h' }, aiOrb(), titleEl, h('span', { class: 'ch-spacer' }), statusChip),
      h('div', { style: { padding: '16px' } },
        h('div', { style: { marginBottom: '14px' } }, stateLine),
        h('div', { class: 'tm-stat-row' },
          h('div', null, h('div', { class: 'kpi-label' }, 'Epoch'), epochEl),
          h('div', null, h('div', { class: 'kpi-label' }, 'Elapsed'), elapsedEl),
          h('div', null, h('div', { class: 'kpi-label' }, 'ETA'), etaEl),
          h('div', null, h('div', { class: 'kpi-label' }, 'Version'), verEl)),
        h('div', { class: 'tm-bar' }, bar),
        h('div', { style: { margin: '14px 0' } }, logBox),
        actions)));
  document.body.appendChild(modal);
  App.trainModal = { modal: modal, stateLine: stateLine, bar: bar, epochEl: epochEl, elapsedEl: elapsedEl, etaEl: etaEl, verEl: verEl, statusChip: statusChip, logBox: logBox, titleEl: titleEl, cancelBtn: cancelBtn, closeBtn: closeBtn };
}
function startTrainPolling() {
  if (App.trainPoll) clearInterval(App.trainPoll);
  pollTrainStatus(); pollTrainLog();
  App.trainPoll = setInterval(function () { pollTrainStatus(); pollTrainLog(); }, 2000);
}
function pollTrainStatus() {
  fetch('/train/status').then(function (r) { return r.json(); }).then(updateTrainModal).catch(function () {});
}
function pollTrainLog() {
  var m = App.trainModal; if (!m) return;
  fetch('/train/log').then(function (r) { return r.json(); }).then(function (d) {
    var txt = (d && d.log) || '';
    if (!txt) { m.logBox.textContent = 'waiting for training subprocess output…'; return; }
    var pinned = (m.logBox.scrollHeight - m.logBox.clientHeight - m.logBox.scrollTop) < 24;
    m.logBox.textContent = txt;
    if (pinned) m.logBox.scrollTop = m.logBox.scrollHeight;
  }).catch(function () {});
}
function updateTrainModal(s) {
  var m = App.trainModal; if (!m) return;
  var cur = s.current_epoch || 0, tot = s.total_epochs || 0, pct = tot > 0 ? Math.round(100 * cur / tot) : 0;
  m.bar.style.width = pct + '%';
  m.stateLine.textContent = (s.message || s.state || '…') + (tot ? ('   (' + pct + '%)') : '');
  clear(m.epochEl); m.epochEl.appendChild(document.createTextNode(cur || '—'));
  if (tot) m.epochEl.appendChild(h('span', { style: { color: 'var(--dim)', fontSize: '14px' } }, '/' + tot));
  m.elapsedEl.textContent = fmtSec(s.elapsed_sec);
  m.etaEl.textContent = fmtSec(s.eta_sec);
  m.verEl.textContent = s.version != null ? ('v' + s.version) : '—';
  var running = s.running || s.state === 'training' || s.state === 'starting' || s.state === 'exporting';
  if (App.setFeedPaused) App.setFeedPaused(running);
  if (s.state === 'done') {
    m.titleEl.textContent = 'Training complete';
    clear(m.statusChip); m.statusChip.className = 'chip on'; m.statusChip.textContent = 'done';
    m.stateLine.innerHTML = '✓ Saved <b>' + ((s.engine_path || '').split('/').pop() || 'engine') + '</b>. Click Close to resume inference with the new model.';
    m.cancelBtn.style.display = 'none'; m.closeBtn.style.display = '';
    if (App.trainPoll) { clearInterval(App.trainPoll); App.trainPoll = null; }
    if (s.latest_engine) { var base = s.latest_engine.split('/').pop(); fetch('/model?v=' + encodeURIComponent(base)); }
  } else if (s.state === 'failed') {
    m.titleEl.textContent = 'Training failed';
    clear(m.statusChip); m.statusChip.className = 'chip alert'; m.statusChip.textContent = 'failed';
    m.stateLine.textContent = '✗ ' + (s.message || 'unknown error');
    m.cancelBtn.style.display = 'none'; m.closeBtn.style.display = '';
    if (App.trainPoll) { clearInterval(App.trainPoll); App.trainPoll = null; }
  }
}
function cancelTraining() {
  if (!confirm('Cancel training? Progress will be lost.')) return;
  fetch('/train/cancel').then(function (r) { return r.json(); }).catch(function () {});
}
function closeTrainModal() {
  fetch('/train/acknowledge').catch(function () {});
  if (App.trainPoll) { clearInterval(App.trainPoll); App.trainPoll = null; }
  if (App.trainModal && App.trainModal.modal.parentNode) document.body.removeChild(App.trainModal.modal);
  App.trainModal = null; App.trainModalOpen = false;
  if (App.setFeedPaused) App.setFeedPaused(false);
}

/* ============================================================
   DEVICE HEALTH VIEW
   ============================================================ */
function SystemView() {
  var uptime = 6 * 3600 + 41 * 60;
  var gpuHist = Array.from({ length: 48 }, function () { return App.live.gpu; });
  var cpuHist = Array.from({ length: 48 }, function () { return App.live.cpu; });
  function tempColor(t) { return t >= 75 ? 'var(--alert)' : t >= 68 ? 'var(--warn)' : 'var(--good)'; }

  var gaugeWrap = h('div', { class: 'grid cols-4' });
  var gpuChart = AreaChart({ vals: gpuHist, color: 'var(--warn)', height: 170, min: 40, max: 90, fmtY: function (v) { return v + '°'; }, live: true });
  var cpuChart = AreaChart({ vals: cpuHist, color: 'var(--deep)', height: 170, min: 40, max: 90, fmtY: function (v) { return v + '°'; }, live: true });
  var infBody = h('div', null), memBody = h('div', null);
  var uptimeEl = h('span', { class: 'tabular', style: { color: 'var(--teal)' } }, fmtUptime(uptime));
  var engineEl = h('span', { class: 'tabular' }, App.live.model || '—');
  var tempNote = h('span', { class: 'ch-sub' }, App.live.realTemps ? 'last 48s · throttle @ 87°C' : 'simulated (no telemetry)');

  function renderGauges() {
    clear(gaugeWrap);
    var L = App.live;
    [gauge(L.cpu, 'CPU Temp', '°C', tempColor(L.cpu)),
     gauge(L.gpu, 'GPU Temp', '°C', tempColor(L.gpu)),
     gauge(L.gpuUtil, 'GPU Util', '%', 'var(--teal)'),
     gauge(Math.round(L.ram / 8 * 100), 'RAM', '%', 'var(--cyan)')].forEach(function (g) {
      gaugeWrap.appendChild(h('div', { class: 'card gauge-card' }, g));
    });
  }
  function gauge(v, label, unit, color) { return Gauge({ value: Math.round(v), max: 100, label: label, unit: unit, color: color }); }
  function kvRows(parent, rows) {
    clear(parent);
    rows.forEach(function (r) {
      parent.appendChild(h('div', { class: 'kv-row' }, h('span', { class: 'mono dimc' }, r[0]), h('span', { class: 'tabular', style: { color: r[2] || 'var(--text)' } }, r[1])));
    });
  }
  function renderPanels() {
    var L = App.live;
    kvRows(infBody, [
      ['Pipeline FPS', L.fps.toFixed(1), 'var(--teal)'],
      ['Latency / frame', (L.fps ? (1000 / L.fps).toFixed(1) : '—') + 'ms', null],
      ['Tracked IDs', String(L.total), null],
      ['Detections / s', String(Math.round(L.fps * L.active)), null]]);
    kvRows(memBody, [
      ['RAM used', L.ram.toFixed(1) + ' / 8 GB', null],
      ['Swap', '0.2 / 4 GB', null],
      ['GPU shared', '2.1 GB', null],
      ['Model VRAM', '0.9 GB', null]]);
    engineEl.textContent = L.model || '—';
  }

  App.viewLive = function () {
    renderGauges(); renderPanels();
    gpuHist = gpuHist.slice(1).concat(App.live.gpu); cpuHist = cpuHist.slice(1).concat(App.live.cpu);
    gpuChart.update(gpuHist); cpuChart.update(cpuHist);
  };
  every(1000, function () { uptime += 1; uptimeEl.textContent = fmtUptime(uptime); });
  renderGauges(); renderPanels();

  return h('div', { class: 'view', style: { display: 'flex', flexDirection: 'column', gap: '16px' } },
    h('div', { class: 'card pad device-banner' },
      h('span', { class: 'bracket tl' }), h('span', { class: 'bracket tr' }), h('span', { class: 'bracket bl' }), h('span', { class: 'bracket br' }),
      h('div', { class: 'dev-icon' }, h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'var(--teal)', 'stroke-width': '1.6' },
        h('rect', { x: '4', y: '4', width: '16', height: '16', rx: '2' }), h('rect', { x: '8', y: '8', width: '8', height: '8', rx: '1' }),
        h('path', { d: 'M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3' }))),
      h('div', { style: { flex: 1, minWidth: 0 } },
        h('div', { style: { fontSize: '17px', fontWeight: 600 } }, 'Jetson Orin Nano 8GB'),
        h('div', { class: 'mono', style: { color: 'var(--dim)', fontSize: '12px', marginTop: '3px' } }, 'JetPack 6.1 · 15W / MAXN · CUDA 12.2 · TensorRT 10.3')),
      h('div', { class: 'db-stats' },
        h('div', null, h('span', { class: 'kpi-label' }, 'Uptime'), uptimeEl),
        h('div', null, h('span', { class: 'kpi-label' }, 'Power mode'), h('span', { class: 'tabular' }, 'MAXN')),
        h('div', null, h('span', { class: 'kpi-label' }, 'Engine'), engineEl))),
    gaugeWrap,
    h('div', { class: 'grid cols-2' },
      h('div', { class: 'card' },
        h('div', { class: 'card-h' }, h('span', { class: 'ch-title' }, 'GPU Temperature'), h('span', { class: 'ch-spacer' }), tempNote),
        h('div', { style: { padding: '14px 14px 10px' } }, gpuChart.node)),
      h('div', { class: 'card' },
        h('div', { class: 'card-h' }, h('span', { class: 'ch-title' }, 'CPU Temperature'), h('span', { class: 'ch-spacer' }), h('span', { class: 'ch-sub' }, '6-core Arm Cortex-A78AE')),
        h('div', { style: { padding: '14px 14px 10px' } }, cpuChart.node))),
    h('div', { class: 'grid cols-3' },
      h('div', { class: 'card pad' }, eyebrow('Inference', { marginBottom: '12px' }), infBody),
      h('div', { class: 'card pad' }, eyebrow('Memory', { marginBottom: '12px' }), memBody),
      h('div', { class: 'card pad' }, eyebrow('Capture', { marginBottom: '12px' }),
        (function () { var b = h('div', null); kvRows(b, [['Camera', 'Logitech C920'], ['Driver', 'V4L2 /dev/video0'], ['Resolution', App.live.resolution], ['Exposure', 'auto']]); return b; })())));
}

/* ============================================================
   SETTINGS VIEW (toggles wired where a backend exists)
   ============================================================ */
function SettingsView() {
  var s = { enhance: true, trails: false, sahi: false, record: true, public: false, hats: false, quality: 75, exposure: 'auto', interval: 60 };
  function settingsRow(label, sub, control) {
    return h('div', { class: 'set-row' },
      h('div', null, h('div', { class: 'set-label' }, label), sub && h('div', { class: 'set-sub mono' }, sub)),
      h('div', { class: 'set-control' }, control));
  }
  // toggle wired to a real endpoint (endpoint optional → local only)
  function toggle(key, endpoint, respKey) {
    var tog = h('div', { class: 'toggle' + (s[key] ? ' on' : '') }, h('i', null));
    tog.onclick = function () {
      if (endpoint) {
        fetch(endpoint).then(function (r) { return r.json(); }).then(function (d) { s[key] = !!d[respKey]; tog.classList.toggle('on', s[key]); });
      } else { s[key] = !s[key]; tog.classList.toggle('on', s[key]); }
    };
    return tog;
  }
  var qVal = h('span', { class: 'tabular', style: { color: 'var(--teal)', width: '26px' } }, s.quality);
  var iVal = h('span', { class: 'tabular', style: { color: 'var(--teal)', width: '36px' } }, s.interval + 's');

  return h('div', { class: 'view', style: { display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '760px' } },
    h('div', { class: 'card' },
      h('div', { class: 'card-h' }, h('span', { class: 'ch-title' }, 'Pipeline')),
      h('div', { style: { padding: '4px 16px 8px' } },
        settingsRow('Image enhancement', 'CLAHE + white-balance on each frame', toggle('enhance', '/enhance', 'enhance')),
        settingsRow('Motion trails', 'render fish paths on the feed', toggle('trails', '/trails', 'trails')),
        settingsRow('SAHI sliced inference', 'better small-fish recall · lowers FPS (display only)', toggle('sahi')),
        settingsRow('Stream quality', 'JPEG quality · lower = less bandwidth (display only)',
          h('div', { style: { display: 'flex', alignItems: 'center', gap: '10px', width: '180px' } },
            h('input', { class: 'rng', type: 'range', min: '40', max: '95', value: s.quality, oninput: function (e) { s.quality = +e.target.value; qVal.textContent = s.quality; } }), qVal)))),
    h('div', { class: 'card' },
      h('div', { class: 'card-h' }, h('span', { class: 'ch-title' }, 'Camera')),
      h('div', { style: { padding: '4px 16px 8px' } },
        settingsRow('Exposure', null,
          h('select', { class: 'sel', style: { width: '160px' }, onchange: function (e) { s.exposure = e.target.value; } },
            h('option', { value: 'auto' }, 'Auto'), h('option', { value: '-6' }, 'Manual −6 (dim)'), h('option', { value: '-4' }, 'Manual −4'), h('option', { value: '-2' }, 'Manual −2'))),
        settingsRow('Record to disk', 'recording_YYYYMMDD.mp4 (display only)', toggle('record')))),
    h('div', { class: 'card' },
      h('div', { class: 'card-h' }, h('span', { class: 'ch-title' }, 'Logging & Sharing')),
      h('div', { style: { padding: '4px 16px 8px' } },
        settingsRow('Stats log interval', 'JSON snapshot to fish_logs/ (display only)',
          h('div', { style: { display: 'flex', alignItems: 'center', gap: '10px', width: '180px' } },
            h('input', { class: 'rng', type: 'range', min: '15', max: '120', step: '15', value: s.interval, oninput: function (e) { s.interval = +e.target.value; iVal.textContent = s.interval + 's'; } }), iVal)),
        settingsRow('Public Cloudflare tunnel', 'expose dashboard via trycloudflare.com (display only)', toggle('public')),
        settingsRow('Party hats 🎉', 'purely for science', toggle('hats', '/hat', 'hat')))));
}

/* ============================================================
   Boot
   ============================================================ */
VIEWS = { live: LiveView, analytics: AnalyticsView, insights: InsightsView, training: TrainingView, system: SystemView, settings: SettingsView };

function mobileBar() {
  var items = [['live', 'live'], ['analytics', 'chart'], ['insights', 'ai'], ['training', 'train'], ['system', 'device']];
  return h('nav', { class: 'mobile-bar' }, items.map(function (it) {
    return h('button', { class: 'mb-item' + (App.page === it[0] ? ' on' : ''), 'data-mb': it[0], onclick: function () { go(it[0]); } }, navIcon(it[1]));
  }));
}

function boot() {
  var root = $('root');
  var scrim = h('div', { class: 'scrim', id: 'scrim', onclick: closeNav });
  var sidebar = buildSidebar();
  var mainCol = h('div', { class: 'main', id: 'main-col' }, buildTopbar(), h('div', { class: 'content', id: 'content' }));
  root.appendChild(h('div', { class: 'shell' }, scrim, sidebar, mainCol));
  root.appendChild(mobileBar());
  startLivePolling();
  // initial view
  $('content').appendChild(VIEWS[App.page]());
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();

})();
</script>
</body>
</html>
"""
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ── Public API ────────────────────────────────────────────
def start_stream(port: int) -> None:
    server = _ThreadingHTTPServer(("0.0.0.0", port), _MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def start_public_tunnel(port: int) -> None:
    def _run():
        try:
            proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            for raw in (proc.stdout or []):
                line = raw.decode(errors="ignore")
                match = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
                if match:
                    print(f"[STREAM] Public URL: {match.group(0)}")
                    break
            proc.wait()
        except FileNotFoundError:
            print("[STREAM] cloudflared not found — install it:")
            print("  wget https://github.com/cloudflare/cloudflared/releases/latest"
                  "/download/cloudflared-linux-arm64 -O cloudflared")
            print("  chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/")

    threading.Thread(target=_run, daemon=True).start()
