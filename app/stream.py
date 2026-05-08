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
_TRAIN_DEFAULT_EPOCHS: int = 5
_TRAIN_EPOCH_CHOICES: tuple[int, ...] = (5, 10, 15, 20, 30, 50)
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
    if epochs not in _TRAIN_EPOCH_CHOICES:
        return {"error": f"epochs must be one of {list(_TRAIN_EPOCH_CHOICES)}"}
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
                # Put the trainer in its own process group so cancel_training
                # can kill the YOLO dataloader workers along with the parent
                # via os.killpg — without start_new_session they'd share our
                # group and survive a single SIGTERM to the parent pid.
                start_new_session=True,
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
    """Tear down the trainer subprocess group asynchronously.

    Returns immediately so the dashboard's HTTP request doesn't hang for
    seconds while ultralytics shuts down. A background thread sends SIGTERM
    to the whole process group (catches YOLO dataloader workers, not just
    the parent), waits a few seconds, then escalates to SIGKILL. We also
    stamp a cancelling/failed status so the modal flips to "Close" on the
    very next poll instead of after the kill completes.
    """
    import signal as _signal

    global _train_proc
    with _train_lock:
        if _train_proc is None or _train_proc.poll() is not None:
            return {"error": "no training in progress"}
        proc = _train_proc
        pid = proc.pid

    # Optimistic status update so the UI flips immediately. The real subprocess
    # exit will overwrite this via train_jetson.py's except-handler if it's
    # still alive, or via get_train_status()'s rc-reconciliation otherwise.
    try:
        with open(_train_status_file, "w") as f:
            json.dump({
                "state": "failed",
                "message": "Cancelled by user",
            }, f)
    except OSError:
        pass

    def _reaper(p: subprocess.Popen, group_pid: int) -> None:
        global _train_proc, _train_log_fh
        try:
            os.killpg(group_pid, _signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(group_pid, _signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        with _train_lock:
            if _train_log_fh is not None:
                try:
                    _train_log_fh.close()
                except OSError:
                    pass
                _train_log_fh = None

    threading.Thread(target=_reaper, args=(proc, pid), daemon=True).start()
    return {"cancelled": True, "pid": pid}


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

    def _parse_epochs(self) -> int:
        """Read ?epochs=N from the query string, snapping to the allowed choices."""
        qs = self.path.split("?", 1)[1] if "?" in self.path else ""
        params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
        try:
            n = int(params.get("epochs", _TRAIN_DEFAULT_EPOCHS))
        except ValueError:
            n = _TRAIN_DEFAULT_EPOCHS
        return n if n in _TRAIN_EPOCH_CHOICES else _TRAIN_DEFAULT_EPOCHS

    def _serve_train_labels(self):
        count = count_user_labels()
        epochs = self._parse_epochs()
        body = json.dumps({
            "count": count,
            "min_required": _TRAIN_MIN_LABELS,
            "ready": count >= _TRAIN_MIN_LABELS,
            "epochs": epochs,
            "epoch_choices": list(_TRAIN_EPOCH_CHOICES),
            "default_epochs": _TRAIN_DEFAULT_EPOCHS,
            "estimate": estimate_training_minutes(count, epochs),
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
        epochs = self._parse_epochs()
        self._json_response(json.dumps(start_training(epochs=epochs)).encode())

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
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AquaScope</title>
<style>
:root {
  --bg:      #0e1117;
  --sidebar: #13181f;
  --card:    #181e27;
  --border:  #1e2d3d;
  --accent:  #f5c518;
  --teal:    #00d4aa;
  --blue:    #4fc3f7;
  --dim:     #4a5568;
  --text:    #c8d8e8;
  --danger:  #fc5c65;
  --font:    'Segoe UI', system-ui, sans-serif;
  --mono:    'Courier New', monospace;
}
*{margin:0;padding:0;box-sizing:border-box}
body{
  background:linear-gradient(160deg,#0b1520 0%,#0e1117 40%,#0a1a1a 100%);color:var(--text);
  font-family:var(--font);font-size:13px;
  display:grid;
  grid-template-columns: 200px 1fr;
  grid-template-rows: 52px 1fr 140px;
  grid-template-areas:
    "sidebar topbar"
    "sidebar main"
    "sidebar filmstrip";
  height:100vh;overflow:hidden;
}

/* ── Mobile ── */
@media (max-width: 700px) {
  html,body{ overflow-x:hidden;width:100% }
  body{
    display:flex;flex-direction:column;
    height:auto;min-height:100vh;
    overflow-y:auto;overflow-x:hidden;
  }
  #sidebar{ display:none !important }
  #topbar{
    padding:0 10px;flex-shrink:0;
    position:sticky;top:0;z-index:20;
  }
  .page-title{ font-size:13px }
  .live-badge{ padding:2px 7px;font-size:10px }
  #snap-btn{ padding:5px 10px;font-size:11px }
  #uptime{ display:none }
  #main{
    display:flex !important;flex-direction:column !important;
    grid-template-columns:unset !important;
    width:100%;padding:0;gap:0;flex-shrink:0;
  }
  #feed-wrap{
    width:100%;height:56vw;min-height:180px;
    border-radius:0;border-left:none;border-right:none;border-top:none;flex-shrink:0;
  }
  #stats-panel{
    display:flex;flex-direction:row;flex-wrap:wrap;
    gap:8px;padding:10px;width:100%;overflow:visible;flex-shrink:0;
  }
  #stats-panel .card{ flex:1 1 calc(50% - 8px);min-width:130px;max-width:100% }
  #stats-panel #fish-list{ max-height:160px }
  #filmstrip{
    width:100%;height:auto;min-height:100px;flex-shrink:0;
  }
  #snap-popover{ right:auto;left:0;width:calc(100vw - 20px);max-width:320px }
}

/* ── Sidebar ── */
#sidebar{
  grid-area:sidebar;
  background:linear-gradient(180deg,#161d27 0%,#0f1820 100%);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;
  padding:16px 0;
}
.logo{
  display:flex;align-items:center;gap:10px;
  padding:0 18px 20px;
  font-size:16px;font-weight:700;letter-spacing:1px;
  background:linear-gradient(90deg,#f5c518,#00d4aa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.logo-icon{font-size:22px}
.nav-section{
  font-size:10px;letter-spacing:2px;color:var(--dim);
  padding:14px 18px 6px;text-transform:uppercase;
}
.nav-item{
  display:flex;align-items:center;gap:10px;
  padding:9px 18px;cursor:pointer;border-radius:0;
  color:var(--dim);transition:all 0.15s;font-size:13px;
  border-left:3px solid transparent;
}
.nav-item:hover{background:rgba(255,255,255,0.04);color:var(--text)}
.nav-item.active{color:var(--accent);border-left-color:var(--accent);background:linear-gradient(90deg,rgba(245,197,24,0.13) 0%,rgba(245,197,24,0.02) 100%)}
.nav-icon{font-size:15px;width:18px;text-align:center}
.nav-spacer{flex:1}
.status-dot{
  display:inline-block;width:7px;height:7px;border-radius:50%;
  background:var(--teal);margin-right:6px;
  animation:pulse 1.8s infinite;
}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.5;transform:scale(0.85)}}

/* ── Topbar ── */
#topbar{
  grid-area:topbar;
  background:linear-gradient(90deg,#161d27 0%,#131a23 100%);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 20px;
}
.topbar-left{display:flex;align-items:center;gap:14px}
.page-title{font-size:15px;font-weight:600;color:var(--text)}
.live-badge{
  background:linear-gradient(90deg,rgba(0,212,170,0.18) 0%,rgba(0,212,170,0.06) 100%);color:var(--teal);
  border:1px solid rgba(0,212,170,0.3);
  padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;
}
.topbar-right{display:flex;align-items:center;gap:12px}
#clock{color:var(--blue);font-family:var(--mono);font-size:13px}
#uptime{color:var(--dim);font-size:11px}
#snap-btn{
  background:linear-gradient(135deg,#f5c518 0%,#d4a017 100%);color:#111;border:none;
  padding:6px 14px;border-radius:6px;font-size:12px;font-weight:700;
  cursor:pointer;transition:opacity 0.15s;letter-spacing:0.5px;
}
#snap-btn:hover{opacity:0.85}
#snap-btn:active{opacity:0.7}
#snap-wrap{position:relative}
#snap-popover{
  display:none;
  position:absolute;top:calc(100% + 8px);right:0;
  width:340px;
  background:linear-gradient(145deg,#1a2130 0%,#141c26 100%);border:1px solid var(--border);border-radius:8px;
  padding:10px;z-index:50;
  box-shadow:0 8px 32px rgba(0,0,0,0.7);
}
#snap-wrap:focus-within #snap-popover,
#snap-wrap.open #snap-popover{display:block}
#snap-pop-row{
  display:flex;gap:6px;overflow-x:auto;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent;
  padding-bottom:4px;
}
#snap-pop-empty{color:var(--dim);font-size:11px;text-align:center;padding:10px 0}
.pop-thumb{
  flex-shrink:0;width:100px;height:56px;border-radius:5px;overflow:hidden;
  border:1px solid var(--border);cursor:pointer;position:relative;transition:border-color 0.15s;
}
.pop-thumb:hover{border-color:var(--accent)}
.pop-thumb img{width:100%;height:100%;object-fit:cover}
.pop-thumb .del{
  position:absolute;top:2px;right:2px;
  background:rgba(0,0,0,0.7);color:#fc5c65;border:none;
  border-radius:3px;font-size:10px;padding:1px 4px;cursor:pointer;display:none;
}
.pop-thumb:hover .del{display:block}

/* ── Main ── */
#main{
  grid-area:main;
  display:grid;
  grid-template-columns:1fr 200px;
  gap:10px;padding:10px;
  overflow:hidden;
}

/* ── Label panel (Training tab) ── */
#label-panel{
  grid-area:main;
  display:flex;flex-direction:column;gap:10px;padding:14px;
  overflow:auto;
}
#label-controls{
  display:flex;align-items:center;gap:18px;flex-wrap:wrap;
}
#label-toggle{
  background:linear-gradient(135deg,var(--accent) 0%,#d4a017 100%);color:#111;border:none;
  padding:8px 14px;border-radius:6px;font-size:13px;font-weight:700;cursor:pointer;
}
#label-toggle.on{background:linear-gradient(135deg,var(--danger) 0%,#c73e4a 100%);color:#fff}
#label-canvas-wrap{
  flex:1;min-height:300px;
  background:#000;border:1px solid var(--border);border-radius:10px;
  display:flex;align-items:center;justify-content:center;overflow:hidden;
}
#label-canvas{max-width:100%;max-height:70vh}
#label-empty{color:var(--dim);font-size:13px;padding:30px;text-align:center}
#label-actions{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
#label-actions .accept{
  background:linear-gradient(135deg,var(--teal) 0%,#00a88a 100%);color:#0e1117;border:none;
  padding:10px 22px;border-radius:6px;font-size:14px;font-weight:700;cursor:pointer;
}
#label-actions .reject{
  background:linear-gradient(135deg,var(--danger) 0%,#c73e4a 100%);color:#fff;border:none;
  padding:10px 22px;border-radius:6px;font-size:14px;font-weight:700;cursor:pointer;
}
#label-actions .accept:hover, #label-actions .reject:hover, #label-toggle:hover{opacity:0.88}

/* ── Train button + epochs dropdown ── */
#train-controls{display:flex;align-items:center;gap:8px;margin-left:auto}
#train-epochs{
  background:#1a2130;color:var(--fg);border:1px solid var(--border);
  padding:6px 8px;border-radius:6px;font-size:12px;cursor:pointer;
}
#train-epochs:disabled{opacity:0.55;cursor:not-allowed}
#train-eta-hint{color:var(--dim);font-size:11px;white-space:nowrap}
#train-btn{
  background:linear-gradient(135deg,#7c4dff 0%,#5e35b1 100%);color:#fff;border:none;
  padding:8px 14px;border-radius:6px;font-size:13px;font-weight:700;cursor:pointer;
}
#train-btn:disabled{
  background:linear-gradient(135deg,#3a3a4a 0%,#2a2a35 100%);color:#666;cursor:not-allowed;
}
#train-btn:not(:disabled):hover{opacity:0.88}

/* ── Training modal ── */
#train-overlay{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,0.85);
  z-index:400;align-items:center;justify-content:center;backdrop-filter:blur(2px);
}
#train-overlay.open{display:flex}
#train-card{
  background:linear-gradient(145deg,#1a2130 0%,#141c26 100%);border:1px solid var(--border);
  border-radius:12px;padding:24px 28px;min-width:360px;max-width:680px;width:min(680px,92vw);
  box-shadow:0 16px 48px rgba(0,0,0,0.7);
  display:flex;flex-direction:column;max-height:88vh;
}
#train-log-wrap{
  background:#0a0f15;border:1px solid var(--border);border-radius:6px;
  margin-bottom:14px;padding:8px 10px;
  font-family:var(--mono);font-size:11px;line-height:1.45;color:#9fb3c8;
  height:240px;overflow:auto;white-space:pre-wrap;word-break:break-word;
}
#train-log-wrap.empty{color:var(--dim);font-style:italic}
#train-log-label{
  font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--dim);
  margin-bottom:5px;
}
#train-title{font-size:16px;font-weight:700;color:var(--accent);margin-bottom:12px}
#train-state-line{color:var(--text);font-size:14px;margin-bottom:14px}
#train-bar-wrap{background:#0d141d;border:1px solid var(--border);border-radius:4px;height:10px;overflow:hidden;margin-bottom:14px;position:relative}
#train-bar{height:100%;width:0%;background:linear-gradient(90deg,#7c4dff,#00d4aa);transition:width 0.4s}
/* Indeterminate animation used during the TensorRT export phase (no % known) */
#train-bar-wrap.indeterminate #train-bar{
  width:100% !important;
  background:linear-gradient(90deg,
    rgba(124,77,255,0.15) 0%, rgba(124,77,255,0.85) 25%,
    rgba(0,212,170,0.85) 50%, rgba(124,77,255,0.85) 75%,
    rgba(124,77,255,0.15) 100%);
  background-size:200% 100%;
  animation:train-stripes 1.4s linear infinite;
}
@keyframes train-stripes{0%{background-position:200% 0}100%{background-position:-200% 0}}
#train-info{display:grid;grid-template-columns:1fr 1fr;gap:6px 18px;font-size:12px;margin-bottom:14px}
#train-info .stat-val{font-family:var(--mono)}
#train-msg{color:var(--dim);font-size:11px;margin-bottom:12px}
#train-actions{display:flex;gap:10px;justify-content:flex-end}
#train-actions button{
  border:1px solid var(--border);background:#1a2130;color:var(--text);
  padding:7px 14px;border-radius:6px;font-size:12px;cursor:pointer;font-family:var(--font);
}
#train-cancel{color:var(--danger);border-color:rgba(252,92,101,0.4)}
#train-actions button:hover{opacity:0.85}

/* ── Inference-disabled overlay on the live feed ── */
#start-inference-btn{
  position:absolute;top:14px;right:14px;z-index:14;
  background:linear-gradient(135deg,var(--teal) 0%,#00a88a 100%);color:#0e1117;
  border:none;border-radius:6px;padding:10px 16px;
  font-size:13px;font-weight:700;cursor:pointer;
  box-shadow:0 4px 14px rgba(0,212,170,0.35);
}
#start-inference-btn:hover{opacity:0.92}
#start-inference-btn:disabled{opacity:0.55;cursor:not-allowed}

#feed-disabled{
  position:absolute;inset:0;background:rgba(0,0,0,0.78);
  z-index:5;display:flex;align-items:center;justify-content:center;
}
.feed-disabled-card{
  text-align:center;padding:20px 26px;max-width:300px;
  background:linear-gradient(145deg,#1a2130 0%,#141c26 100%);
  border:1px solid var(--border);border-radius:10px;
}
.feed-disabled-title{font-size:15px;font-weight:700;color:var(--accent);margin-bottom:6px}
.feed-disabled-msg{color:var(--dim);font-size:11px;line-height:1.5}

/* ── Feed ── */
#feed-wrap{
  position:relative;
  background:#000;border-radius:10px;overflow:hidden;
  border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;
}
#feed{max-width:100%;max-height:100%;object-fit:contain;display:block}
.corner{position:absolute;width:20px;height:20px;border-color:var(--teal);border-style:solid;opacity:0.6}
.tl{top:8px;left:8px;border-width:2px 0 0 2px;border-radius:2px 0 0 2px}
.tr{top:8px;right:8px;border-width:2px 2px 0 0;border-radius:0 2px 0 0}
.bl{bottom:8px;left:8px;border-width:0 0 2px 2px;border-radius:0 0 0 2px}
.br{bottom:8px;right:8px;border-width:0 2px 2px 0;border-radius:0 0 2px 0}
.scanlines{
  position:absolute;inset:0;pointer-events:none;
  background:repeating-linear-gradient(to bottom,transparent 0 3px,rgba(0,0,0,0.07) 3px 4px);
}
#stream-expired{
  display:none;position:absolute;inset:0;z-index:6;
  background:rgba(0,0,0,0.82);backdrop-filter:blur(2px);
  align-items:center;justify-content:center;
}
#stream-expired.show{display:flex}
.expired-card{
  text-align:center;padding:22px 28px;max-width:280px;
  background:linear-gradient(145deg,#1a2130 0%,#141c26 100%);
  border:1px solid var(--border);border-radius:10px;
  box-shadow:0 8px 32px rgba(0,0,0,0.6);
}
.expired-title{font-size:16px;font-weight:700;color:var(--accent);margin-bottom:6px}
.expired-msg{color:var(--dim);font-size:11px;margin-bottom:14px;line-height:1.5}
.expired-card button{
  background:linear-gradient(135deg,var(--teal) 0%,#00a88a 100%);color:#0e1117;
  border:none;padding:8px 18px;border-radius:6px;
  font-size:12px;font-weight:700;cursor:pointer;font-family:var(--font);
}
.expired-card button:hover{opacity:0.85}

/* ── Stats panel ── */
#stats-panel{
  display:flex;flex-direction:column;gap:8px;overflow:hidden;
}
.card{
  background:linear-gradient(145deg,#1a2130 0%,#141c26 100%);border:1px solid var(--border);
  border-radius:8px;padding:12px;
}
.card-title{
  font-size:10px;letter-spacing:2px;text-transform:uppercase;
  color:var(--dim);margin-bottom:10px;
}
.big-stat{
  display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;
}
.big-val{font-size:22px;font-weight:700;font-family:var(--mono)}
.big-label{font-size:10px;color:var(--dim)}
.fps-val{color:var(--teal)}
.active-val{color:var(--blue)}
.fps-bar-wrap{background:#111;border-radius:3px;height:4px;margin-bottom:10px}
.fps-bar{height:4px;border-radius:3px;transition:width 0.5s,background 0.5s}
.stat-row{display:flex;justify-content:space-between;margin-bottom:5px;font-size:12px}
.stat-lbl{color:var(--dim)}
.stat-val{color:var(--text);font-family:var(--mono)}
.tag{
  display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;
}
.tag-on{background:linear-gradient(90deg,rgba(0,212,170,0.22) 0%,rgba(0,212,170,0.08) 100%);color:var(--teal);border:1px solid rgba(0,212,170,0.3)}
.tag-off{background:linear-gradient(90deg,rgba(74,85,104,0.25) 0%,rgba(74,85,104,0.08) 100%);color:var(--dim);border:1px solid var(--border)}
#hat-btn{
  width:100%;padding:7px;border-radius:6px;border:1px solid var(--border);
  background:linear-gradient(135deg,rgba(245,197,24,0.10) 0%,rgba(245,197,24,0.03) 100%);color:var(--accent);
  font-size:12px;cursor:pointer;transition:all 0.15s;font-family:var(--font);
  margin-bottom:6px;
}
#hat-btn:hover{background:linear-gradient(135deg,rgba(245,197,24,0.22) 0%,rgba(245,197,24,0.08) 100%)}
#hat-btn.on{background:linear-gradient(135deg,rgba(245,197,24,0.30) 0%,rgba(245,197,24,0.12) 100%);border-color:var(--accent)}
#trails-btn{
  width:100%;padding:7px;border-radius:6px;border:1px solid var(--border);
  background:linear-gradient(135deg,rgba(79,195,247,0.10) 0%,rgba(79,195,247,0.03) 100%);color:var(--blue);
  font-size:12px;cursor:pointer;transition:all 0.15s;font-family:var(--font);
  margin-bottom:6px;
}
#trails-btn:hover{background:linear-gradient(135deg,rgba(79,195,247,0.22) 0%,rgba(79,195,247,0.08) 100%)}
#trails-btn.on{background:linear-gradient(135deg,rgba(79,195,247,0.30) 0%,rgba(79,195,247,0.12) 100%);border-color:var(--blue)}
#enhance-btn{
  width:100%;padding:7px;border-radius:6px;border:1px solid var(--border);
  background:linear-gradient(135deg,rgba(0,212,170,0.10) 0%,rgba(0,212,170,0.03) 100%);color:var(--teal);
  font-size:12px;cursor:pointer;transition:all 0.15s;font-family:var(--font);
  margin-bottom:6px;
}
#enhance-btn:hover{background:linear-gradient(135deg,rgba(0,212,170,0.22) 0%,rgba(0,212,170,0.08) 100%)}
#enhance-btn.on{background:linear-gradient(135deg,rgba(0,212,170,0.30) 0%,rgba(0,212,170,0.12) 100%);border-color:var(--teal)}
/* Confidence slider card */
#conf-card{margin-bottom:0}
.conf-title-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.conf-title-row .card-title{margin-bottom:0}
#conf-val{font-family:var(--mono);font-size:12px;color:var(--accent)}
.conf-row{display:flex;align-items:center;gap:6px}
/* Resolution + Model dropdowns */
#res-select, #model-select{
  width:100%;padding:7px 10px;border-radius:6px;border:1px solid var(--border);
  background:linear-gradient(135deg,#1a2130 0%,#141c26 100%);color:var(--text);
  font-size:12px;font-family:var(--font);cursor:pointer;outline:none;
  appearance:none;-webkit-appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%234a5568'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 10px center;
  padding-right:28px;
}
#conf-slider{
  flex:1;-webkit-appearance:none;appearance:none;
  height:4px;border-radius:2px;outline:none;cursor:pointer;
  background:linear-gradient(to right,var(--accent) 0%,var(--accent) 35%,#1e2d3d 35%,#1e2d3d 100%);
}
#conf-slider::-webkit-slider-thumb{
  -webkit-appearance:none;width:14px;height:14px;border-radius:50%;
  background:var(--accent);cursor:pointer;border:2px solid #0e1117;
  box-shadow:0 0 4px rgba(245,197,24,0.5);
}
#conf-slider::-moz-range-thumb{
  width:14px;height:14px;border-radius:50%;border:2px solid #0e1117;
  background:var(--accent);cursor:pointer;
}
#conf-val{font-family:var(--mono);font-size:12px;color:var(--accent);width:30px;text-align:right;flex-shrink:0}
#reset-btn{
  width:100%;padding:7px;border-radius:6px;border:1px solid var(--border);
  background:linear-gradient(135deg,rgba(252,92,101,0.12) 0%,rgba(252,92,101,0.04) 100%);color:var(--danger);
  font-size:12px;cursor:pointer;transition:background 0.15s;font-family:var(--font);
}
#reset-btn:hover{background:linear-gradient(135deg,rgba(252,92,101,0.24) 0%,rgba(252,92,101,0.08) 100%)}

/* Fish list */
#fish-list{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:4px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.fish-card{
  background:linear-gradient(145deg,#1a2130 0%,#141c26 100%);border:1px solid var(--border);border-radius:6px;
  padding:7px 9px;transition:border-color 0.2s;
}
.fish-card.alive{border-left:3px solid var(--teal);background:linear-gradient(90deg,rgba(0,212,170,0.08) 0%,#141c26 60%)}
.fish-card.dead{border-left:3px solid var(--border);opacity:0.55}
.fish-id{font-size:12px;font-weight:600;color:var(--text);font-family:var(--mono)}
.fish-meta{font-size:10px;color:var(--dim);margin-top:2px}

/* ── Filmstrip ── */
#filmstrip{
  grid-area:filmstrip;
  background:linear-gradient(180deg,#0f1820 0%,#161d27 100%);border-top:1px solid var(--border);
  padding:10px 14px;display:flex;flex-direction:column;gap:8px;overflow:hidden;
}
.film-header{
  display:flex;align-items:center;justify-content:space-between;
}
.film-title{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--dim)}
.film-count{font-size:11px;color:var(--accent);font-family:var(--mono)}
#snap-row{
  display:flex;gap:8px;overflow-x:auto;flex:1;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent;
  padding-bottom:4px;
}
.snap-thumb{
  flex-shrink:0;width:140px;height:79px;
  border-radius:6px;overflow:hidden;border:1px solid var(--border);
  position:relative;cursor:pointer;transition:border-color 0.2s;
}
.snap-thumb:hover{border-color:var(--accent)}
.snap-thumb img{width:100%;height:100%;object-fit:cover}
.snap-label{
  position:absolute;bottom:0;left:0;right:0;
  background:linear-gradient(to top,rgba(0,0,0,0.85) 0%,transparent 100%);color:var(--text);
  font-size:9px;padding:6px 5px 3px;font-family:var(--mono);
}
.snap-empty{color:var(--dim);font-size:12px;padding-top:20px}

/* Modal */
#modal{
  display:none;position:fixed;inset:0;
  background:rgba(0,0,0,0.85);z-index:100;
  align-items:center;justify-content:center;
}
#modal.open{display:flex}
#modal img{max-width:90vw;max-height:85vh;border-radius:8px;border:1px solid var(--border)}
#modal-close{
  position:absolute;top:20px;right:24px;
  background:none;border:none;color:var(--text);font-size:24px;cursor:pointer;
}

/* Fullscreen video overlay */
#fs-overlay{
  display:none;position:fixed;inset:0;
  background:#000;z-index:300;
  align-items:center;justify-content:center;
}
#fs-overlay.open{display:flex}
#fs-overlay img{
  width:100%;height:100%;object-fit:contain;
}
#fs-close{
  position:absolute;top:14px;right:14px;
  background:rgba(0,0,0,0.65);color:#fff;
  border:1px solid rgba(255,255,255,0.25);border-radius:50%;
  width:38px;height:38px;font-size:18px;
  cursor:pointer;display:flex;align-items:center;justify-content:center;
  z-index:301;transition:background 0.15s;
}
#fs-close:hover{background:rgba(255,255,255,0.15)}
#feed-wrap{cursor:pointer}
</style>
</head><body>

<!-- Sidebar -->
<nav id="sidebar">
  <div class="logo"><span class="logo-icon">🐟</span>AquaScope</div>

  <div class="nav-section">Monitor</div>
  <div class="nav-item active" id="nav-live" onclick="switchTab('live')"><span class="nav-icon">📹</span>Live Feed</div>
  <div class="nav-item"><span class="nav-icon">📊</span>Analytics</div>
  <div class="nav-item"><span class="nav-icon">🖼️</span>Snapshots</div>

  <div class="nav-section">Training</div>
  <div class="nav-item" id="nav-train" onclick="switchTab('train')"><span class="nav-icon">🏷️</span>Label Fish</div>

  <div class="nav-section">System</div>
  <div class="nav-item"><span class="nav-icon">⚙️</span>Settings</div>

  <div class="nav-spacer"></div>
  <div class="nav-item" style="margin-bottom:6px">
    <span class="status-dot"></span>
    <span id="conn-status">Connected</span>
  </div>
</nav>

<!-- Topbar -->
<header id="topbar">
  <div class="topbar-left">
    <div class="page-title">Live Feed</div>
    <div class="live-badge">● LIVE</div>
  </div>
  <div class="topbar-right">
    <span id="uptime" style="color:var(--dim);font-size:11px"></span>
    <span id="clock"></span>
    <div id="snap-wrap" style="position:relative">
      <button id="snap-btn" onclick="takeSnap()">📷 Snapshot</button>
      <div id="snap-popover">
        <div id="snap-pop-row"></div>
        <div id="snap-pop-empty">No snapshots yet</div>
      </div>
    </div>
  </div>
</header>

<!-- Main -->
<main id="main">
  <!-- Feed -->
  <div id="feed-wrap">
    <img id="feed" src="/stream" alt="live">
    <div class="corner tl"></div>
    <div class="corner tr"></div>
    <div class="corner bl"></div>
    <div class="corner br"></div>
    <div class="scanlines"></div>
    <div id="feed-disabled" style="display:none">
      <div class="feed-disabled-card">
        <div class="feed-disabled-title">⏸ Inference paused</div>
        <div class="feed-disabled-msg">Training is using the GPU. The live feed will resume automatically when training completes.</div>
      </div>
    </div>
    <div id="stream-expired">
      <div class="expired-card">
        <div class="expired-title">⏸ Stream paused</div>
        <div class="expired-msg">3-minute session limit reached.<br>Refresh the page to keep streaming.</div>
        <button onclick="location.reload()">↻ Refresh</button>
      </div>
    </div>
    <!-- Shown after the user toggles labeling off — one tap to kill any
         lingering training process and snap inference back to the user-tuned
         confidence threshold. -->
    <button id="start-inference-btn" onclick="startInferencing(event)" style="display:none">
      ▶ Start inferencing
    </button>
  </div>

  <!-- Stats -->
  <div id="stats-panel">
    <!-- FPS + Active -->
    <div class="card">
      <div class="card-title">Detection</div>
      <div class="big-stat">
        <div>
          <div class="big-val fps-val" id="s-fps">--</div>
          <div class="big-label">FPS</div>
        </div>
        <div style="text-align:right">
          <div class="big-val active-val" id="s-active">--</div>
          <div class="big-label">Active</div>
        </div>
      </div>
      <div class="fps-bar-wrap"><div class="fps-bar" id="fps-bar"></div></div>
      <div class="stat-row"><span class="stat-lbl">Total IDs</span><span class="stat-val" id="s-total">--</span></div>
      <div class="stat-row"><span class="stat-lbl">Frame</span><span class="stat-val" id="s-frame">--</span></div>
    </div>


    <!-- Fish list -->
    <div class="card" style="flex:1;overflow:hidden;display:flex;flex-direction:column">
      <div class="card-title">Fish Activity</div>
      <div id="fish-list"></div>
    </div>

    <div class="card">
      <div class="card-title">Resolution</div>
      <select id="res-select" onchange="setResolution(this.value)">
        <option value="480p">480p &nbsp;(854×480)</option>
        <option value="720p">720p &nbsp;(1280×720)</option>
        <option value="1080p" selected>1080p (1920×1080)</option>
      </select>
    </div>

    <div class="card">
      <div class="card-title">Model</div>
      <select id="model-select" onchange="setModel(this.value)">
        <option>loading…</option>
      </select>
    </div>

    <div class="card" id="conf-card">
      <div class="conf-title-row">
        <div class="card-title">Confidence</div>
        <span id="conf-val">35%</span>
      </div>
      <div class="conf-row">
        <input id="conf-slider" type="range" min="5" max="95" step="5" value="35" oninput="onConfSlider(this.value)">
      </div>
    </div>
    <button id="trails-btn" onclick="toggleTrails()">〰 Trails: OFF</button>
    <button id="enhance-btn" class="on" onclick="toggleEnhance()">✨ Enhance: ON</button>
    <button id="hat-btn" onclick="toggleHat()">🎩 Party Hats: OFF</button>
    <button id="reset-btn" onclick="doReset()">↺ Reset Trails</button>
  </div>
</main>

<!-- Label panel (Training tab) -->
<main id="label-panel" style="display:none">
  <div class="card" id="label-controls">
    <button id="label-toggle" onclick="toggleLabeling()">▶ Start labeling</button>
    <div class="stat-row" style="margin:0">
      <span class="stat-lbl">Status</span>
      <span class="stat-val" id="label-state">off</span>
    </div>
    <div class="stat-row" style="margin:0">
      <span class="stat-lbl">Queued</span>
      <span class="stat-val" id="label-count">0</span>
    </div>
    <div class="stat-row" style="margin:0">
      <span class="stat-lbl">Saved</span>
      <span class="stat-val" id="label-saved">0/100</span>
    </div>
    <div class="stat-row" style="margin:0">
      <span class="stat-lbl">Track</span>
      <span class="stat-val" id="label-track">—</span>
    </div>
    <div class="stat-row" style="margin:0">
      <span class="stat-lbl">Class</span>
      <span class="stat-val" id="label-class">—</span>
    </div>
    <div id="train-controls">
      <select id="train-epochs" onchange="onEpochsChange()" title="Number of training epochs">
        <option value="5" selected>5 epochs</option>
        <option value="10">10 epochs</option>
        <option value="15">15 epochs</option>
        <option value="20">20 epochs</option>
        <option value="30">30 epochs</option>
        <option value="50">50 epochs</option>
      </select>
      <span id="train-eta-hint">—</span>
      <button id="train-btn" disabled onclick="confirmTraining()">🧠 Train model</button>
    </div>
  </div>
  <div id="label-canvas-wrap">
    <canvas id="label-canvas" style="display:none"></canvas>
    <div id="label-empty">No candidates yet — press <b>▶ Start labeling</b> and let the tracker collect detections.</div>
  </div>
  <div class="card" id="label-actions">
    <button class="reject" onclick="labelDecision(0)">✗ Not a fish</button>
    <button class="accept" onclick="labelDecision(1)">✓ Yes, fish</button>
    <span class="stat-lbl" style="margin-left:auto">Shortcuts: <span class="stat-val">y</span> / <span class="stat-val">n</span></span>
  </div>
</main>

<!-- Filmstrip -->
<footer id="filmstrip">
  <div class="film-header">
    <div class="film-title">Snapshots</div>
    <div class="film-count" id="snap-count">0</div>
  </div>
  <div id="snap-row"><div class="snap-empty" id="snap-empty">No snapshots yet — press 📷 to capture</div></div>
</footer>

<!-- Modal lightbox -->
<div id="modal">
  <button id="modal-close" onclick="closeModal()">✕</button>
  <img id="modal-img" src="" alt="snapshot">
</div>

<!-- Fullscreen video overlay -->
<div id="fs-overlay">
  <button id="fs-close" onclick="closeFeed()">✕</button>
  <img id="fs-feed" src="" alt="fullscreen feed">
</div>

<!-- Training progress modal -->
<div id="train-overlay">
  <div id="train-card">
    <div id="train-title">🧠 Training model</div>
    <div id="train-state-line">starting…</div>
    <div id="train-bar-wrap"><div id="train-bar"></div></div>
    <div id="train-info">
      <div><span class="stat-lbl">Epoch</span> <span class="stat-val" id="train-epoch">—</span> / <span id="train-total">—</span></div>
      <div><span class="stat-lbl">Elapsed</span> <span class="stat-val" id="train-elapsed">—</span></div>
      <div><span class="stat-lbl">ETA</span> <span class="stat-val" id="train-eta">—</span></div>
      <div><span class="stat-lbl">Version</span> <span class="stat-val" id="train-version">—</span></div>
    </div>
    <div id="train-msg">Inference is paused while training runs.</div>
    <div id="train-log-label">Training output</div>
    <div id="train-log-wrap" class="empty">waiting for training subprocess output…</div>
    <div id="train-actions">
      <button id="train-cancel" onclick="cancelTraining()">Cancel</button>
      <button id="train-close" onclick="closeTrainModal()" style="display:none">Close</button>
    </div>
  </div>
</div>


<script>
let startTime = Date.now();

function doReset() {
  fetch('/reset').then(() => {
    document.getElementById('fish-list').innerHTML = '';
  });
}

function toggleTrails() {
  fetch('/trails').then(r => r.json()).then(d => {
    const btn = document.getElementById('trails-btn');
    btn.textContent = '〰 Trails: ' + (d.trails ? 'ON' : 'OFF');
    btn.classList.toggle('on', d.trails);
  });
}

// ── Resolution dropdown ──────────────────────────────────
function setResolution(val) {
  fetch('/resolution?v=' + encodeURIComponent(val));
}

// ── Model dropdown ───────────────────────────────────────
function setModel(val) {
  fetch('/model?v=' + encodeURIComponent(val));
}

function loadModels() {
  fetch('/models').then(r => r.json()).then(d => {
    const sel = document.getElementById('model-select');
    sel.innerHTML = '';
    const models = d.models || [];
    if (!models.length) {
      const opt = document.createElement('option');
      opt.textContent = '(no models found)';
      opt.disabled = true;
      sel.appendChild(opt);
      return;
    }
    const currentBase = (d.current || '').split('/').pop();
    models.forEach(m => {
      const base = m.split('/').pop();
      const opt = document.createElement('option');
      opt.value = base;
      opt.textContent = m;
      if (base === currentBase) opt.selected = true;
      sel.appendChild(opt);
    });
  });
}

// ── Confidence slider ────────────────────────────────────
let _confDebounce = null;

function _sendConf(pct) {
  clearTimeout(_confDebounce);
  _confDebounce = setTimeout(() => {
    fetch('/conf?v=' + (pct / 100).toFixed(2));
  }, 120);
}

function _updateConfUI(pct) {
  const slider = document.getElementById('conf-slider');
  const fill = Math.round((pct - 5) / 90 * 100); // map [5,95] → [0,100]%
  slider.value = pct;
  slider.style.background =
    'linear-gradient(to right,var(--accent) 0%,var(--accent) ' + fill + '%,#1e2d3d ' + fill + '%,#1e2d3d 100%)';
  document.getElementById('conf-val').textContent = pct + '%';
}

function onConfSlider(val) {
  const pct = parseInt(val, 10);
  _updateConfUI(pct);
  _sendConf(pct);
}

function toggleEnhance() {
  fetch('/enhance').then(r => r.json()).then(d => {
    const btn = document.getElementById('enhance-btn');
    btn.textContent = '✨ Enhance: ' + (d.enhance ? 'ON' : 'OFF');
    btn.classList.toggle('on', d.enhance);
  });
}

function toggleHat() {
  fetch('/hat').then(r => r.json()).then(d => {
    const btn = document.getElementById('hat-btn');
    btn.textContent = '🎩 Party Hats: ' + (d.hat ? 'ON' : 'OFF');
    btn.classList.toggle('on', d.hat);
  });
}

// ── Snapshot state ──────────────────────────────────────
let snapList = [];

function addThumb(s) {
  const url = '/screenshots/' + s.filename;
  // Popover thumb
  const popRow = document.getElementById('snap-pop-row');
  const popEmpty = document.getElementById('snap-pop-empty');
  popEmpty.style.display = 'none';
  const pt = document.createElement('div');
  pt.className = 'pop-thumb';
  pt.dataset.file = s.filename;
  pt.innerHTML = '<img src="' + url + '"><button class="del" onclick="deleteSnap(event,\'' + s.filename + '\')">✕</button>';
  pt.querySelector('img').onclick = () => openModal(url);
  popRow.prepend(pt);

  // Filmstrip thumb
  const row = document.getElementById('snap-row');
  const empty = document.getElementById('snap-empty');
  empty.style.display = 'none';
  const ft = document.createElement('div');
  ft.className = 'snap-thumb';
  ft.dataset.file = s.filename;
  ft.innerHTML = '<img src="' + url + '" loading="lazy"><div class="snap-label">' + s.label + '</div>';
  ft.onclick = () => openModal(url);
  row.prepend(ft);

  document.getElementById('snap-count').textContent = snapList.length;
}

function takeSnap() {
  const btn = document.getElementById('snap-btn');
  const wrap = document.getElementById('snap-wrap');
  btn.disabled = true;
  btn.textContent = '⏳';
  fetch('/screenshot').then(r => r.json()).then(d => {
    btn.textContent = '✓';
    wrap.classList.add('open');
    const s = {filename: d.filename, ts: d.filename.replace('snap_','').replace('.jpg',''), label: 'Snap ' + d.filename.slice(9,15)};
    snapList.unshift(s);
    addThumb(s);
    setTimeout(() => { btn.textContent = '📷 Snapshot'; btn.disabled = false; }, 1200);
  }).catch(() => { btn.textContent = '📷 Snapshot'; btn.disabled = false; });
}

// Toggle popover on button area click
document.getElementById('snap-wrap').addEventListener('click', function(e) {
  if (e.target.id !== 'snap-btn' && snapList.length) {
    this.classList.toggle('open');
  }
});
document.addEventListener('click', e => {
  if (!document.getElementById('snap-wrap').contains(e.target))
    document.getElementById('snap-wrap').classList.remove('open');
});

function deleteSnap(e, filename) {
  e.stopPropagation();
  snapList = snapList.filter(s => s.filename !== filename);
  document.querySelectorAll('[data-file="' + filename + '"]').forEach(el => el.remove());
  document.getElementById('snap-count').textContent = snapList.length;
  if (!snapList.length) {
    document.getElementById('snap-empty').style.display = '';
    document.getElementById('snap-pop-empty').style.display = '';
  }
}

function openModal(url) {
  document.getElementById('modal-img').src = url;
  document.getElementById('modal').classList.add('open');
}
function closeModal() {
  document.getElementById('modal').classList.remove('open');
}
document.getElementById('modal').addEventListener('click', e => {
  if (e.target === document.getElementById('modal')) closeModal();
});

function loadSnapshots() {
  fetch('/screenshots').then(r => r.json()).then(list => {
    if (list.length === snapList.length) return; // no change
    // sync any new entries added by other clients
    const existing = new Set(snapList.map(s => s.filename));
    list.forEach(s => {
      if (!existing.has(s.filename)) {
        snapList.unshift(s);
        addThumb(s);
      }
    });
  });
}

function fpsColor(fps) {
  if (fps >= 20) return '#00d4aa';
  if (fps >= 12) return '#f5c518';
  return '#fc5c65';
}

function updateStats(d) {
  const fps = d.fps ?? 0;
  document.getElementById('s-fps').textContent    = fps.toFixed(1);
  document.getElementById('s-active').textContent = d.active ?? '--';
  document.getElementById('s-total').textContent  = d.total_ids ?? '--';
  document.getElementById('s-frame').textContent  = (d.frame ?? 0).toLocaleString();
  if (d.resolution) {
    const sel = document.getElementById('res-select');
    if (sel.value !== d.resolution) sel.value = d.resolution;
  }

  const bar = document.getElementById('fps-bar');
  bar.style.width = Math.min(fps / 30 * 100, 100) + '%';
  bar.style.background = fpsColor(fps);

  // Fish list
  const fish = d.fish ?? {};
  const now  = Date.now() / 1000;
  const list = document.getElementById('fish-list');
  list.innerHTML = '';
  Object.entries(fish)
    .sort((a, b) => (b[1].last_seen_ts ?? 0) - (a[1].last_seen_ts ?? 0))
    .slice(0, 10)
    .forEach(([id, f]) => {
      const alive = now - (f.last_seen_ts ?? 0) < 2;
      const dist  = Math.round(f.total_distance_px ?? 0);
      const div = document.createElement('div');
      div.className = 'fish-card ' + (alive ? 'alive' : 'dead');
      div.innerHTML =
        '<div class="fish-id">Fish #' + id + (alive ? ' <span style="color:var(--teal);font-size:9px">●</span>' : '') + '</div>' +
        '<div class="fish-meta">' + dist + 'px &nbsp;|&nbsp; ' + f.frame_count + ' frames</div>';
      list.appendChild(div);
    });
}

function tick() {
  const now = new Date();
  document.getElementById('clock').textContent = now.toLocaleTimeString();
  const ms = Date.now() - startTime;
  const s = Math.floor(ms/1000), m = Math.floor(s/60), h = Math.floor(m/60);
  document.getElementById('uptime').textContent =
    'UP ' + (h ? h+'h ' : '') + (m%60 ? (m%60)+'m ' : '') + (s%60) + 's';

  fetch('/stats').then(r => r.json()).then(updateStats).catch(() => {
    document.getElementById('conn-status').textContent = 'Reconnecting…';
  });
}

setInterval(tick, 1000);
setInterval(loadSnapshots, 2000);
tick();
loadSnapshots();
loadModels();

// ── Training / Label tab ─────────────────────────────────
let labelCurrent = null;
let labelTabActive = false;

function switchTab(tab) {
  labelTabActive = (tab === 'train');
  document.getElementById('main').style.display      = labelTabActive ? 'none' : '';
  document.getElementById('label-panel').style.display = labelTabActive ? 'flex' : 'none';
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const navId = labelTabActive ? 'nav-train' : 'nav-live';
  const el = document.getElementById(navId);
  if (el) el.classList.add('active');
  if (labelTabActive) labelTick();
}

function toggleLabeling() {
  fetch('/label/toggle').then(r => r.json()).then(d => {
    const btn = document.getElementById('label-toggle');
    btn.textContent = d.enabled ? '⏸ Stop labeling' : '▶ Start labeling';
    btn.classList.toggle('on', d.enabled);
    document.getElementById('label-state').textContent = d.enabled ? 'on' : 'off';
    // After labeling stops, surface a one-tap "Start inferencing" button on
    // the live feed so the user can explicitly tear down any leftover
    // training subprocess and reset inference back to clean state.
    const sib = document.getElementById('start-inference-btn');
    if (sib) sib.style.display = d.enabled ? 'none' : '';
  });
}

function startInferencing(e) {
  if (e && e.stopPropagation) e.stopPropagation();   // don't open fullscreen feed
  const btn = document.getElementById('start-inference-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Starting…'; }
  // Cancel any leftover trainer subprocess (no-op if none), then ack the
  // post-training pause so the tracker re-acquires camera + model.
  fetch('/train/cancel').catch(() => {}).finally(() => {
    fetch('/train/acknowledge').catch(() => {}).finally(() => {
      setFeedDisabled(false);
      const feed = document.getElementById('feed');
      if (feed && !streamExpired) feed.src = '/stream?t=' + Date.now();
      if (btn) {
        btn.style.display = 'none';
        btn.disabled = false;
        btn.textContent = '▶ Start inferencing';
      }
    });
  });
}

function labelTick() {
  fetch('/label/queue').then(r => r.json()).then(d => {
    document.getElementById('label-count').textContent = d.count;
    if (!d.count) {
      labelCurrent = null;
      document.getElementById('label-empty').style.display = '';
      document.getElementById('label-canvas').style.display = 'none';
      document.getElementById('label-track').textContent = '—';
      document.getElementById('label-class').textContent = '—';
      return;
    }
    const next = d.queue[0];
    if (labelCurrent && labelCurrent.id === next.id) return;   // already showing
    labelCurrent = next;
    document.getElementById('label-empty').style.display = 'none';
    document.getElementById('label-canvas').style.display = '';
    document.getElementById('label-class').textContent = next.class_name;
    document.getElementById('label-track').textContent = '#' + next.track_id;
    drawLabelCanvas(next);
  });
}

function drawLabelCanvas(c) {
  const cv = document.getElementById('label-canvas');
  const ctx = cv.getContext('2d');
  const img = new Image();
  img.onload = () => {
    cv.width = c.img_w;
    cv.height = c.img_h;
    ctx.drawImage(img, 0, 0);
    const [x1, y1, x2, y2] = c.bbox;
    ctx.lineWidth = Math.max(3, Math.round(c.img_w / 400));
    ctx.strokeStyle = '#f5c518';
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    // Filled label tag
    ctx.fillStyle = 'rgba(245,197,24,0.85)';
    const tag = `${c.class_name}  #${c.track_id}`;
    ctx.font = `${Math.max(14, Math.round(c.img_w / 60))}px sans-serif`;
    const m = ctx.measureText(tag);
    const th = parseInt(ctx.font, 10) + 4;
    ctx.fillRect(x1, Math.max(0, y1 - th - 2), m.width + 12, th + 2);
    ctx.fillStyle = '#111';
    ctx.fillText(tag, x1 + 6, Math.max(th, y1 - 6));
  };
  img.src = c.image_url + '?t=' + Date.now();
}

function labelDecision(keep) {
  if (!labelCurrent) return;
  const cid = labelCurrent.id;
  fetch(`/label/decision?id=${encodeURIComponent(cid)}&keep=${keep}`)
    .then(r => r.json()).then(() => {
      labelCurrent = null;
      labelTick();
    });
}

document.addEventListener('keydown', e => {
  if (!labelTabActive) return;
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
  if (e.key === 'y' || e.key === 'Y' || e.key === 'ArrowRight') labelDecision(1);
  if (e.key === 'n' || e.key === 'N' || e.key === 'ArrowLeft')  labelDecision(0);
});

// Poll the queue while the Training tab is open.
setInterval(() => { if (labelTabActive) labelTick(); }, 1000);

// Initial label state sync (so the toggle button reflects server-side state on reload).
fetch('/label/state').then(r => r.json()).then(d => {
  if (d.enabled) {
    const btn = document.getElementById('label-toggle');
    btn.textContent = '⏸ Stop labeling';
    btn.classList.add('on');
    document.getElementById('label-state').textContent = 'on';
  }
});

// ── Training: label count + Train button ─────────────────
let trainEstimate = null;
function selectedEpochs() {
  const sel = document.getElementById('train-epochs');
  return sel ? parseInt(sel.value, 10) : 5;
}
function refreshTrainLabels() {
  const ep = selectedEpochs();
  fetch('/train/labels?epochs=' + ep).then(r => r.json()).then(d => {
    document.getElementById('label-saved').textContent =
      d.count + '/' + d.min_required;
    const btn = document.getElementById('train-btn');
    btn.disabled = !d.ready || trainModalOpen;
    trainEstimate = d.estimate;
    const hint = document.getElementById('train-eta-hint');
    if (hint && d.estimate) {
      hint.textContent = `~${d.estimate.low_min}–${d.estimate.high_min} min`;
    }
  }).catch(() => {});
}
function onEpochsChange() { refreshTrainLabels(); }
setInterval(refreshTrainLabels, 3000);
refreshTrainLabels();

// ── Training: confirm + start + poll progress ────────────
let trainPollInterval = null;
let trainModalOpen = false;

function fmtSec(s) {
  if (s == null || isNaN(s)) return '—';
  s = Math.max(0, Math.round(s));
  const m = Math.floor(s / 60), r = s % 60;
  return m ? `${m}m ${r}s` : `${r}s`;
}

function confirmTraining() {
  if (!trainEstimate) return;
  const e = trainEstimate;
  const ep = selectedEpochs();
  const ok = window.confirm(
    `Train a new model on your labeled data?\n\n` +
    `• Estimated time: ~${e.low_min}–${e.high_min} minutes (${ep} epochs)\n` +
    `• Inference will pause while the GPU is in use\n` +
    `• On success a new models/best.engine_v<N> appears in the Model dropdown\n\n` +
    `Continue?`
  );
  if (!ok) return;
  fetch('/train/start?epochs=' + ep).then(r => r.json()).then(d => {
    if (d.error) { alert('Could not start training: ' + d.error); return; }
    openTrainModal();
    startTrainPolling();
  });
}

function openTrainModal() {
  trainModalOpen = true;
  document.getElementById('train-overlay').classList.add('open');
  const cancelBtn = document.getElementById('train-cancel');
  cancelBtn.style.display = '';
  cancelBtn.disabled = false;
  cancelBtn.textContent = 'Cancel';
  document.getElementById('train-close').style.display = 'none';
  // Disable model dropdown, epochs dropdown, and train button
  const ms = document.getElementById('model-select');
  if (ms) ms.disabled = true;
  const es = document.getElementById('train-epochs');
  if (es) es.disabled = true;
  document.getElementById('train-btn').disabled = true;
}

function closeTrainModal() {
  // Tell the tracker the user has acknowledged the training-finished modal —
  // this is what triggers inference to actually resume on the Jetson side.
  fetch('/train/acknowledge').catch(() => {});
  trainModalOpen = false;
  document.getElementById('train-overlay').classList.remove('open');
  if (trainPollInterval) { clearInterval(trainPollInterval); trainPollInterval = null; }
  const ms = document.getElementById('model-select');
  if (ms) ms.disabled = false;
  const es = document.getElementById('train-epochs');
  if (es) es.disabled = false;
  refreshTrainLabels();
}

function cancelTraining() {
  if (!confirm('Cancel training? Progress will be lost.')) return;
  // Lock the button + show that something's happening, so a slow ultralytics
  // shutdown doesn't make the modal feel frozen between click and the next
  // 2s status poll.
  const btn = document.getElementById('train-cancel');
  if (btn) { btn.disabled = true; btn.textContent = 'Cancelling…'; }
  document.getElementById('train-state-line').textContent = 'Cancelling — stopping the trainer…';
  fetch('/train/cancel').catch(() => {});
}

function startTrainPolling() {
  if (trainPollInterval) clearInterval(trainPollInterval);
  pollTrainStatus();
  pollTrainLog();
  trainPollInterval = setInterval(() => { pollTrainStatus(); pollTrainLog(); }, 2000);
}

function pollTrainStatus() {
  fetch('/train/status').then(r => r.json()).then(updateTrainModal).catch(() => {});
}

function pollTrainLog() {
  fetch('/train/log').then(r => r.json()).then(d => {
    const box = document.getElementById('train-log-wrap');
    if (!box) return;
    const txt = (d && d.log) || '';
    if (!txt) {
      box.classList.add('empty');
      box.textContent = 'waiting for training subprocess output…';
      return;
    }
    box.classList.remove('empty');
    // Pin to bottom only if user is already at the bottom.
    const pinned = (box.scrollHeight - box.clientHeight - box.scrollTop) < 24;
    box.textContent = txt;
    if (pinned) box.scrollTop = box.scrollHeight;
  }).catch(() => {});
}

function updateTrainModal(s) {
  const cur = s.current_epoch || 0;
  const tot = s.total_epochs || 0;
  const pct = (tot > 0) ? Math.round(100 * cur / tot) : 0;
  // The TensorRT export runs AFTER all epochs and has no progress signal.
  // Show an indeterminate animated bar so the operator can tell it's still
  // working (the static 100% bar made it look frozen).
  const wrap = document.getElementById('train-bar-wrap');
  const exporting = (s.state === 'exporting');
  if (wrap) wrap.classList.toggle('indeterminate', exporting);
  if (!exporting) document.getElementById('train-bar').style.width = pct + '%';
  const stateLine = exporting
    ? ((s.message || 'Exporting TensorRT engine') + ' — this can take ~2–3 min on Orin Nano…')
    : ((s.message || s.state || '…') + (tot ? `   (${pct}%)` : ''));
  document.getElementById('train-state-line').textContent = stateLine;
  document.getElementById('train-epoch').textContent = cur || '—';
  document.getElementById('train-total').textContent = tot || '—';
  document.getElementById('train-elapsed').textContent = fmtSec(s.elapsed_sec);
  document.getElementById('train-eta').textContent = exporting ? '—' : fmtSec(s.eta_sec);
  document.getElementById('train-version').textContent = s.version != null ? ('v' + s.version) : '—';

  // Pause / resume the live feed view based on training state.
  const running = s.running || (s.state === 'training' || s.state === 'starting' || s.state === 'exporting');
  setFeedDisabled(running);

  if (s.state === 'done') {
    const engineName = (s.engine_path || s.latest_engine || '').split('/').pop() || 'engine';
    document.getElementById('train-msg').innerHTML =
      `✓ Saved <b>${engineName}</b>. ` +
      `Click <b>Close</b> to resume inference with the new model.`;
    document.getElementById('train-cancel').style.display = 'none';
    document.getElementById('train-close').style.display = '';
    if (trainPollInterval) { clearInterval(trainPollInterval); trainPollInterval = null; }
  } else if (s.state === 'failed') {
    document.getElementById('train-msg').textContent =
      '✗ Training failed: ' + (s.message || 'unknown error');
    document.getElementById('train-cancel').style.display = 'none';
    document.getElementById('train-close').style.display = '';
    if (trainPollInterval) { clearInterval(trainPollInterval); trainPollInterval = null; }
  }
}

function setFeedDisabled(disabled) {
  const overlay = document.getElementById('feed-disabled');
  if (overlay) overlay.style.display = disabled ? 'flex' : 'none';
}

// On page load, see if a training run is already in progress (e.g. user reloaded mid-run).
fetch('/train/status').then(r => r.json()).then(s => {
  if (s.running || s.state === 'training' || s.state === 'starting' || s.state === 'exporting') {
    openTrainModal();
    startTrainPolling();
  }
}).catch(() => {});

// ── 3-minute session limit ──────────────────────────────
const STREAM_LIMIT_MS = 180000;
let streamExpired = false;

function expireStream() {
  streamExpired = true;
  document.getElementById('feed').src = '';
  const fsFeed = document.getElementById('fs-feed');
  if (fsFeed) fsFeed.src = '';
  document.getElementById('fs-overlay').classList.remove('open');
  document.getElementById('stream-expired').classList.add('show');
}
setTimeout(expireStream, STREAM_LIMIT_MS);

document.getElementById('feed').onerror = function() {
  if (streamExpired) return;
  setTimeout(() => { if (!streamExpired) this.src = '/stream?t=' + Date.now(); }, 2000);
};

// ── Fullscreen feed ──────────────────────────────────────
document.getElementById('feed-wrap').addEventListener('click', function() {
  if (streamExpired) return;
  const overlay = document.getElementById('fs-overlay');
  const fsFeed  = document.getElementById('fs-feed');
  fsFeed.src = '/stream?t=' + Date.now();
  overlay.classList.add('open');
  // Try native fullscreen on supported browsers
  if (overlay.requestFullscreen) overlay.requestFullscreen().catch(() => {});
  else if (overlay.webkitRequestFullscreen) overlay.webkitRequestFullscreen();
});

function closeFeed() {
  const overlay = document.getElementById('fs-overlay');
  overlay.classList.remove('open');
  document.getElementById('fs-feed').src = '';
  if (document.fullscreenElement || document.webkitFullscreenElement) {
    (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  }
}

// Close fullscreen overlay on backdrop tap (not on close button)
document.getElementById('fs-overlay').addEventListener('click', function(e) {
  if (e.target === this) closeFeed();
});

// Sync if browser exits native fullscreen via Escape
document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement)
    document.getElementById('fs-overlay').classList.remove('open');
});
</script>
</body></html>"""
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
    # Clear stale training artifacts from a previous dashboard process so the
    # page loads straight into inference instead of replaying a stale modal.
    # (If the prior run was killed mid-training, /tmp/aquascope_train_status.json
    # could still say state=exporting, which would pop the modal on connect.)
    for path in (_train_status_file, _train_log_file):
        try:
            os.remove(path)
        except OSError:
            pass
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
