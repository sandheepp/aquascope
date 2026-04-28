"""
MJPEG HTTP streaming server + Cloudflare public tunnel.

Endpoints:
  /           — dashboard UI
  /stream     — MJPEG video stream
  /stats      — JSON live stats
  /reset      — reset tracker trails
  /screenshot — save current frame, return JSON {filename}
  /screenshots/<file> — serve saved screenshot
"""

import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

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
/* Resolution dropdown */
#res-select{
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
  <div class="nav-item active"><span class="nav-icon">📹</span>Live Feed</div>
  <div class="nav-item"><span class="nav-icon">📊</span>Analytics</div>
  <div class="nav-item"><span class="nav-icon">🖼️</span>Snapshots</div>

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
    <div id="stream-expired">
      <div class="expired-card">
        <div class="expired-title">⏸ Stream paused</div>
        <div class="expired-msg">3-minute session limit reached.<br>Refresh the page to keep streaming.</div>
        <button onclick="location.reload()">↻ Refresh</button>
      </div>
    </div>
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
