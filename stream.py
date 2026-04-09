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


def hat_mode_enabled() -> bool:
    with _hat_lock:
        return _hat_mode

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
        try:
            while True:
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
  background:var(--bg);color:var(--text);
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

/* ── Mobile: stack vertically, hide sidebar ── */
@media (max-width: 700px) {
  body{
    grid-template-columns: 1fr;
    grid-template-rows: 48px auto auto auto;
    grid-template-areas:
      "topbar"
      "main-feed"
      "main-stats"
      "filmstrip";
    height:auto;overflow-y:auto;overflow-x:hidden;
  }
  #sidebar{ display:none }
  #topbar{ grid-area:topbar }
  #main{
    grid-area:unset;
    display:flex;flex-direction:column;gap:0;padding:0;
  }
  #feed-wrap{
    height:56vw;min-height:200px;
    border-radius:0;border-left:none;border-right:none;border-top:none;
  }
  #stats-panel{
    flex-direction:row;flex-wrap:wrap;
    gap:8px;padding:10px;overflow:visible;
  }
  #stats-panel .card{ flex:1 1 140px;min-width:130px }
  #stats-panel #fish-list{ max-height:180px }
  #filmstrip{ height:auto;min-height:110px }
  #uptime{ display:none }
  #snap-popover{ right:auto;left:0;width:280px }
}

/* ── Sidebar ── */
#sidebar{
  grid-area:sidebar;
  background:var(--sidebar);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;
  padding:16px 0;
}
.logo{
  display:flex;align-items:center;gap:10px;
  padding:0 18px 20px;
  font-size:16px;font-weight:700;letter-spacing:1px;color:var(--accent);
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
.nav-item.active{color:var(--accent);border-left-color:var(--accent);background:rgba(245,197,24,0.07)}
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
  background:var(--sidebar);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 20px;
}
@media (max-width: 700px) {
  #topbar{ padding:0 10px }
  .page-title{ font-size:13px }
  .live-badge{ padding:2px 7px;font-size:10px }
  #snap-btn{ padding:5px 10px;font-size:11px }
}
.topbar-left{display:flex;align-items:center;gap:14px}
.page-title{font-size:15px;font-weight:600;color:var(--text)}
.live-badge{
  background:rgba(0,212,170,0.12);color:var(--teal);
  border:1px solid rgba(0,212,170,0.3);
  padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;
}
.topbar-right{display:flex;align-items:center;gap:12px}
#clock{color:var(--blue);font-family:var(--mono);font-size:13px}
#uptime{color:var(--dim);font-size:11px}
#snap-btn{
  background:var(--accent);color:#111;border:none;
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
  background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:10px;z-index:50;
  box-shadow:0 8px 32px rgba(0,0,0,0.6);
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
@media (max-width: 700px) {
  #main{
    display:flex;flex-direction:column;
    gap:0;padding:0;
    grid-area:unset;
    overflow:visible;
  }
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

/* ── Stats panel ── */
#stats-panel{
  display:flex;flex-direction:column;gap:8px;overflow:hidden;
}
.card{
  background:var(--card);border:1px solid var(--border);
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
.tag-on{background:rgba(0,212,170,0.15);color:var(--teal);border:1px solid rgba(0,212,170,0.3)}
.tag-off{background:rgba(74,85,104,0.2);color:var(--dim);border:1px solid var(--border)}
#hat-btn{
  width:100%;padding:7px;border-radius:6px;border:1px solid var(--border);
  background:rgba(245,197,24,0.07);color:var(--accent);
  font-size:12px;cursor:pointer;transition:all 0.15s;font-family:var(--font);
  margin-bottom:6px;
}
#hat-btn:hover{background:rgba(245,197,24,0.18)}
#hat-btn.on{background:rgba(245,197,24,0.22);border-color:var(--accent)}
#reset-btn{
  width:100%;padding:7px;border-radius:6px;border:1px solid var(--border);
  background:rgba(252,92,101,0.08);color:var(--danger);
  font-size:12px;cursor:pointer;transition:background 0.15s;font-family:var(--font);
}
#reset-btn:hover{background:rgba(252,92,101,0.18)}

/* Fish list */
#fish-list{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:4px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.fish-card{
  background:var(--card);border:1px solid var(--border);border-radius:6px;
  padding:7px 9px;transition:border-color 0.2s;
}
.fish-card.alive{border-left:3px solid var(--teal)}
.fish-card.dead{border-left:3px solid var(--border);opacity:0.55}
.fish-id{font-size:12px;font-weight:600;color:var(--text);font-family:var(--mono)}
.fish-meta{font-size:10px;color:var(--dim);margin-top:2px}

/* ── Filmstrip ── */
#filmstrip{
  grid-area:filmstrip;
  background:var(--sidebar);border-top:1px solid var(--border);
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
  background:rgba(0,0,0,0.65);color:var(--text);
  font-size:9px;padding:3px 5px;font-family:var(--mono);
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

    <!-- Model -->
    <div class="card">
      <div class="card-title">Model</div>
      <div class="stat-row"><span class="stat-lbl">File</span><span class="stat-val" id="s-model" style="font-size:10px;max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">--</span></div>
      <div class="stat-row"><span class="stat-lbl">SAHI</span><span id="s-sahi" class="tag tag-off">OFF</span></div>
    </div>

    <!-- Fish list -->
    <div class="card" style="flex:1;overflow:hidden;display:flex;flex-direction:column">
      <div class="card-title">Fish Activity</div>
      <div id="fish-list"></div>
    </div>

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

<script>
let startTime = Date.now();

function doReset() {
  fetch('/reset').then(() => {
    document.getElementById('fish-list').innerHTML = '';
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
  document.getElementById('s-model').textContent  = d.model ?? '--';

  const sahi = d.sahi;
  const sahibadge = document.getElementById('s-sahi');
  sahibadge.textContent = sahi ? 'ON' : 'OFF';
  sahibadge.className = 'tag ' + (sahi ? 'tag-on' : 'tag-off');

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

document.getElementById('feed').onerror = function() {
  setTimeout(() => { this.src = '/stream?t=' + Date.now(); }, 2000);
};
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
