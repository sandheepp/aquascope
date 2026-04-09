"""
MJPEG HTTP streaming server + Cloudflare public tunnel.

Usage:
    start_stream(port)          — start local MJPEG server
    start_public_tunnel(port)   — expose via Cloudflare (no account needed)
    push_frame(jpeg_bytes)      — call from tracker loop to update the stream
"""

import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# ── Shared frame state ────────────────────────────────────
_frame: bytes = b""
_lock = threading.Lock()
_reset_flag = False
_reset_lock = threading.Lock()


def request_reset() -> bool:
    """Return True (and clear the flag) if a reset was requested via /reset."""
    global _reset_flag
    with _reset_lock:
        if _reset_flag:
            _reset_flag = False
            return True
    return False


def push_frame(jpeg_bytes: bytes) -> None:
    """Update the frame served to all connected clients."""
    global _frame
    with _lock:
        _frame = jpeg_bytes


# ── MJPEG handler ─────────────────────────────────────────
class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass  # suppress per-request access logs

    def do_GET(self):
        if self.path == "/":
            self._serve_index()
        elif self.path == "/stream":
            self._serve_stream()
        elif self.path == "/reset":
            self._serve_reset()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_reset(self):
        global _reset_flag
        with _reset_lock:
            _reset_flag = True
        body = b'{"status":"reset requested"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_index(self):
        body = (
            b"<!doctype html><html><head>"
            b"<meta charset='utf-8'>"
            b"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            b"<title>AquaScope Live</title>"
            b"<style>"
            b"*{margin:0;padding:0;box-sizing:border-box}"
            b"body{background:#000;height:100vh;overflow:hidden;font-family:monospace}"
            b"#feed{width:100%;height:100vh;object-fit:contain;display:block}"
            b"#bar{position:fixed;bottom:0;left:0;right:0;padding:5px 14px;"
            b"background:rgba(0,0,0,0.6);color:#aaa;font-size:12px;"
            b"display:flex;justify-content:space-between;align-items:center;"
            b"pointer-events:none;border-top:1px solid #333}"
            b"#resetbtn{position:fixed;bottom:6px;right:10px;padding:3px 10px;"
            b"background:#1a1a2e;color:#7af;border:1px solid #444;border-radius:4px;"
            b"font-size:12px;cursor:pointer;pointer-events:all}"
            b"</style></head><body>"
            b"<img id='feed' src='/stream'>"
            b"<div id='bar'>"
            b"<span>AquaScope Live</span>"
            b"<span id='ts'></span>"
            b"</div>"
            b"<button id='resetbtn' onclick=\"fetch('/reset')\">Reset Trails</button>"
            b"<script>"
            b"setInterval(()=>{document.getElementById('ts').textContent=new Date().toLocaleTimeString();},1000);"
            b"document.getElementById('feed').onerror=function(){"
            b"  setTimeout(()=>{this.src='/stream?t='+Date.now();},2000);};"
            b"</script>"
            b"</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
                time.sleep(0.033)  # ~30 fps cap per client
        except (BrokenPipeError, ConnectionResetError):
            pass


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Each client connection runs in its own thread."""
    daemon_threads = True


# ── Public API ────────────────────────────────────────────
def start_stream(port: int) -> None:
    """Start the MJPEG server in a background daemon thread."""
    server = _ThreadingHTTPServer(("0.0.0.0", port), _MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def start_public_tunnel(port: int) -> None:
    """
    Start a Cloudflare quick tunnel and print the public URL.
    Runs in background — prints the URL once it appears in cloudflared output.
    Requires cloudflared to be on PATH (no account needed for quick tunnels).
    """
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
