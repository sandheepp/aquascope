# AquaScope Dashboard — Remaining Work

The cyber/lab dashboard now runs entirely on **real data** — every view is wired
to a live `app/stream.py` endpoint, and all simulated/demo data has been removed.
This file tracks the features that **could not be made real yet** because they
need backend work that doesn't exist today (or hardware/services we don't have).

## What was built on real data (for reference)

- **Live Feed** — real MJPEG `/stream`, KPIs + fish list from `/stats`, snapshots,
  and controls (`/trails`, `/enhance`, `/conf`, `/models`, `/model`, `/resolution`, `/reset`).
- **Analytics** — `/inference/history` (active-fish & unique-track time series,
  activity-by-hour bucketed from snapshot timestamps, top-tracked-fish table).
- **AI Insights** — tank-health score + sub-scores computed from live metrics;
  rule-based insight feed (thermals, FPS, activity trend, model versions, label
  queue); "Ask Your Tank" answers the suggested questions from real tracking data.
- **Device Health** — new `GET /telemetry` endpoint (CPU %, GPU util %, GPU freq,
  RAM, swap, uptime, thermal zones) read from sysfs/procfs; shows **N/A** where a
  metric isn't exposed on the host.
- **Label & Train** — full `/label/*` + `/train/*` loop, editable retrain threshold
  (`/train/min-labels`), epochs selector (`/train/labels` → `/train/start?epochs=N`),
  model versions with **real mAP@50** from `/train/history?version=N`, and a real
  training-loss curve from `loss_samples` in `/train/status`.

---

## TODO — needs backend work

### 1. Spatial occupancy heatmap (Analytics)
The "where fish dwell" heatmap was removed. The tracker logs only
`total_distance_px` / `frame_count` per track — **no per-fish centroid positions**.
- **Build:** in `app/tracker.py`, accumulate a coarse 2-D occupancy grid (e.g. 12×8
  bins) from each track's centroid per frame; include it in `_build_stats()` /
  `fish_logs` snapshots; add a `/analytics/heatmap` endpoint (or extend
  `/inference/history`). Then re-add the `Heatmap` card to the Analytics view
  (the `Heatmap()` chart component is still in the JS, unused).

### 2. Mean detection-confidence over time (Analytics KPI)
The "Mean Confidence" KPI was dropped — per-detection confidence is not logged or
exposed.
- **Build:** capture per-frame mean confidence in the tracker, log it in the
  `fish_logs` snapshots, and surface it in `/inference/history.samples[].mean_conf`.

### 3. "Ask Your Tank" — free-form natural language (AI Insights)
The suggested questions are answered from real data; **free-form NL input** returns
a "needs an LLM" message.
- **Build:** add an LLM backend (on-device small model, or an API call with a key)
  and a `/ask?q=...` endpoint that grounds answers in the live `/stats` +
  `/inference/history` numbers. Optionally generate the daily summary narrative
  the same way (currently a deterministic template from real metrics).

### 4. SAHI sliced-inference runtime toggle (Settings)
SAHI is selected at **model-load time** (`app/model.py` builds a sliced predictor
when `config["sahi"]`), so it can't be flipped like `/trails`.
- **Build:** a shared flag + predictor rebuild path so `/sahi?v=0|1` can switch
  modes without restarting; then add the toggle back to Settings.

### 5. Stream JPEG-quality runtime control (Settings)
`config["stream_quality"]` is read once at startup by the MJPEG encoder.
- **Build:** a shared, lock-guarded `_stream_quality` (like `_conf_threshold`) read
  per-encode, plus `GET /quality?v=N`.

### 6. Camera exposure runtime control (Settings / Device Health "Capture")
`config["exposure"]` is applied once at camera open.
- **Build:** a shared flag + a V4L2 re-set call in `app/camera.py`, plus
  `GET /exposure?v=...`. Would also let Device Health show the live exposure value
  (currently N/A).

### 7. Record-to-disk runtime toggle (Settings)
`config["record"]` is startup-only.
- **Build:** start/stop the `cv2.VideoWriter` on demand from the tracker via a
  shared flag, plus `GET /record?v=0|1`.

### 8. Public Cloudflare tunnel runtime toggle (Settings)
`start_public_tunnel()` only runs at launch.
- **Build:** start/stop the `cloudflared` subprocess from the dashboard and report
  the public URL via a `GET /tunnel` endpoint.

### 9. Stats-log interval runtime control (Settings)
`config["log_interval_sec"]` is startup-only (drives the `fish_logs` cadence that
feeds Analytics).
- **Build:** shared, lock-guarded interval read by the tracker's log loop, plus
  `GET /log-interval?v=N`.

### 10. Live GPU/model VRAM (Device Health "Memory")
Shows **N/A** — `torch` isn't imported in the HTTP-server process, and per-process
VRAM isn't exposed via sysfs. (The tracker only publishes VRAM during training, in
`/train/status.gpu_samples`.)
- **Build:** have the tracker publish `torch.cuda.memory_allocated()` into `/stats`
  (or a field in `/telemetry`) during inference.

### 11. Power mode / nvpmodel (Device Health banner)
Not collected. Querying needs `nvpmodel -q` (often root) and JetPack/CUDA/TensorRT
versions aren't introspected.
- **Build:** read `nvpmodel -q` (or `/etc/nvpmodel.conf`) in `read_telemetry()` and
  add `power_mode` to `/telemetry`.

### 12. Per-fish friendly names / species (Analytics, Live)
Fish are anonymous tracking IDs (e.g. `#1196`). The old demo named them and
assigned species — both removed.
- **Build (optional):** a species classifier head, or a manual "name this track" UI
  that persists `id → name` server-side and is merged into `/stats` / history.

### 13. Party-hats overlay (cosmetic)
The `/hat` endpoint was removed on `main`, so the toggle was dropped.
- **Build (optional):** re-add the `/hat` endpoint + the tracker's hat-drawing
  easter egg if wanted, then restore the toggle in Live controls / Settings.

### 14. Behavioral baselines / anomaly ML (AI Insights)
The insight feed is **rule-based** (thresholds on current metrics). Statements like
"61% below 7-day baseline" need historical per-fish modeling.
- **Build:** persist rolling per-track baselines and compare against them to produce
  genuine behavioral-anomaly insights.
