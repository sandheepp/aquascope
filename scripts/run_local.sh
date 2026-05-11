#!/usr/bin/env bash
# AquaScope — laptop / non-Jetson launcher.
#
# Bootstraps a Python venv (if missing), installs the laptop requirements,
# and starts the dashboard. Cross-platform: macOS, Linux, Windows (Git Bash).
#
# Usage:
#   bash scripts/run_local.sh                                 # default webcam
#   bash scripts/run_local.sh --video samples/aquarium.mp4    # demo on a file
#   bash scripts/run_local.sh --camera 1                      # second webcam
#   bash scripts/run_local.sh --no-display                    # browser only
#
# Any flags are forwarded to app/fish_tracker.py — see fish_tracker.py for the
# full option list.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv"

# Pick a python interpreter that's actually available.
if command -v python3 >/dev/null 2>&1; then
  PY_BOOTSTRAP="python3"
elif command -v python >/dev/null 2>&1; then
  PY_BOOTSTRAP="python"
else
  echo "error: neither python3 nor python is on PATH" >&2
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "[setup] Creating venv at $VENV ..."
  "$PY_BOOTSTRAP" -m venv "$VENV"
fi

# venv layout differs on Windows.
if [[ -x "$VENV/bin/python" ]]; then
  PY="$VENV/bin/python"
elif [[ -x "$VENV/Scripts/python.exe" ]]; then
  PY="$VENV/Scripts/python.exe"
else
  echo "error: venv created but no python binary found under $VENV" >&2
  exit 1
fi

# Install requirements once. Touch a marker file so subsequent runs are fast.
MARKER="$VENV/.aq_local_installed"
if [[ ! -f "$MARKER" ]] || [[ "$ROOT/scripts/requirements_local.txt" -nt "$MARKER" ]]; then
  echo "[setup] Installing dependencies (one-time) ..."
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$ROOT/scripts/requirements_local.txt"
  touch "$MARKER"
fi

cd "$ROOT"
exec "$PY" app/fish_tracker.py "$@"
