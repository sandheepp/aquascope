#!/usr/bin/env bash
# Launch the AquaScope dashboard from the project venv.
#
# Usage:
#   scripts/run_dashboard.sh                    # local display + browser stream
#   scripts/run_dashboard.sh --no-display       # headless + browser stream
#   scripts/run_dashboard.sh --public           # + public Cloudflare URL
#   scripts/run_dashboard.sh --model models/best.pt
#
# Any additional args are forwarded to app/fish_tracker.py — see fish_tracker.py
# for the full option list.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "error: venv missing or broken at $ROOT/venv" >&2
  echo "       create it with:  python3 -m venv venv && venv/bin/python -m pip install -r scripts/requirements.txt" >&2
  exit 1
fi

cd "$ROOT"
exec "$PY" app/fish_tracker.py "$@"
