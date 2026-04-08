#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

REMOTE_USER="${JETSON_USER:-jetson}"
REMOTE_HOST="${JETSON_HOST:-192.168.31.141}"
REMOTE_PATH="${JETSON_PATH:-/home/jetson/projects/aquascope}"
PASSWORD="${JETSON_PASSWORD:?JETSON_PASSWORD not set. Add it to .env or export it.}"

if ! command -v sshpass >/dev/null 2>&1; then
  echo "ERROR: sshpass is required to use this script."
  echo "Install it with:"
  echo "  brew install hudochenkov/sshpass/sshpass"
  echo "or on Ubuntu/Debian: sudo apt install sshpass"
  exit 1
fi

echo "Syncing code from ${SCRIPT_DIR} to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
sshpass -p "$PASSWORD" rsync -avz --delete \
  --exclude='venv' \
  --exclude='fish_logs' \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  --exclude='*.log' \
  --exclude='*.tmp' \
  --exclude='*.swp' \
  --exclude='*.engine' \
  --exclude='*.pt' \
  --exclude='*.onnx' \
  --exclude='*.trt' \
  "$SCRIPT_DIR"/ "$REMOTE_USER"@"$REMOTE_HOST":"$REMOTE_PATH"/

echo "Done. Code synced to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
