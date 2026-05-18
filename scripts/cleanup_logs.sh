#!/usr/bin/env bash
#
# scripts/cleanup_logs.sh — prune old fish_stats_*.json files.
#
# The tracker writes one JSON snapshot per minute to fish_logs/ while it
# runs. Over a long uptime that adds up (1440 files / day) and slows the
# /inference/history endpoint that scans them. This script deletes
# fish_stats_*.json files older than MAX_AGE_HOURS so the directory
# stays bounded.
#
# Usage:
#   bash scripts/cleanup_logs.sh                # delete files >3h old
#   MAX_AGE_HOURS=12 bash scripts/cleanup_logs.sh
#   LOG_DIR=/path/to/logs bash scripts/cleanup_logs.sh
#
# The dashboard already runs an equivalent cleanup as a daemon thread
# every 3 hours (see _start_log_cleanup_thread in app/stream.py), so you
# only need to run this script manually if you want a different cadence,
# a different retention window, or you're cleaning up after the dashboard
# has been stopped.
#
# Optional: schedule via cron for a system that runs the dashboard 24/7:
#   crontab -e
#   0 */3 * * * /usr/bin/bash /home/jetson/projects/aquascope/scripts/cleanup_logs.sh >> /tmp/aquascope_cleanup.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/fish_logs}"
MAX_AGE_HOURS="${MAX_AGE_HOURS:-3}"
MAX_AGE_MINUTES=$(( MAX_AGE_HOURS * 60 ))

if [ ! -d "$LOG_DIR" ]; then
  echo "[cleanup] No log dir at $LOG_DIR — nothing to do."
  exit 0
fi

# -mmin +N matches files with modification age STRICTLY GREATER than N
# minutes. We pre-count so the user gets a clear "removed N files" line
# without grepping `find -print` output.
count=$(find "$LOG_DIR" -maxdepth 1 -name 'fish_stats_*.json' \
          -mmin "+$MAX_AGE_MINUTES" -type f 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
  find "$LOG_DIR" -maxdepth 1 -name 'fish_stats_*.json' \
       -mmin "+$MAX_AGE_MINUTES" -type f -delete
fi
echo "[cleanup] Removed $count fish_stats_*.json files older than ${MAX_AGE_HOURS}h from $LOG_DIR"
