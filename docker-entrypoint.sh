#!/usr/bin/env sh
set -e
export DEEPTRACE_DATA_DIR="${DEEPTRACE_DATA_DIR:-/data}"
export DEEPTRACE_WORK_DIR="${DEEPTRACE_WORK_DIR:-$DEEPTRACE_DATA_DIR/runs}"
mkdir -p "$DEEPTRACE_WORK_DIR"
# keep the fitted calibration available even when /data is a fresh volume
if [ ! -f "$DEEPTRACE_DATA_DIR/calibration.json" ] && [ -f /app/data/calibration.json ]; then
  cp /app/data/calibration.json "$DEEPTRACE_DATA_DIR/calibration.json"
fi
exec python serve.py
