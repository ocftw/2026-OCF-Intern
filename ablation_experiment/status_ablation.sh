#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$HERE/output"
PID_FILE="$OUTPUT_DIR/ablation.pid"
STATUS_FILE="$OUTPUT_DIR/status.json"
LOG_FILE="$OUTPUT_DIR/ablation.log"

if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "State: running (PID $pid)"
  else
    echo "State: process not running (last PID $pid)"
  fi
else
  echo "State: not started"
fi

if [ -f "$STATUS_FILE" ]; then
  echo
  echo "Status JSON:"
  cat "$STATUS_FILE"
fi

if [ -f "$LOG_FILE" ]; then
  echo
  echo "Latest log:"
  tail -n 20 "$LOG_FILE"
fi
