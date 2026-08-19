#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$HERE/output"
PID_FILE="$OUTPUT_DIR/ablation.pid"
LOG_FILE="$OUTPUT_DIR/ablation.log"
mkdir -p "$OUTPUT_DIR"

if [ -f "$PID_FILE" ]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Ablation is already running (PID $old_pid)."
    echo "Status: $HERE/status_ablation.sh"
    exit 1
  fi
fi

if [ -n "${PYTHON_BIN:-}" ]; then
  python_bin="$PYTHON_BIN"
elif [ -x "$HERE/../.venv/bin/python" ]; then
  python_bin="$HERE/../.venv/bin/python"
elif [ -x "$HERE/../../venv/bin/python" ]; then
  python_bin="$HERE/../../venv/bin/python"
else
  python_bin="$(command -v python3)"
fi

nohup "$python_bin" -u "$HERE/run_ablation.py" --output-dir "$OUTPUT_DIR" "$@" >>"$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"

echo "Ablation started in background."
echo "PID: $pid"
echo "Log: $LOG_FILE"
echo "Status: $HERE/status_ablation.sh"
echo "Report after completion: $OUTPUT_DIR/ablation_report.html"
