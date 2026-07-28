#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG="${BENCHMARK_CONFIG:-${SUITE_DIR}/configs/experiment.yaml}"
STATE_DIR="${SUITE_DIR}/runs/.state"
mkdir -p "$STATE_DIR"

[[ "$(uname -s)" == "Linux" ]] || { echo "[FAIL] 需要 Linux/Ubuntu" >&2; exit 1; }
for command in python3 curl git ollama docker nohup setsid flock; do
  command -v "$command" >/dev/null 2>&1 || { echo "[FAIL] 找不到 $command" >&2; exit 1; }
done
python3 -c 'import sys; raise SystemExit(sys.version_info < (3,10))' ||
  { echo "[FAIL] 需要 Python 3.10+" >&2; exit 1; }
curl --fail --silent --show-error --max-time 5 \
  "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null
docker info >/dev/null 2>&1 || { echo "[FAIL] Docker daemon 不可用" >&2; exit 1; }
[[ -w "$STATE_DIR" ]] || { echo "[FAIL] output directory 不可寫" >&2; exit 1; }
MIN_FREE_GB=80
[[ "$(basename -- "$CONFIG")" == "constrained.yaml" ]] && MIN_FREE_GB=45
FREE_KB="$(df -Pk "$STATE_DIR" | awk 'NR==2 {print $4}')"
if ((FREE_KB < MIN_FREE_GB * 1024 * 1024)); then
  echo "[FAIL] 磁碟空間不足：需要至少 ${MIN_FREE_GB} GiB" >&2
  exit 1
fi

exec 9>"${STATE_DIR}/active.lock"
if ! flock -n 9; then
  echo "[FAIL] 已有 benchmark suite 執行中；請用 ${SUITE_DIR}/status.sh 查看。" >&2
  exit 1
fi

REQUESTED_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${STATE_DIR}/launch_${REQUESTED_RUN_ID}.log"
setsid nohup "${SUITE_DIR}/run_all.sh" --config "$CONFIG" --resume \
  --run-id "$REQUESTED_RUN_ID" >"$LOG_PATH" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" > "${STATE_DIR}/active.pid"
printf '%s\n' "$LOG_PATH" > "${STATE_DIR}/active.log"
printf '%s\n' "$REQUESTED_RUN_ID" > "${STATE_DIR}/requested_run_id"

echo "run ID（新 run；若自動續跑將沿用舊 ID）: ${REQUESTED_RUN_ID}"
echo "PID: ${PID}"
echo "log: ${LOG_PATH}"
echo "results: ${SUITE_DIR}/runs/<run_id>/"
echo "status: ${SUITE_DIR}/status.sh"
