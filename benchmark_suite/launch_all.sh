#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG="${BENCHMARK_CONFIG:-${SUITE_DIR}/configs/experiment.yaml}"
RUNS_DIR="${BENCHMARK_RUNS_DIR:-${SUITE_DIR}/runs}"
[[ "$RUNS_DIR" == /* ]] || RUNS_DIR="${SUITE_DIR}/../${RUNS_DIR}"
STATE_DIR="${RUNS_DIR}/.state"
SRV_WORK_DIR="${BENCHMARK_SCRATCH_ROOT:-/srv/ocf-benchmark/work}"
FOLLOW_PROGRESS=1
PROGRESS_INTERVAL="${BENCHMARK_STATUS_INTERVAL:-30}"

while (($#)); do
  case "$1" in
    --detach)
      FOLLOW_PROGRESS=0
      shift
      ;;
    --interval)
      [[ $# -ge 2 ]] || { echo "[FAIL] --interval 需要秒數" >&2; exit 2; }
      PROGRESS_INTERVAL="$2"
      shift 2
      ;;
    *)
      echo "[FAIL] 未知參數: $1" >&2
      echo "用法: $0 [--detach] [--interval SECONDS]" >&2
      exit 2
      ;;
  esac
done

[[ "$PROGRESS_INTERVAL" =~ ^[1-9][0-9]*$ ]] ||
  { echo "[FAIL] interval 必須是正整數秒" >&2; exit 2; }
mkdir -p "$STATE_DIR"

# 由 scripts/configure_srv_scratch.sh 建立後自動採用 ephemeral /srv。
# 使用者明確設定的 BENCHMARK_DATA_DIR/BENCHMARK_CACHE_DIR 永遠優先。
if [[ -d "$SRV_WORK_DIR" && -w "$SRV_WORK_DIR" ]]; then
  export BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-${SRV_WORK_DIR}/data}"
  export BENCHMARK_CACHE_DIR="${BENCHMARK_CACHE_DIR:-${SRV_WORK_DIR}/cache}"
fi
SCRATCH_DIR="${BENCHMARK_CACHE_DIR:-${SUITE_DIR}/cache}"
mkdir -p "$SCRATCH_DIR"

[[ "$(uname -s)" == "Linux" ]] || { echo "[FAIL] 需要 Linux/Ubuntu" >&2; exit 1; }
for command in python3 curl git ollama docker nohup setsid flock; do
  command -v "$command" >/dev/null 2>&1 || { echo "[FAIL] 找不到 $command" >&2; exit 1; }
done
python3 -c 'import sys; raise SystemExit(sys.version_info < (3,10))' ||
  { echo "[FAIL] 需要 Python 3.10+" >&2; exit 1; }
curl --fail --silent --show-error --max-time 5 \
  "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null
if ! docker info >/dev/null 2>&1; then
  echo "[FAIL] Docker daemon 不可用" >&2
  CURRENT_USER="${USER:-$(id -un)}"
  if id -nG "$CURRENT_USER" 2>/dev/null | tr ' ' '\n' | grep -qx docker &&
    ! id -nG | tr ' ' '\n' | grep -qx docker; then
    echo "[HINT] 帳號已有 docker 群組，但目前 session 尚未繼承；請重新登入或先執行 newgrp docker" >&2
  fi
  exit 1
fi
[[ -w "$STATE_DIR" ]] || { echo "[FAIL] output directory 不可寫" >&2; exit 1; }
MIN_SCRATCH_GB=80
MIN_RESULTS_GB=20
if [[ "$(basename -- "$CONFIG")" == "constrained.yaml" ]]; then
  MIN_SCRATCH_GB=45
  MIN_RESULTS_GB=10
fi
SCRATCH_FREE_KB="$(df -Pk "$SCRATCH_DIR" | awk 'NR==2 {print $4}')"
RESULTS_FREE_KB="$(df -Pk "$STATE_DIR" | awk 'NR==2 {print $4}')"
if ((SCRATCH_FREE_KB < MIN_SCRATCH_GB * 1024 * 1024)); then
  echo "[FAIL] scratch 磁碟空間不足：${SCRATCH_DIR} 需要至少 ${MIN_SCRATCH_GB} GiB" >&2
  echo "[HINT] 請先執行 sudo ${SUITE_DIR}/scripts/configure_srv_scratch.sh" >&2
  exit 1
fi
if ((RESULTS_FREE_KB < MIN_RESULTS_GB * 1024 * 1024)); then
  echo "[FAIL] results 磁碟空間不足：${RUNS_DIR} 需要至少 ${MIN_RESULTS_GB} GiB" >&2
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
printf '{"benchmarks":["omnidocbench","tc_str","vistw_mcq"],"limit":null,"combination_total":15}\n' \
  > "${STATE_DIR}/active_scope.json"

echo "run ID（新 run；若自動續跑將沿用舊 ID）: ${REQUESTED_RUN_ID}"
echo "PID: ${PID}"
echo "log: ${LOG_PATH}"
echo "scratch data: ${BENCHMARK_DATA_DIR:-${SUITE_DIR}/data}"
echo "scratch cache: ${BENCHMARK_CACHE_DIR:-${SUITE_DIR}/cache}"
echo "results: ${RUNS_DIR}/<run_id>/"
echo "status: ${SUITE_DIR}/status.sh"

if ((FOLLOW_PROGRESS)); then
  echo
  echo "每 ${PROGRESS_INTERVAL} 秒更新進度；按 Ctrl-C 僅離開監看，背景 benchmark 會繼續。"
  trap 'echo; echo "已離開進度監看；背景 benchmark 仍在執行。"; exit 0' INT TERM
  while kill -0 "$PID" 2>/dev/null; do
    echo
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    "${SUITE_DIR}/status.sh"
    sleep "$PROGRESS_INTERVAL" &
    wait $!
  done
  echo
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') final ====="
  "${SUITE_DIR}/status.sh"
fi
