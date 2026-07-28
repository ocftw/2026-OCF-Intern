#!/usr/bin/env bash
# 單 Benchmark 背景啟動器；流程改寫自本套件 launch_all.sh。
set -euo pipefail

SUITE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BENCHMARK="${1:-}"
[[ -n "$BENCHMARK" ]] || { echo "[FAIL] 缺少 benchmark" >&2; exit 2; }
shift
CONFIG="${BENCHMARK_CONFIG:-${SUITE_DIR}/configs/experiment.yaml}"
RUNS_DIR="${BENCHMARK_RUNS_DIR:-${SUITE_DIR}/runs}"
[[ "$RUNS_DIR" == /* ]] || RUNS_DIR="${SUITE_DIR}/../${RUNS_DIR}"
STATE_DIR="${RUNS_DIR}/.state"
FOLLOW_PROGRESS=1
PROGRESS_INTERVAL="${BENCHMARK_STATUS_INTERVAL:-30}"
LIMIT=""
FORWARD_ARGS=()

while (($#)); do
  case "$1" in
    --detach) FOLLOW_PROGRESS=0; shift ;;
    --interval)
      [[ $# -ge 2 ]] || { echo "[FAIL] --interval 需要秒數" >&2; exit 2; }
      PROGRESS_INTERVAL="$2"; shift 2 ;;
    --limit)
      [[ $# -ge 2 ]] || { echo "[FAIL] --limit 需要筆數" >&2; exit 2; }
      LIMIT="$2"; FORWARD_ARGS+=(--limit "$2"); shift 2 ;;
    --config)
      [[ $# -ge 2 ]] || { echo "[FAIL] --config 需要路徑" >&2; exit 2; }
      CONFIG="$2"; shift 2 ;;
    --dry-run|--smoke|--resume|--no-resume|--retry-failed)
      FORWARD_ARGS+=("$1"); shift ;;
    *) echo "[FAIL] 未知參數: $1" >&2; exit 2 ;;
  esac
done
[[ "$PROGRESS_INTERVAL" =~ ^[1-9][0-9]*$ ]] ||
  { echo "[FAIL] interval 必須是正整數秒" >&2; exit 2; }
[[ -z "$LIMIT" || "$LIMIT" =~ ^[1-9][0-9]*$ ]] ||
  { echo "[FAIL] limit 必須是正整數" >&2; exit 2; }

[[ "$(uname -s)" == "Linux" ]] || { echo "[FAIL] 需要 Linux/Ubuntu" >&2; exit 1; }
required_commands=(python3 curl git ollama nohup setsid flock)
[[ "$BENCHMARK" == "omnidocbench" ]] && required_commands+=(docker)
for command in "${required_commands[@]}"; do
  command -v "$command" >/dev/null 2>&1 || { echo "[FAIL] 找不到 $command" >&2; exit 1; }
done
python3 -c 'import sys; raise SystemExit(sys.version_info < (3,10))' ||
  { echo "[FAIL] 需要 Python 3.10+" >&2; exit 1; }
curl --fail --silent --show-error --max-time 5 \
  "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null
if [[ "$BENCHMARK" == "omnidocbench" ]] && ! docker info >/dev/null 2>&1; then
  echo "[FAIL] Docker daemon 不可用（OmniDocBench 官方 evaluator 必需）" >&2
  exit 1
fi

mkdir -p "$STATE_DIR"
exec 9>"${STATE_DIR}/active.lock"
if ! flock -n 9; then
  echo "[FAIL] 已有 benchmark suite 執行中；請用 ${SUITE_DIR}/status.sh 查看。" >&2
  exit 1
fi

REQUESTED_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${STATE_DIR}/launch_${REQUESTED_RUN_ID}.log"
setsid nohup "${SUITE_DIR}/run_benchmark.sh" --benchmark "$BENCHMARK" \
  --config "$CONFIG" --run-id "$REQUESTED_RUN_ID" \
  "${FORWARD_ARGS[@]}" >"$LOG_PATH" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" > "${STATE_DIR}/active.pid"
printf '%s\n' "$LOG_PATH" > "${STATE_DIR}/active.log"
printf '%s\n' "$REQUESTED_RUN_ID" > "${STATE_DIR}/requested_run_id"
printf '{"benchmarks":["%s"],"limit":%s,"combination_total":5}\n' \
  "$BENCHMARK" "${LIMIT:-null}" > "${STATE_DIR}/active_scope.json"

echo "Benchmark: ${BENCHMARK}"
echo "筆數／模型: ${LIMIT:-完整 split}"
echo "PID: ${PID}"
echo "log: ${LOG_PATH}"
echo "status: ${SUITE_DIR}/status.sh"

if ((FOLLOW_PROGRESS)); then
  echo
  echo "每 ${PROGRESS_INTERVAL} 秒更新；Ctrl-C 只停止監看，不停止背景工作。"
  trap 'echo; echo "已離開監看；benchmark 仍在背景執行。"; exit 0' INT TERM
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
