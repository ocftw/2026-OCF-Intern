#!/usr/bin/env bash
# 前景執行單一 benchmark；既有 run_all.sh 保留為完整 15 組入口。
set -euo pipefail

SUITE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SUITE_DIR}/.." && pwd -P)"
CONFIG="${BENCHMARK_CONFIG:-${SUITE_DIR}/configs/experiment.yaml}"
BENCHMARK=""
LIMIT=""
DRY_RUN=0
SMOKE_ONLY=0
RESUME=1
RETRY_FAILED=0
REQUESTED_RUN_ID=""
RUN_DIR=""
CURRENT_STAGE="bootstrap"

while (($#)); do
  case "$1" in
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --config) CONFIG="$(cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE_ONLY=1; shift ;;
    --resume) RESUME=1; shift ;;
    --no-resume) RESUME=0; shift ;;
    --retry-failed) RETRY_FAILED=1; shift ;;
    --run-id) REQUESTED_RUN_ID="$2"; shift 2 ;;
    *) echo "[FAIL] 未知參數: $1" >&2; exit 2 ;;
  esac
done

case "$BENCHMARK" in
  tc_str|omnidocbench|vistw_mcq) ;;
  *) echo "[FAIL] --benchmark 必須是 tc_str、omnidocbench 或 vistw_mcq" >&2; exit 2 ;;
esac
if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[FAIL] --limit 必須是正整數" >&2
  exit 2
fi

mark_failed_on_exit() {
  rc=$?
  if ((rc != 0)) && [[ -n "$RUN_DIR" && -f "${RUN_DIR}/manifest.json" ]]; then
    "$PYTHON" -m ocf_benchmark.cli --config "$CONFIG" mark-failed \
      --run-dir "$RUN_DIR" --reason "stage=${CURRENT_STAGE}; exit_code=${rc}" || true
  fi
  exit "$rc"
}

SRV_WORK_DIR="${BENCHMARK_SCRATCH_ROOT:-/srv/ocf-benchmark/work}"
if [[ -d "$SRV_WORK_DIR" && -w "$SRV_WORK_DIR" ]]; then
  export BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-${SRV_WORK_DIR}/data}"
  export BENCHMARK_CACHE_DIR="${BENCHMARK_CACHE_DIR:-${SRV_WORK_DIR}/cache}"
fi

export PYTHONPATH="${SUITE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -x "${SUITE_DIR}/.venv/bin/python" ]]; then
  python3 -m venv "${SUITE_DIR}/.venv"
fi
"${SUITE_DIR}/.venv/bin/python" -m pip install --disable-pip-version-check \
  -r "${SUITE_DIR}/requirements.lock"
"${SUITE_DIR}/.venv/bin/python" -m pip install --disable-pip-version-check \
  --no-build-isolation --no-deps -e "${SUITE_DIR}"

PYTHON="${SUITE_DIR}/.venv/bin/python"
export PYTHONPATH="${SUITE_DIR}/src"
trap mark_failed_on_exit EXIT

if ((DRY_RUN)); then
  "$PYTHON" -m ocf_benchmark.cli --config "$CONFIG" preflight \
    --non-strict --benchmark "$BENCHMARK"
  echo "[DRY RUN] benchmark=${BENCHMARK} limit=${LIMIT:-full} models=5 combinations=5"
  echo "[DRY RUN] 未下載資料或模型，未呼叫 Ollama，未執行推論。"
  exit 0
fi

CURRENT_STAGE="preflight"
"$PYTHON" -m ocf_benchmark.cli --config "$CONFIG" preflight --benchmark "$BENCHMARK"

RUNS_DIR="${BENCHMARK_RUNS_DIR:-${SUITE_DIR}/runs}"
[[ "$RUNS_DIR" == /* ]] || RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"
STATE_DIR="${RUNS_DIR}/.state"
mkdir -p "$STATE_DIR"
MODELS_METADATA="${STATE_DIR}/models_metadata.json"

CURRENT_STAGE="prepare_data"
"$PYTHON" "${SUITE_DIR}/scripts/prepare_data.py" \
  --config "$CONFIG" --benchmark "$BENCHMARK"
CURRENT_STAGE="prepare_evaluators"
"$PYTHON" "${SUITE_DIR}/scripts/fetch_official_evaluators.py" \
  --config "$CONFIG" --benchmark "$BENCHMARK"
CURRENT_STAGE="prepare_models"
"$PYTHON" "${SUITE_DIR}/scripts/prepare_models.py" \
  --config "$CONFIG" --output "$MODELS_METADATA"

CURRENT_STAGE="init_run"
init_args=(--config "$CONFIG" init-run --models-metadata "$MODELS_METADATA")
[[ -n "$REQUESTED_RUN_ID" ]] && init_args+=(--run-id "$REQUESTED_RUN_ID")
((RESUME)) && init_args+=(--resume)
RUN_DIR="$("$PYTHON" -m ocf_benchmark.cli "${init_args[@]}")"
printf '%s\n' "$RUN_DIR" > "${STATE_DIR}/current_run"
printf '%s\n' "$CONFIG" > "${STATE_DIR}/current_config"

echo "[stage] ${BENCHMARK}: 5 個模型 smoke test"
CURRENT_STAGE="smoke"
"$PYTHON" -m ocf_benchmark.cli --config "$CONFIG" execute \
  --run-dir "$RUN_DIR" --models-metadata "$MODELS_METADATA" \
  --benchmark "$BENCHMARK" --smoke
if ((SMOKE_ONLY)); then
  echo "[done] smoke results: ${RUN_DIR}/smoke"
  exit 0
fi

echo "[stage] ${BENCHMARK}: 5 個模型正式推論（每模型 ${LIMIT:-完整 split}）"
CURRENT_STAGE="inference"
execute_args=(--config "$CONFIG" execute --run-dir "$RUN_DIR"
  --models-metadata "$MODELS_METADATA" --benchmark "$BENCHMARK")
[[ -n "$LIMIT" ]] && execute_args+=(--limit "$LIMIT")
((RETRY_FAILED)) && execute_args+=(--retry-failed)
full_rc=0
"$PYTHON" -m ocf_benchmark.cli "${execute_args[@]}" || full_rc=$?

if [[ "$BENCHMARK" == "omnidocbench" && -z "$LIMIT" ]]; then
  echo "[stage] OmniDocBench 官方 evaluator"
  CURRENT_STAGE="score"
  "$PYTHON" -m ocf_benchmark.cli --config "$CONFIG" score \
    --run-dir "$RUN_DIR" --benchmark "$BENCHMARK"
elif [[ "$BENCHMARK" == "omnidocbench" ]]; then
  echo "[note] limited OmniDocBench 不執行／冒充官方完整分數；prediction 已保留。"
fi

echo "[stage] bootstrap 與單 Benchmark 報告"
CURRENT_STAGE="report"
report_args=(--config "$CONFIG" report --run-dir "$RUN_DIR"
  --models-metadata "$MODELS_METADATA" --benchmark "$BENCHMARK")
[[ -n "$LIMIT" ]] && report_args+=(--limit "$LIMIT")
"$PYTHON" -m ocf_benchmark.cli "${report_args[@]}"
echo "[done] run=${RUN_DIR} benchmark=${BENCHMARK} limit=${LIMIT:-full}"
exit "$full_rc"
