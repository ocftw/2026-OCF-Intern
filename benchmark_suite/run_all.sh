#!/usr/bin/env bash
# 前景完整流程；來源概念參考既有 eval/run_until_done.sh，但失敗樣本不遞補且每筆保留分母。
set -euo pipefail

SUITE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SUITE_DIR}/.." && pwd -P)"
CONFIG="${SUITE_DIR}/configs/experiment.yaml"
DRY_RUN=0
SMOKE_ONLY=0
RESUME=0
RETRY_FAILED=0
REQUESTED_RUN_ID=""

while (($#)); do
  case "$1" in
    --config) CONFIG="$(cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE_ONLY=1; shift ;;
    --resume) RESUME=1; shift ;;
    --retry-failed) RETRY_FAILED=1; shift ;;
    --run-id) REQUESTED_RUN_ID="$2"; shift 2 ;;
    *) echo "未知參數: $1" >&2; exit 2 ;;
  esac
done

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
if ((DRY_RUN)); then
  "$PYTHON" -m ocf_benchmark.cli --config "$CONFIG" preflight --non-strict
  "$PYTHON" - "$CONFIG" <<'PY'
import sys
from ocf_benchmark.config import combinations, config_hash, load_config
cfg = load_config(sys.argv[1])
print(f"[DRY RUN] config_hash={config_hash(cfg)} combinations={len(combinations(cfg))}")
print("[DRY RUN] 未下載 benchmark、模型，未呼叫 Ollama，未執行正式推論。")
PY
  exit 0
fi

"$PYTHON" -m ocf_benchmark.cli --config "$CONFIG" preflight

RUNS_DIR="${BENCHMARK_RUNS_DIR:-${SUITE_DIR}/runs}"
[[ "$RUNS_DIR" == /* ]] || RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"
STATE_DIR="${RUNS_DIR}/.state"
mkdir -p "$STATE_DIR"
MODELS_METADATA="${STATE_DIR}/models_metadata.json"

"$PYTHON" "${SUITE_DIR}/scripts/prepare_data.py" --config "$CONFIG"
"$PYTHON" "${SUITE_DIR}/scripts/fetch_official_evaluators.py" --config "$CONFIG"
"$PYTHON" "${SUITE_DIR}/scripts/prepare_models.py" \
  --config "$CONFIG" --output "$MODELS_METADATA"

init_args=(--config "$CONFIG" init-run --models-metadata "$MODELS_METADATA")
[[ -n "$REQUESTED_RUN_ID" ]] && init_args+=(--run-id "$REQUESTED_RUN_ID")
((RESUME)) && init_args+=(--resume)
RUN_DIR="$("$PYTHON" -m ocf_benchmark.cli "${init_args[@]}")"
printf '%s\n' "$RUN_DIR" > "${STATE_DIR}/current_run"
printf '%s\n' "$CONFIG" > "${STATE_DIR}/current_config"

echo "[stage] 15 組 smoke test"
"$PYTHON" -m ocf_benchmark.cli --config "$CONFIG" execute \
  --run-dir "$RUN_DIR" --models-metadata "$MODELS_METADATA" --smoke
if ((SMOKE_ONLY)); then
  echo "[done] smoke results: ${RUN_DIR}/smoke"
  exit 0
fi

echo "[stage] 15 組正式推論"
full_rc=0
execute_args=(--config "$CONFIG" execute --run-dir "$RUN_DIR" --models-metadata "$MODELS_METADATA")
((RETRY_FAILED)) && execute_args+=(--retry-failed)
"$PYTHON" -m ocf_benchmark.cli "${execute_args[@]}" || full_rc=$?

echo "[stage] 官方評分、bootstrap 與報告"
"$PYTHON" -m ocf_benchmark.cli --config "$CONFIG" score --run-dir "$RUN_DIR"
"$PYTHON" -m ocf_benchmark.cli --config "$CONFIG" report \
  --run-dir "$RUN_DIR" --models-metadata "$MODELS_METADATA"
echo "[done] ${RUN_DIR}"
exit "$full_rc"
