#!/usr/bin/env bash
# Run the official-evaluator ("match") step for several models of one run,
# back to back, in a single tmux-friendly invocation.
#
# Why sequential, not parallel: this host has 8 CPUs, and each model's
# evaluation alone already spawns up to match_workers + cdm_workers +
# teds_workers (currently 1+3+3=7) subprocess workers inside its Docker
# container. Three models at once would oversubscribe the CPU ~2.5x and is
# exactly the kind of contention auto_resume_evaluation.py's stall detector
# exists to catch -- so this deliberately does one model at a time. It uses
# tools/auto_resume_evaluation.py per model (not evaluate_model.py directly)
# so the known quick_match repetition-loop hang is retried/handled the same
# way it already is for completed runs.
#
# Usage:
#   tools/run_match_all.sh [RUN_ID] [MODEL1,MODEL2,...] [BATCH_SIZE]
#
# All arguments are optional and default to this session's current variant
# run and its three models.
set -uo pipefail

tool_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "$tool_dir/.." && pwd)"
cd "$root_dir"

RUN_ID="${1:-v1_6_8ac6f08d111d}"
IFS=',' read -r -a MODELS <<< "${2:-gemma4_e4b,gemma4_12b,gemma4_31b}"
BATCH_SIZE="${3:-100}"
OUTPUT_ROOT="/opt/ocf-ai/outputs/omnidocbench_v1_6"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
POLL_SECONDS=60

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: run directory not found: $RUN_DIR" >&2
    exit 2
fi

expected_pages="$(python3 -c "import json; print(json.load(open('$RUN_DIR/dataset_manifest.json'))['page_count'])")"
echo "run_id=$RUN_ID expected_pages=$expected_pages batch_size=$BATCH_SIZE models=${MODELS[*]}"

terminal_count() {
    local model="$1"
    python3 -c "
import sqlite3
db = sqlite3.connect('$RUN_DIR/results.sqlite')
print(db.execute(\"SELECT count(*) FROM page_result WHERE model_id=? AND status IN ('SUCCESS','FAILED')\", ('$model',)).fetchone()[0])
"
}

wait_for_inference() {
    local model="$1"
    local n
    n="$(terminal_count "$model")"
    while [[ "$n" -lt "$expected_pages" ]]; do
        echo "[$(date -u +%H:%M:%S)] $model: inference not complete yet ($n/$expected_pages terminal) -- waiting ${POLL_SECONDS}s"
        sleep "$POLL_SECONDS"
        n="$(terminal_count "$model")"
    done
    echo "[$(date -u +%H:%M:%S)] $model: inference complete ($n/$expected_pages terminal), starting match"
}

failed_models=()
for model in "${MODELS[@]}"; do
    model_dir="$RUN_DIR/models/$model"
    mkdir -p "$model_dir"
    log_file="$model_dir/match_run_$(date -u +%Y%m%dT%H%M%SZ).log"

    echo "=== $model: match starting, log: $log_file ==="
    wait_for_inference "$model"

    python3 tools/auto_resume_evaluation.py \
        --run-id "$RUN_ID" \
        --model "$model" \
        --batch-size "$BATCH_SIZE" \
        2>&1 | tee -a "$log_file"
    rc="${PIPESTATUS[0]}"

    if [[ "$rc" -eq 0 ]]; then
        echo "=== $model: match SUCCEEDED ==="
    else
        echo "=== $model: match FAILED (exit $rc); see $log_file ==="
        failed_models+=("$model")
    fi
done

if [[ "${#failed_models[@]}" -gt 0 ]]; then
    echo "DONE with failures: ${failed_models[*]}"
    exit 1
fi
echo "DONE: all models matched successfully."
