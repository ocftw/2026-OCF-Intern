#!/usr/bin/env bash
set -uo pipefail

SUITE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUNS_DIR="${BENCHMARK_RUNS_DIR:-${SUITE_DIR}/runs}"
[[ "$RUNS_DIR" == /* ]] || RUNS_DIR="${SUITE_DIR}/../${RUNS_DIR}"
STATE_DIR="${RUNS_DIR}/.state"
PID="$(cat "${STATE_DIR}/active.pid" 2>/dev/null || true)"
LOG_PATH="$(cat "${STATE_DIR}/active.log" 2>/dev/null || true)"
RUN_DIR="$(cat "${STATE_DIR}/current_run" 2>/dev/null || true)"
REQUESTED="$(cat "${STATE_DIR}/requested_run_id" 2>/dev/null || true)"
PROCESS_RUNNING=0
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  PROCESS_RUNNING=1
fi

python3 - "$RUN_DIR" "$REQUESTED" "$PID" "$PROCESS_RUNNING" "$LOG_PATH" \
  "${STATE_DIR}/active_scope.json" <<'PY'
import json
import sys
from pathlib import Path

run_text, requested, pid, running_text, log_path, scope_path = sys.argv[1:]
running = running_text == "1"
run_dir = Path(run_text) if run_text else None

model_names = {
    "qwen3_vl_4b": "Qwen3-VL 4B",
    "gemma4_e2b": "Gemma 4 E2B",
    "gemma4_e4b": "Gemma 4 E4B",
    "sea_lion_4b": "SEA-LION 4B（第三方 GGUF）",
    "smolvlm2_2_2b": "SmolVLM2 2.2B",
}
benchmark_names = {
    "omnidocbench": "OmniDocBench",
    "tc_str": "TC-STR",
    "vistw_mcq": "VisTW-MCQ",
}
model_order = list(model_names)
benchmark_order = list(benchmark_names)
all_combinations = [(model, bench) for model in model_order for bench in benchmark_order]


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def bar(completed, total, width=24):
    if not total:
        return "[" + "·" * width + "]"
    ratio = max(0.0, min(1.0, completed / total))
    filled = round(ratio * width)
    return "[" + "█" * filled + "·" * (width - filled) + "]"


manifest = read_json(run_dir / "manifest.json") if run_dir else None
scope = read_json(Path(scope_path)) or {}
selected_benchmarks = scope.get("benchmarks") or benchmark_order
combination_total = int(scope.get("combination_total") or 15)
requested_limit = scope.get("limit")
all_combinations = [
    (model, bench)
    for model in model_order
    for bench in benchmark_order
    if bench in selected_benchmarks
]
formal_progress = read_json(run_dir / "progress.json") if run_dir else None
smoke_progress = read_json(run_dir / "smoke/progress.json") if run_dir else None
formal_rows = read_json(run_dir / "combination_status.json") if run_dir else None
smoke_rows = read_json(run_dir / "smoke/combination_status.json") if run_dir else None

if formal_progress or formal_rows:
    stage = f"正式推論（本次 {combination_total} 組）"
    progress = formal_progress
    rows = formal_rows or []
elif smoke_progress or smoke_rows:
    stage = "Smoke test（每組極小量樣本）"
    progress = smoke_progress
    rows = smoke_rows or []
else:
    stage = "準備資料、evaluator 與模型"
    progress = None
    rows = []

rows = [row for row in rows if row.get("benchmark") != "_unload"]
rows = [row for row in rows if row.get("benchmark") in selected_benchmarks]
completed_combinations = {
    (row.get("model"), row.get("benchmark"))
    for row in rows
    if row.get("status") == "completed"
}
failed_rows = [row for row in rows if row.get("status") == "failed"]

current_model = progress.get("current_model") if progress else None
current_benchmark = progress.get("current_benchmark") if progress else None
phase = progress.get("phase") if progress else None
completed = int(progress.get("completed", 0)) if progress else 0
total = int(progress.get("total", 0)) if progress else 0
updated = progress.get("updated_at_utc") if progress else None
inferred = False

# 舊版 runner 只在 request 完成後更新 progress。若該組已列為 completed，
# process 又仍在跑，依固定執行順序顯示下一組，避免誤稱仍在上一組。
if running and current_model and (current_model, current_benchmark) in completed_combinations:
    next_combo = next(
        (combo for combo in all_combinations if combo not in completed_combinations),
        None,
    )
    if next_combo:
        current_model, current_benchmark = next_combo
        phase = "switching"
        completed = 0
        total = 1 if stage.startswith("Smoke") else 0
        inferred = True

phase_names = {
    "warmup": "Warm-up（不計分）",
    "inference": "推論中",
    "complete": "本組完成",
    "switching": "切換模型／準備 warm-up",
}

print("=" * 62)
print("VLM Benchmark 執行狀態")
print("=" * 62)
print(f"Run ID       : {(run_dir.name if run_dir else requested) or '尚未建立'}")
print(f"背景程序     : {'執行中' if running else '已停止'}" + (f"（PID {pid}）" if pid else ""))
print(f"目前階段     : {stage}")
print(
    "本次範圍     : "
    + ", ".join(benchmark_names.get(item, item) for item in selected_benchmarks)
    + f"；每模型 {requested_limit if requested_limit is not None else '完整 split'}"
)
if current_model and current_benchmark:
    print(f"目前模型     : {model_names.get(current_model, current_model)}")
    print(f"目前 Benchmark: {benchmark_names.get(current_benchmark, current_benchmark)}")
    print(f"目前步驟     : {phase_names.get(phase, phase or '等待更新')}")
    if total:
        percent = completed / total * 100
        print(f"單組進度     : {bar(completed, total)} {completed}/{total}（{percent:.1f}%）")
    else:
        print(f"單組進度     : 尚未完成第一筆")
    if inferred:
        print("進度註記     : 依固定執行順序推定；下一筆完成後會精確更新")
else:
    print("目前工作     : 正在下載／驗證資源，尚未開始模型推論")

print(
    f"組合進度     : {bar(len(completed_combinations), combination_total)} "
    f"{len(completed_combinations)}/{combination_total} 完成"
)
if updated:
    print(f"最近更新     : {updated}")
if failed_rows:
    latest = failed_rows[-1]
    print(
        "最近錯誤     : "
        f"{model_names.get(latest.get('model'), latest.get('model'))} × "
        f"{benchmark_names.get(latest.get('benchmark'), latest.get('benchmark'))}: "
        f"{latest.get('reason') or '未提供原因'}"
    )
elif manifest and manifest.get("status") == "failed":
    print(f"停止原因     : {manifest.get('failure_reason') or '未提供原因'}")
else:
    print("最近錯誤     : 無")

if run_dir:
    print(f"結果目錄     : {run_dir}")
print(f"完整 Log     : {log_path or '尚未建立'}")
print("=" * 62)
PY
