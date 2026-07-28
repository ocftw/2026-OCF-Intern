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

echo "run ID: ${RUN_DIR##*/}"
[[ -n "$RUN_DIR" ]] || echo "requested run ID: ${REQUESTED:-unknown}"
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  echo "process: running (PID ${PID})"
else
  echo "process: stopped"
fi
echo "log path: ${LOG_PATH:-unknown}"
if [[ -f "${RUN_DIR}/progress.json" ]]; then
  python3 - "${RUN_DIR}/progress.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
print(f"current model: {p.get('current_model')}")
print(f"current benchmark: {p.get('current_benchmark')}")
print(f"progress: {p.get('completed')}/{p.get('total')}")
PY
fi
if [[ -f "${RUN_DIR}/combination_status.json" ]]; then
  python3 - "${RUN_DIR}/combination_status.json" <<'PY'
import json, sys
rows=[r for r in json.load(open(sys.argv[1], encoding="utf-8")) if r["benchmark"] != "_unload"]
for r in rows:
    print(f"  {r['model']} × {r['benchmark']}: {r['status']}")
print(f"combinations: {sum(r['status']=='completed' for r in rows)}/15 completed")
errors=[r for r in rows if r["status"]=="failed"]
if errors: print("latest error:", errors[-1].get("reason",""))
PY
fi
[[ -n "$LOG_PATH" && -f "$LOG_PATH" ]] && tail -n 5 "$LOG_PATH"
