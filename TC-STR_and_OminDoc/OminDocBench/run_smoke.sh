#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

on_error() {
  local code=$?
  printf 'run_smoke.sh failed at line %s (exit %s)\n' "${BASH_LINENO[0]:-unknown}" "$code" >&2
  exit "$code"
}
trap on_error ERR

case "${1:-}" in
  "")
    exec python3 -m omnidocbench.cli smoke
    ;;
  --preflight-only)
    exec python3 -m omnidocbench.cli preflight
    ;;
  --smoke-only)
    exec python3 -m omnidocbench.cli smoke
    ;;
  --diagnostic-10)
    exec python3 -m omnidocbench.cli smoke --quick-load
    ;;
  --report-only)
    exec python3 -m omnidocbench.cli report --smoke
    ;;
  *)
    printf 'usage: %s [--preflight-only|--smoke-only|--diagnostic-10|--report-only]\n' "$0" >&2
    exit 64
    ;;
esac
