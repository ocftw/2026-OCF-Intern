#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

on_error() {
  local code=$?
  printf 'run_full.sh stopped at line %s (exit %s). Re-run the same command to resume only after correcting the reported condition.\n' "${BASH_LINENO[0]:-unknown}" "$code" >&2
  exit "$code"
}
trap on_error ERR

if [[ $# -ne 0 ]]; then
  printf 'usage: %s\n' "$0" >&2
  exit 64
fi

exec python3 -m omnidocbench.cli full
