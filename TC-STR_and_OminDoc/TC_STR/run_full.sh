#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
trap 'rc=$?; printf "run_full.sh failed at line %s (exit %s)\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR
cd "$script_dir"
exec python3 -m tc_str_bench.cli full "$@"
