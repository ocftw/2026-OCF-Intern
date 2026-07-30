#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
trap 'rc=$?; printf "run_tests.sh failed at line %s (exit %s)\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR
cd "$script_dir"
python3 -m unittest discover -s tests -v
bash -n run_smoke.sh run_full.sh status.sh build_report.sh run_tests.sh
