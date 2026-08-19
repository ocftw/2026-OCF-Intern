#!/usr/bin/env bash
set -Eeuo pipefail

tool_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$tool_dir/mark_eval_timeout_page.py" "$@"
