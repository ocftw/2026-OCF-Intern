#!/usr/bin/env bash
set -Eeuo pipefail

tool_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$tool_dir/fix_shared_report_permissions.py" "$@"
