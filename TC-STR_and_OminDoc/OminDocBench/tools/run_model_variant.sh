#!/usr/bin/env bash
set -Eeuo pipefail
tool_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$tool_dir/.."
exec python3 tools/run_model_variant.py "$@"
