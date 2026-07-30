#!/usr/bin/env bash
set -Eeuo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
case "${1:-}" in
  "") exec python3 -m omnidocbench.cli report ;;
  --smoke) exec python3 -m omnidocbench.cli report --smoke ;;
  *) printf 'usage: %s [--smoke]\n' "$0" >&2; exit 64 ;;
esac
