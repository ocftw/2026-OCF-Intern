#!/usr/bin/env python3
"""Stop any evaluator container left running for a given run/model.

Companion to the stall/interrupt handling in evaluate_model.py: SIGKILL (or a
supervisor killing the process tree) cannot be intercepted, so a container can
still be orphaned. This finds it the same way evaluation_status.py does --
by bind-mount source, not by name or PID -- and stops it explicitly.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _container_ops import find_containers_with_mount_prefix, stop_container  # noqa: E402


DEFAULT_OUTPUT_ROOT = pathlib.Path("/opt/ocf-ai/outputs/omnidocbench_v1_6")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stop any running evaluator container for a run/model, if one exists."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_base = pathlib.Path(args.output_root) / args.run_id / "models" / args.model
    if not model_base.is_dir():
        print(f"ERROR: model directory does not exist: {model_base}", file=sys.stderr)
        return 1
    containers = find_containers_with_mount_prefix(model_base)
    if not containers:
        print(f"no running container found for run_id={args.run_id} model={args.model}")
        return 0
    for container in containers:
        print(
            f"stopping {container['name']} ({container['id']}, status={container['status']})"
        )
        stop_container(container["id"])
    print(f"stopped {len(containers)} container(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
