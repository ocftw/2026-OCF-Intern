#!/usr/bin/env python3
"""Clone/fetch evaluator，並 checkout YAML 固定 commit（非 dataset revision）。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE / "src"))
from ocf_benchmark.config import load_config, resolve_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SUITE / "configs/experiment.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolve_path(cfg, cfg["paths"]["cache_dir"]) / "evaluators"
    root.mkdir(parents=True, exist_ok=True)
    for benchmark in cfg["benchmarks"]:
        evaluator = benchmark.get("evaluator")
        if not evaluator:
            continue
        target = root / benchmark["id"]
        if not (target / ".git").exists():
            subprocess.run(
                ["git", "clone", "--filter=blob:none", evaluator["repository"], str(target)],
                check=True,
            )
        subprocess.run(
            ["git", "-C", str(target), "fetch", "origin", evaluator["commit"]], check=True
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", evaluator["commit"]], check=True
        )
        actual = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != evaluator["commit"]:
            raise RuntimeError(f"{benchmark['id']} evaluator commit mismatch")
        if evaluator.get("docker_image"):
            subprocess.run(["docker", "pull", evaluator["docker_image"]], check=True)
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--entrypoint",
                    "python",
                    evaluator["docker_image"],
                    "--version",
                ],
                check=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
