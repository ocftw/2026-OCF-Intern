"""Run manifest 與硬體／Git provenance。"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import canonical_config, config_hash


def _command(command: list[str]) -> str:
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    return (proc.stdout or proc.stderr).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(repo_root: Path) -> dict[str, Any]:
    commit = _command(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    porcelain = _command(["git", "-C", str(repo_root), "status", "--porcelain=v1"])
    diff = bytearray(
        subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--binary", "HEAD"],
            capture_output=True,
            check=False,
        ).stdout
    )
    untracked = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=False,
    ).stdout.split(b"\0")
    for encoded in sorted(path for path in untracked if path):
        path = repo_root / encoded.decode()
        if path.is_file():
            diff.extend(b"\0UNTRACKED\0" + encoded + b"\0" + path.read_bytes())
    return {
        "commit": commit,
        "dirty": bool(porcelain),
        "dirty_diff_sha256": hashlib.sha256(bytes(diff)).hexdigest(),
    }


def hardware() -> dict[str, str]:
    return {
        "hostname": socket.gethostname(),
        "uname": platform.platform(),
        "ubuntu": _command(
            ["bash", "-lc", 'source /etc/os-release; echo "${PRETTY_NAME:-unknown}"']
        ),
        "python": sys.version,
        "ollama": _command(["ollama", "--version"]),
        "gpu": _command(
            [
                "bash",
                "-lc",
                "command -v nvidia-smi >/dev/null && "
                "nvidia-smi --query-gpu=name,driver_version,memory.total "
                "--format=csv,noheader || echo unavailable",
            ]
        ),
        "cpu": _command(["bash", "-lc", "lscpu | sed -n '1,20p'"]),
        "ram": _command(["free", "-h"]),
    }


def create_manifest(
    cfg: dict[str, Any], repo_root: Path, models: list[dict[str, Any]], run_id: str
) -> dict[str, Any]:
    suite = repo_root / "benchmark_suite"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "config_hash": config_hash(cfg),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "ended_at_utc": None,
        "git": git_state(repo_root),
        "environment": hardware(),
        "models": models,
        "model_order": [m["id"] for m in cfg["models"]],
        "datasets": [
            {
                "name": b["dataset"],
                "revision": b["revision"],
                "split": b["split"],
                "expected_samples": b.get("expected_samples"),
                "expected_subjects": b.get("expected_subjects"),
                "evaluator": b.get("evaluator"),
            }
            for b in cfg["benchmarks"]
        ],
        "prompts": {b["id"]: sha256_file(suite / b["prompt"]) for b in cfg["benchmarks"]},
        "effective_config": canonical_config(cfg),
        "random_seed": cfg["seed"],
        "failure_policy": cfg["failure_policy"],
        "package_lock_sha256": sha256_file(suite / "requirements.lock"),
    }


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
