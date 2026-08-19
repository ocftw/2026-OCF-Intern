#!/usr/bin/env python3
"""Small shared helpers for finding/stopping docker containers by bind mount.

Used by evaluate_model.py (pre-flight cleanup + stall recovery),
evaluation_status.py (discovery for status/watch), and stop_evaluation.py
(manual cleanup). Never guesses container identity by name pattern alone;
always confirms via an actual bind-mount source path, since container names
are not guaranteed (older invocations may have used auto-generated names).
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
from typing import Any


def sanitize_container_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "-", value)
    return cleaned[:128]


def _docker_ps_ids() -> list[str]:
    listing = subprocess.run(
        ["docker", "ps", "-q"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if listing.returncode != 0 or not listing.stdout.strip():
        return []
    return listing.stdout.split()


def _inspect(container_id: str) -> dict[str, Any] | None:
    inspected = subprocess.run(
        ["docker", "inspect", container_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if inspected.returncode != 0:
        return None
    try:
        return json.loads(inspected.stdout)[0]
    except (json.JSONDecodeError, IndexError):
        return None


def find_containers_with_mount_prefix(prefix: pathlib.Path) -> list[dict[str, str]]:
    """Return running containers that bind-mount a path under ``prefix``."""
    wanted_prefix = str(prefix.resolve())
    found: list[dict[str, str]] = []
    for container_id in _docker_ps_ids():
        data = _inspect(container_id)
        if data is None:
            continue
        sources = [
            str(pathlib.Path(mount["Source"]).resolve())
            for mount in data.get("Mounts", [])
            if mount.get("Source")
        ]
        if any(source == wanted_prefix or source.startswith(wanted_prefix + "/") for source in sources):
            found.append(
                {
                    "id": container_id[:12],
                    "full_id": container_id,
                    "name": data.get("Name", "").lstrip("/"),
                    "status": data.get("State", {}).get("Status", "unknown"),
                    "started_at": data.get("State", {}).get("StartedAt", ""),
                }
            )
    return found


def find_container_with_mount_source(exact_path: pathlib.Path) -> dict[str, str] | None:
    """Return the single running container that mounts ``exact_path`` exactly, if any."""
    wanted = str(exact_path.resolve())
    for container_id in _docker_ps_ids():
        data = _inspect(container_id)
        if data is None:
            continue
        sources = {
            str(pathlib.Path(mount["Source"]).resolve())
            for mount in data.get("Mounts", [])
            if mount.get("Source")
        }
        if wanted in sources:
            return {
                "id": container_id[:12],
                "full_id": container_id,
                "name": data.get("Name", "").lstrip("/"),
                "status": data.get("State", {}).get("Status", "unknown"),
                "started_at": data.get("State", {}).get("StartedAt", ""),
            }
    return None


def stop_container(name_or_id: str, timeout: int = 5) -> None:
    """Best-effort graceful stop; docker SIGKILLs after ``timeout`` seconds itself."""
    subprocess.run(
        ["docker", "stop", "--time", str(timeout), name_or_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def container_cpu_percent(name_or_id: str) -> float | None:
    result = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}", name_or_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return float(result.stdout.strip().rstrip("%"))
    except ValueError:
        return None
