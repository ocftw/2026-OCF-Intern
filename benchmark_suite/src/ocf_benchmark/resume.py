"""Append-only JSONL checkpoint 與不可混用的 resume key。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ResumeKey:
    config_hash: str
    model_id: str
    model_digest: str
    benchmark: str
    benchmark_revision: str
    sample_id: str
    prompt_hash: str

    def text(self) -> str:
        return "\x1f".join(str(v) for v in self.__dict__.values())


def key_from_record(record: dict[str, Any]) -> ResumeKey:
    return ResumeKey(
        record["config_hash"],
        record["model_logical_id"],
        record["resolved_model_digest"],
        record["benchmark"],
        record["benchmark_revision"],
        str(record["sample_id"]),
        record["prompt_hash"],
    )


def completed_keys(path: Path, include_failed: bool = True) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # 僅容忍 crash 留下的最後半行
            terminal = row.get("status") == "completed" or (
                include_failed and row.get("status") == "terminal_failure"
            )
            if terminal:
                keys.add(key_from_record(row).text())
    return keys


class JsonlCheckpoint:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def append(self, record: dict[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def append_once(self, record: dict[str, Any]) -> bool:
        if key_from_record(record).text() in completed_keys(self.path):
            return False
        self.append(record)
        return True


def find_resumable_run(runs_dir: Path, wanted_hash: str) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates: Iterable[Path] = sorted(runs_dir.iterdir(), reverse=True)
    for run in candidates:
        manifest = run / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("config_hash") == wanted_hash and data.get("status") not in {"completed"}:
            return run
    return None
