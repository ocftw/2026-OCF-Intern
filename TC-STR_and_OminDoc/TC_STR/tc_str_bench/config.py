from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .util import canonical_json, source_fingerprint


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    models: list[dict[str, Any]]
    root: Path = ROOT

    @property
    def dataset_dir(self) -> Path:
        return Path(self.raw["paths"]["dataset"])

    @property
    def output_root(self) -> Path:
        return Path(self.raw["paths"]["durable_outputs"])

    @property
    def prompt(self) -> str:
        return str(self.raw["prompt"])

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.raw["generation"])

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    @property
    def source_hash(self) -> str:
        return source_fingerprint(self.root)


def load_settings(root: Path = ROOT) -> Settings:
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    model_doc = yaml.safe_load((root / "models.yaml").read_text(encoding="utf-8"))
    models = model_doc.get("models", [])
    if len(models) != 8 or len({m["id"] for m in models}) != 8:
        raise ValueError("models.yaml 必須包含 8 個不重複 logical model id")
    allowed_completion_modes = {
        "strict",
        "allow_nonempty_response_without_terminal_metadata",
    }
    for model in models:
        policy = model.get("completion_policy") or {"mode": "strict"}
        mode = policy.get("mode", "strict")
        if mode not in allowed_completion_modes:
            raise ValueError(f"{model['id']} completion_policy.mode 不支援: {mode}")
        if mode != "strict" and not (
            policy.get("approved_by_user") is True
            and policy.get("approved_at")
            and policy.get("rationale")
            and policy.get("accepted_limitations")
        ):
            raise ValueError(f"{model['id']} 放寬 completion policy 必須記錄核准、理由與限制")
    return Settings(config, models, root)


def run_signature(
    settings: Settings,
    dataset_fingerprint: str,
    resolved_models: list[dict[str, Any]],
    ollama_version: str,
) -> tuple[str, dict[str, Any]]:
    material = {
        "schema": 1,
        "benchmark": settings.raw["benchmark"],
        "dataset_fingerprint": dataset_fingerprint,
        "models": [
            {
                "id": item["id"],
                "exact_tag": item.get("exact_tag"),
                "digest": item.get("digest"),
                "quantization": item.get("quantization"),
                "completion_policy": item.get("completion_policy") or {"mode": "strict"},
            }
            for item in resolved_models
        ],
        "endpoint": settings.raw["ollama"]["endpoint"],
        "prompt_sha256": settings.prompt_hash,
        "options": settings.options,
        "ollama_version": ollama_version,
        "versions": settings.raw["versions"],
        "evaluation_policy": settings.raw["evaluation_policy"],
        "source_fingerprint": settings.source_hash,
    }
    return hashlib.sha256(canonical_json(material).encode()).hexdigest(), material
