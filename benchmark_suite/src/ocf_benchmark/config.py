"""YAML 載入、環境變數覆寫、schema 驗證與穩定雜湊。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")
REQUIRED_BENCHMARKS = {"omnidocbench", "tc_str", "vistw_mcq"}


class ConfigError(ValueError):
    pass


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ConfigError(f"缺少必要環境變數 {name}")

    return _ENV.sub(replace, value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "extends" in raw:
        parent = load_config(path.parent / raw.pop("extends"))
        benchmark_overrides = raw.pop("benchmarks_overrides", {})
        raw = _deep_merge(parent, raw)
        for benchmark in raw["benchmarks"]:
            if benchmark["id"] in benchmark_overrides:
                merged = _deep_merge(benchmark, benchmark_overrides[benchmark["id"]])
                benchmark.clear()
                benchmark.update(merged)
    cfg = _expand_env(raw)
    cfg["_config_path"] = str(path)
    suite_dir = path.parent.parent
    prompt_hashes = {}
    for benchmark in cfg.get("benchmarks", []):
        prompt_path = Path(benchmark.get("prompt", ""))
        if not prompt_path.is_absolute():
            prompt_path = suite_dir / prompt_path
        if not prompt_path.exists():
            raise ConfigError(f"找不到 prompt: {prompt_path}")
        prompt_hashes[benchmark["id"]] = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    cfg["prompt_hashes"] = prompt_hashes
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg.get("schema_version") != 1:
        raise ConfigError("schema_version 必須是 1")
    models = cfg.get("models", [])
    benchmarks = cfg.get("benchmarks", [])
    model_ids = [m.get("id") for m in models]
    benchmark_ids = [b.get("id") for b in benchmarks]
    if len(models) != 5 or len(set(model_ids)) != 5:
        raise ConfigError("必須設定 5 個不重複 model id")
    if set(benchmark_ids) != REQUIRED_BENCHMARKS or len(benchmarks) != 3:
        raise ConfigError("必須且只能設定 omnidocbench、tc_str、vistw_mcq")
    if cfg.get("ollama", {}).get("concurrency") != 1:
        raise ConfigError("正式 benchmark concurrency 必須固定為 1")
    if cfg.get("ollama", {}).get("endpoint") != "/api/chat":
        raise ConfigError("主實驗必須使用 /api/chat")
    for key in ("minimum_free_disk_gb", "minimum_results_free_disk_gb"):
        value = cfg.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"{key} 必須是正數")
    for model in models:
        if not model.get("tag"):
            raise ConfigError(f"model {model.get('id')} 缺少 tag")
    for benchmark in benchmarks:
        if not benchmark.get("revision"):
            raise ConfigError(f"{benchmark['id']} 必須固定 dataset revision")
        options = benchmark.get("options", {})
        required = {"temperature", "seed", "repeat_penalty", "num_ctx", "num_predict"}
        if not required.issubset(options):
            raise ConfigError(f"{benchmark['id']} options 缺少 {sorted(required - set(options))}")
        if options["temperature"] != 0 or options["seed"] != 42:
            raise ConfigError("主實驗必須 temperature=0、seed=42")
        if options["repeat_penalty"] != 1.0:
            raise ConfigError("主實驗 repeat_penalty 必須固定 1.0")
    if len(combinations(cfg)) != 15:
        raise ConfigError("組合數必須剛好是 15")


def combinations(cfg: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [(model, bench) for model in cfg["models"] for bench in cfg["benchmarks"]]


def canonical_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def config_hash(cfg: dict[str, Any]) -> str:
    encoded = json.dumps(canonical_config(cfg), ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def benchmark_by_id(cfg: dict[str, Any], benchmark_id: str) -> dict[str, Any]:
    return next(b for b in cfg["benchmarks"] if b["id"] == benchmark_id)


def effective_options(benchmark: dict[str, Any], model_id: str) -> dict[str, Any]:
    options = dict(benchmark["options"])
    options.update(benchmark.get("model_options", {}).get(model_id, {}))
    return options


def resolve_path(cfg: dict[str, Any], value: str) -> Path:
    repo_root = Path(cfg["_config_path"]).parents[2]
    path = Path(value)
    return path if path.is_absolute() else repo_root / path
