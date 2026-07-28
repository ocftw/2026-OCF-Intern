from copy import deepcopy

import pytest

from ocf_benchmark.config import (
    ConfigError,
    combinations,
    config_hash,
    load_config,
    validate_config,
)


def test_schema_and_exactly_15_combinations(suite_dir):
    cfg = load_config(suite_dir / "configs/experiment.yaml")
    assert len(combinations(cfg)) == 15
    assert [m["id"] for m in cfg["models"]] == [
        "qwen3_vl_4b",
        "gemma4_e2b",
        "gemma4_e4b",
        "sea_lion_4b",
        "smolvlm2_2_2b",
    ]


def test_invalid_concurrency_rejected(suite_dir):
    cfg = load_config(suite_dir / "configs/experiment.yaml")
    cfg["ollama"]["concurrency"] = 2
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_config_hash_changes(suite_dir):
    cfg = load_config(suite_dir / "configs/experiment.yaml")
    changed = deepcopy(cfg)
    changed["seed"] = 43
    assert config_hash(cfg) != config_hash(changed)


def test_constrained_is_distinct(suite_dir):
    main = load_config(suite_dir / "configs/experiment.yaml")
    constrained = load_config(suite_dir / "configs/constrained.yaml")
    assert config_hash(main) != config_hash(constrained)
    assert constrained["profile"] == "constrained"
    assert constrained["minimum_free_disk_gb"] == 45
    assert constrained["minimum_results_free_disk_gb"] == 10


def test_disk_thresholds_must_be_positive(suite_dir):
    cfg = load_config(suite_dir / "configs/experiment.yaml")
    cfg["minimum_results_free_disk_gb"] = 0
    with pytest.raises(ConfigError, match="minimum_results_free_disk_gb"):
        validate_config(cfg)
