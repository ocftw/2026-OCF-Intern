from ocf_benchmark.config import config_hash, load_config
from ocf_benchmark.reporting import build_summary


def test_summary_has_15_rows_even_when_all_failed(tmp_path, suite_dir):
    cfg = load_config(suite_dir / "configs/experiment.yaml")
    cfg["_effective_hash"] = config_hash(cfg)
    summary = build_summary(cfg, tmp_path, {})
    assert len(summary) == 15
    assert all(row["status"] in {"failed", "blocked"} for row in summary)
