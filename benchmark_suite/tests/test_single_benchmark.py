import json
from pathlib import Path

from ocf_benchmark import cli
from ocf_benchmark.config import config_hash, load_config
from ocf_benchmark.datasets.common import Sample
from ocf_benchmark.reporting import build_summary, write_reports


class FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def model_digest(self, tag):
        return f"digest:{tag}"

    def unload(self, tag):
        return None


class FakeRunner:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def run_combination(self, model, metadata, benchmark, samples, prompt, smoke=False):
        sample_ids = [sample.sample_id for sample in samples]
        self.calls.append((model["id"], benchmark["id"], sample_ids, smoke))
        return {
            "status": "completed",
            "completed": len(sample_ids),
            "total": len(sample_ids),
            "terminal_failures": 0,
            "reason": "",
        }


def test_execute_one_benchmark_uses_same_limited_prefix_for_all_models(
    monkeypatch, tmp_path, suite_dir
):
    cfg = load_config(suite_dir / "configs/experiment.yaml")
    samples = [Sample(str(index), index, Path("/tmp/image.png"), "gt") for index in range(10)]
    monkeypatch.setattr(cli, "_samples", lambda cfg, benchmark: samples)
    monkeypatch.setattr(cli, "OllamaClient", FakeClient)
    monkeypatch.setattr(cli, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(cli, "LegacyPostprocessor", lambda path: object())
    FakeRunner.calls = []
    metadata = {
        model["id"]: {
            "digest": f"digest:{model['tag']}",
            "quantization": "Q4_K_M",
        }
        for model in cfg["models"]
    }

    rc = cli.execute(
        cfg,
        tmp_path,
        metadata,
        smoke=False,
        retry_failed=False,
        benchmark_ids=["tc_str"],
        limit=3,
    )

    assert rc == 0
    assert len(FakeRunner.calls) == 5
    assert {call[1] for call in FakeRunner.calls} == {"tc_str"}
    assert all(call[2] == ["0", "1", "2"] for call in FakeRunner.calls)
    statuses = json.loads((tmp_path / "combination_status.json").read_text())
    assert len([row for row in statuses if row["benchmark"] == "tc_str"]) == 5


def test_single_benchmark_status_merge_preserves_previous_results(monkeypatch, tmp_path, suite_dir):
    cfg = load_config(suite_dir / "configs/experiment.yaml")
    (tmp_path / "combination_status.json").write_text(
        json.dumps(
            [
                {
                    "model": "qwen3_vl_4b",
                    "benchmark": "omnidocbench",
                    "status": "completed",
                }
            ]
        )
    )
    monkeypatch.setattr(
        cli,
        "_samples",
        lambda cfg, benchmark: [Sample("s", 0, Path("/tmp/image.png"), "gt")],
    )
    monkeypatch.setattr(cli, "OllamaClient", FakeClient)
    monkeypatch.setattr(cli, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(cli, "LegacyPostprocessor", lambda path: object())
    metadata = {model["id"]: {"digest": f"digest:{model['tag']}"} for model in cfg["models"]}

    cli.execute(cfg, tmp_path, metadata, False, False, ["tc_str"], 1)
    statuses = json.loads((tmp_path / "combination_status.json").read_text())

    assert any(
        row["model"] == "qwen3_vl_4b"
        and row["benchmark"] == "omnidocbench"
        and row["status"] == "completed"
        for row in statuses
    )


def test_partial_report_has_five_rows_and_does_not_change_full_summary_shape(tmp_path, suite_dir):
    cfg = load_config(suite_dir / "configs/experiment.yaml")
    cfg["_effective_hash"] = config_hash(cfg)
    full = build_summary(cfg, tmp_path, {})
    partial = [row for row in full if row["benchmark"] == "tc_str"]
    output = tmp_path / "partial" / "tc_str" / "limit_10"
    output.mkdir(parents=True)
    write_reports(partial, output, source_run_dir=tmp_path)

    assert len(full) == 15
    assert len(json.loads((output / "summary.json").read_text())) == 5
    assert "TC-STR Exact Accuracy" in (output / "RESULTS.md").read_text()
    assert "OmniDocBench Overall" not in (output / "RESULTS.md").read_text()


def test_partial_summary_excludes_records_beyond_requested_limit(tmp_path, suite_dir):
    cfg = load_config(suite_dir / "configs/experiment.yaml")
    cfg["_effective_hash"] = config_hash(cfg)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    rows = [
        {
            "sample_id": str(index),
            "sample_index": index,
            "status": "completed",
            "truncated": False,
            "metrics": {
                "exact": 1.0,
                "edit_distance": 0.0,
                "gt_length": 1.0,
                "anls": 1.0,
                "containment": 1.0,
                "char_f1": 1.0,
                "raw_exact": 1.0,
                "format_compliance": 1.0,
                "empty": 0.0,
            },
        }
        for index in range(3)
    ]
    (predictions / "qwen3_vl_4b__tc_str.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )

    summary = build_summary(cfg, tmp_path, {}, sample_limit=2)
    qwen_tc = next(
        row for row in summary if row["model"] == "qwen3_vl_4b" and row["benchmark"] == "tc_str"
    )
    assert qwen_tc["n"] == 2
