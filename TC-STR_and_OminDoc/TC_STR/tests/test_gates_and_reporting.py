import json
import tempfile
import unittest
from pathlib import Path

from tc_str_bench.cli import _ready_run
from tc_str_bench.config import Settings, load_settings, run_signature
from tc_str_bench.reporting import _summary, build_main_report
from tc_str_bench.storage import Checkpoint


class GateTests(unittest.TestCase):
    def test_signature_changes_with_digest(self):
        settings = load_settings()
        models = [{"id": m["id"], "exact_tag": m.get("exact_tag"), "digest": "a", "quantization": "Q4"} for m in settings.models]
        first, _ = run_signature(settings, "dataset", models, "ollama 1")
        models[0]["digest"] = "b"
        second, _ = run_signature(settings, "dataset", models, "ollama 1")
        self.assertNotEqual(first, second)

    def test_signature_includes_completion_policy(self):
        settings = load_settings()
        models = [
            {
                "id": m["id"],
                "exact_tag": m.get("exact_tag"),
                "digest": "a",
                "quantization": "Q4",
                "completion_policy": m.get("completion_policy") or {"mode": "strict"},
            }
            for m in settings.models
        ]
        first, _ = run_signature(settings, "dataset", models, "ollama 1")
        models[0]["completion_policy"] = {
            "mode": "allow_nonempty_response_without_terminal_metadata"
        }
        second, _ = run_signature(settings, "dataset", models, "ollama 1")
        self.assertNotEqual(first, second)

    def test_full_gate_requires_ready(self):
        base = load_settings()
        with tempfile.TemporaryDirectory() as directory:
            raw = json.loads(json.dumps(base.raw))
            raw["paths"]["durable_outputs"] = directory
            settings = Settings(raw, base.models, base.root)
            blocked = Path(directory) / "tcstr_2"
            blocked.mkdir()
            (blocked / "smoke_report.json").write_text('{"status":"BLOCKED"}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _ready_run(settings, None)
            ready = Path(directory) / "tcstr_3"
            ready.mkdir()
            (ready / "smoke_report.json").write_text('{"status":"READY_FOR_FULL_RUN"}', encoding="utf-8")
            self.assertEqual(_ready_run(settings, None), ready)

    def test_signature_includes_evaluation_policy(self):
        base = load_settings()
        models = [
            {
                "id": model["id"],
                "exact_tag": model.get("exact_tag"),
                "digest": "a",
                "quantization": "Q4",
            }
            for model in base.models
        ]
        first, _ = run_signature(base, "dataset", models, "ollama 1")
        raw = json.loads(json.dumps(base.raw))
        raw["evaluation_policy"]["version"] = "different"
        changed = Settings(raw, base.models, base.root)
        second, _ = run_signature(changed, "dataset", models, "ollama 1")
        self.assertNotEqual(first, second)


class ReportingTests(unittest.TestCase):
    def test_terminal_failures_remain_in_metric_denominator(self):
        models = [{"id": "m", "logical_name": "Model", "exact_tag": "m:tag"}]
        base = {
            "model_id": "m",
            "model_digest": "digest",
            "anomaly_flags": {
                "empty": False,
                "control_only": False,
                "prompt_echo": False,
                "refusal": False,
                "formatting": False,
                "repetition": False,
                "thinking": False,
                "unrelated": False,
                "truncated": False,
            },
            "completion_validation": {},
            "latency_seconds": 1.0,
            "attempt_count": 1,
            "http_status": 200,
            "primary_metrics": {"em": 1, "cm": 1, "anls": 1, "f1": 1},
            "aligned_metrics": {"em": 1, "cm": 1, "anls": 1, "f1": 1},
        }
        failed = json.loads(json.dumps(base))
        failed.update(
            record_status="request_failure",
            http_status=None,
            primary_metrics={"em": 0, "cm": 0, "anls": 0, "f1": 0},
            aligned_metrics={"em": 0, "cm": 0, "anls": 0, "f1": 0},
        )
        completed = dict(base, record_status="completed")
        result = _summary([completed, failed], models)[0]
        self.assertEqual(result["scored_samples"], 2)
        self.assertEqual(result["request_failures"], 1)
        self.assertEqual(result["primary"]["em"], 0.5)

    def test_model_output_and_ground_truth_are_html_escaped(self):
        base = load_settings()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "sample.jpg").write_bytes(b"small")
            raw = json.loads(json.dumps(base.raw))
            raw["paths"]["dataset"] = str(dataset)
            raw["paths"]["durable_outputs"] = str(root / "outputs")
            settings = Settings(raw, base.models[:1], base.root)
            run_dir = root / "run"
            db = Checkpoint(run_dir / "results.sqlite")
            db.record_success(
                {
                    "run_signature": "sig", "phase": "smoke", "model_id": settings.models[0]["id"],
                    "sample_index": 0, "exact_tag": "exact:tag", "model_digest": "digest",
                    "image_relative_path": "sample.jpg", "image_sha256": "hash",
                    "ground_truth": "<b>甲</b>", "prediction_raw": "<img src=x onerror=alert(1)>",
                    "prediction_primary": "<img src=x onerror=alert(1)>",
                    "prediction_aligned": "", "aligned_changes": ["remove_html_tags"],
                    "primary_metrics": {"em":0,"cm":0,"anls":0,"f1":0},
                    "aligned_metrics": {"em":0,"cm":0,"anls":0,"f1":0},
                    "anomaly_flags": {
                        "empty":False,"control_only":False,"prompt_echo":False,"refusal":False,
                        "formatting":True,"repetition":False,"thinking":False,"unrelated":False,"truncated":False,
                    },
                    "ollama_metadata": {}, "done_reason": "stop", "prompt_eval_count": 1,
                    "eval_count": 1,
                    "completion_validation": {
                        "policy_mode": "strict",
                        "truncation_observable": True,
                    },
                    "latency_seconds": 1, "attempt_count": 1,
                }
            )
            db.close()
            manifest = {
                "samples": [{
                    "index": 0, "image_relative_path": "sample.jpg", "image_sha256": "hash",
                    "ground_truth": "<b>甲</b>", "width": 1, "height": 1,
                }]
            }
            build_main_report(settings, run_dir, "sig", "smoke", manifest, {"run_id":"run"})
            document = (run_dir / "report.html").read_text(encoding="utf-8")
            self.assertNotIn("<img src=x onerror=alert(1)>", document)
            self.assertIn("&lt;img src=x onerror=alert(1)&gt;", document)
            self.assertIn("&lt;b&gt;甲&lt;/b&gt;", document)


if __name__ == "__main__":
    unittest.main()
