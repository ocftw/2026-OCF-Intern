"""單一 concurrency、model-major 的推論 runner。"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import config_hash, effective_options
from .datasets.common import Sample
from .manifest import sha256_file, write_json_atomic
from .ollama_client import OllamaClient
from .resume import JsonlCheckpoint, ResumeKey, completed_keys
from .scorers import tc_str, vistw_mcq


def _duration_metrics(data: dict[str, Any]) -> dict[str, Any]:
    eval_duration = data.get("eval_duration") or 0
    eval_count = data.get("eval_count") or 0
    return {
        "ollama_total_duration": data.get("total_duration"),
        "load_duration": data.get("load_duration"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "prompt_eval_duration": data.get("prompt_eval_duration"),
        "eval_count": eval_count,
        "eval_duration": eval_duration,
        "output_tokens_per_second": (eval_count / (eval_duration / 1e9) if eval_duration else None),
    }


class BenchmarkRunner:
    def __init__(
        self,
        cfg: dict[str, Any],
        client: OllamaClient,
        run_dir: Path,
        run_id: str,
        legacy_postprocessor: tc_str.LegacyPostprocessor,
        retry_failed: bool = False,
    ):
        self.cfg = cfg
        self.client = client
        self.run_dir = run_dir
        self.run_id = run_id
        self.legacy = legacy_postprocessor
        self.retry_failed = retry_failed
        self.progress_path = run_dir / "progress.json"

    def run_combination(
        self,
        model: dict[str, Any],
        metadata: dict[str, Any],
        benchmark: dict[str, Any],
        samples: Iterable[Sample],
        prompt_template: str,
        smoke: bool = False,
    ) -> dict[str, Any]:
        sample_list = list(samples)
        if smoke:
            sample_list = sample_list[: self.cfg["smoke_samples_per_combination"]]
        prediction_path = self.run_dir / "predictions" / f"{model['id']}__{benchmark['id']}.jsonl"
        checkpoint = JsonlCheckpoint(prediction_path)
        done = completed_keys(prediction_path, include_failed=not self.retry_failed)
        options = effective_options(benchmark, model["id"])
        digest = metadata["digest"]

        if sample_list:
            warm_prompt = self._prompt(benchmark["id"], prompt_template, sample_list[0])
            warmup = self.client.chat(
                model=model["tag"],
                prompt=warm_prompt,
                image_path=sample_list[0].image_path,
                options=options,
                timeout=benchmark["timeout_seconds"],
            )
            warmup_path = self.run_dir / "logs" / "warmups.jsonl"
            warmup_path.parent.mkdir(parents=True, exist_ok=True)
            with warmup_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "model": model["id"],
                            "benchmark": benchmark["id"],
                            "sample_id": sample_list[0].sample_id,
                            "status": warmup.status,
                            "wall_clock_latency": warmup.wall_clock_latency,
                            "load_duration": (warmup.data or {}).get("load_duration"),
                            "excluded_from_accuracy": True,
                            "excluded_from_steady_state_latency": True,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())

        completed = 0
        terminal_failures = 0
        for sample in sample_list:
            prompt = self._prompt(benchmark["id"], prompt_template, sample)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            key = ResumeKey(
                config_hash(self.cfg),
                model["id"],
                digest,
                benchmark["id"],
                benchmark["revision"],
                sample.sample_id,
                prompt_hash,
            )
            if key.text() in done:
                completed += 1
                continue
            result = self.client.chat(
                model=model["tag"],
                prompt=prompt,
                image_path=sample.image_path,
                options=options,
                timeout=benchmark["timeout_seconds"],
            )
            prediction = result.text if result.status == "completed" else ""
            metrics: dict[str, Any] = {}
            parsed_answer = None
            legacy_prediction = None
            if benchmark["id"] == "tc_str":
                normalized = tc_str.normalize_minimal(prediction)
                metrics = tc_str.score(prediction, str(sample.ground_truth))
                legacy_prediction = self.legacy.clean(prediction)
                legacy_metrics = tc_str.score(legacy_prediction, str(sample.ground_truth))
            elif benchmark["id"] == "vistw_mcq":
                normalized = prediction.strip()
                metrics = vistw_mcq.score(prediction, sample.ground_truth["answer"])
                parsed_answer = metrics["parsed_answer"]
                legacy_metrics = None
            else:
                normalized = prediction.strip()
                legacy_metrics = None
                official_dir = self.run_dir / "predictions" / "omnidocbench_md" / model["id"]
                official_dir.mkdir(parents=True, exist_ok=True)
                (official_dir / f"{Path(sample.dataset_key).stem}.md").write_text(
                    prediction, encoding="utf-8"
                )
            data = result.data or {}
            record = {
                "schema_version": 1,
                "run_id": self.run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "config_hash": config_hash(self.cfg),
                "model_logical_id": model["id"],
                "requested_model_tag": model["tag"],
                "resolved_model_digest": digest,
                "quantization": metadata.get("quantization", ""),
                "benchmark": benchmark["id"],
                "benchmark_revision": benchmark["revision"],
                "split": benchmark["split"],
                "subject": sample.subject or None,
                "sample_id": sample.sample_id,
                "sample_index": sample.sample_index,
                "image_path_or_dataset_key": sample.dataset_key or str(sample.image_path),
                "image_sha256": sha256_file(sample.image_path),
                "prompt_text": prompt,
                "prompt_hash": prompt_hash,
                "effective_ollama_options": options,
                "ground_truth": sample.ground_truth,
                "raw_response": result.text,
                "minimally_normalized_prediction": normalized,
                "parsed_answer": parsed_answer,
                "legacy_cleaned_prediction": legacy_prediction,
                "legacy_metrics": legacy_metrics,
                "legacy_postprocessor_sha256": (
                    self.legacy.sha256 if benchmark["id"] == "tc_str" else None
                ),
                "scorer_version": "ocf-vlm-benchmark/1.0.0",
                "metrics": metrics,
                "wall_clock_latency": result.wall_clock_latency,
                **_duration_metrics(data),
                "done_reason": data.get("done_reason"),
                "truncated": result.truncated,
                "retry_count": result.retry_count,
                "http_status": result.http_status,
                "error_type": result.error_type or None,
                "error_message": result.error_message or None,
                "status": result.status,
            }
            checkpoint.append(record)
            terminal_failures += int(result.status == "terminal_failure")
            completed += 1
            self._progress(model["id"], benchmark["id"], completed, len(sample_list))
        return {
            "status": "failed" if smoke and terminal_failures else "completed",
            "completed": completed,
            "total": len(sample_list),
            "terminal_failures": terminal_failures,
            "reason": (
                f"smoke 有 {terminal_failures} 筆 terminal failure"
                if smoke and terminal_failures
                else ""
            ),
        }

    @staticmethod
    def _prompt(benchmark_id: str, template: str, sample: Sample) -> str:
        if benchmark_id == "vistw_mcq":
            return template.format(**sample.ground_truth)
        return template

    def _progress(self, model: str, benchmark: str, completed: int, total: int) -> None:
        write_json_atomic(
            self.progress_path,
            {
                "run_id": self.run_id,
                "current_model": model,
                "current_benchmark": benchmark,
                "completed": completed,
                "total": total,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
