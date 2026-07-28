"""Benchmark suite command line orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

from .config import (
    benchmark_by_id,
    canonical_config,
    config_hash,
    load_config,
    resolve_path,
)
from .datasets import omnidocbench as omni_data
from .datasets import tc_str as tc_data
from .datasets import vistw_mcq as vistw_data
from .manifest import create_manifest, write_json_atomic
from .ollama_client import OllamaClient
from .reporting import build_summary, write_pairwise, write_reports
from .resume import find_resumable_run
from .runner import BenchmarkRunner
from .scorers.omnidocbench import run_official_evaluator
from .scorers.tc_str import LegacyPostprocessor, score as score_tc

SUITE = Path(__file__).resolve().parents[2]
REPO_ROOT = SUITE.parent


def preflight(cfg: dict[str, Any], strict: bool = True) -> list[str]:
    errors = []
    if platform.system() != "Linux":
        errors.append("只支援 Linux/Ubuntu")
    if sys.version_info < (3, 10):
        errors.append("需要 Python 3.10+")
    for command in ("curl", "git"):
        if shutil.which(command) is None:
            errors.append(f"找不到 {command}")
    if strict:
        for command in ("ollama", "docker"):
            if shutil.which(command) is None:
                errors.append(f"找不到 {command}")
        try:
            response = requests.get(f"{cfg['ollama']['host'].rstrip('/')}/api/tags", timeout=5)
            if not response.ok:
                errors.append(f"Ollama health HTTP {response.status_code}")
        except requests.RequestException as exc:
            errors.append(f"Ollama 無法連線: {exc}")
        if shutil.which("docker"):
            proc = subprocess.run(["docker", "info"], text=True, capture_output=True, check=False)
            if proc.returncode:
                errors.append("Docker daemon 不可用（OmniDocBench 官方 evaluator 必需）")
    runs_dir = resolve_path(cfg, cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(runs_dir, os.W_OK):
        errors.append(f"output directory 不可寫: {runs_dir}")
    free_gb = shutil.disk_usage(runs_dir).free / 1024**3
    if strict and free_gb < float(cfg["minimum_free_disk_gb"]):
        errors.append(f"磁碟僅剩 {free_gb:.1f} GiB，profile 要求 {cfg['minimum_free_disk_gb']} GiB")
    return errors


def _load_model_metadata(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return items, {item["logical_id"]: item for item in items}


def _samples(cfg: dict[str, Any], benchmark: dict[str, Any]):
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    cache_dir = resolve_path(cfg, cfg["paths"]["cache_dir"])
    if benchmark["id"] == "tc_str":
        return tc_data.load_samples(data_dir / "TC-STR", benchmark["split"])
    if benchmark["id"] == "vistw_mcq":
        return vistw_data.load_samples(
            benchmark["dataset"],
            benchmark["revision"],
            cache_dir / "materialized_images" / "vistw_mcq",
            benchmark["split"],
            cache_dir / "hf" / "vistw_mcq" / benchmark["revision"],
        )
    return omni_data.load_samples(
        benchmark["dataset"],
        benchmark["revision"],
        cache_dir / "materialized_images" / "omnidocbench",
        snapshot=cache_dir / "hf" / "omnidocbench" / benchmark["revision"],
    )


def execute(
    cfg: dict[str, Any],
    run_dir: Path,
    metadata_map: dict[str, dict[str, Any]],
    smoke: bool,
    retry_failed: bool,
) -> int:
    target = run_dir / "smoke" if smoke else run_dir
    target.mkdir(parents=True, exist_ok=True)
    legacy = LegacyPostprocessor(REPO_ROOT / "ablation_experiment/postprocess.py")
    client = OllamaClient(
        cfg["ollama"]["host"],
        cfg["ollama"]["retry_delays_seconds"],
        cfg["failure_policy"]["max_attempts"],
        cfg["ollama"]["circuit_breaker_failures"],
        cfg["ollama"]["circuit_breaker_poll_seconds"],
    )
    runner = BenchmarkRunner(cfg, client, target, run_dir.name, legacy, retry_failed=retry_failed)
    combo_status = []
    sample_cache: dict[str, list[Any]] = {}
    for benchmark in cfg["benchmarks"]:
        sample_cache[benchmark["id"]] = list(_samples(cfg, benchmark))
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not smoke:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for dataset in manifest.get("datasets", []):
            benchmark_id = next(
                b["id"] for b in cfg["benchmarks"] if b["dataset"] == dataset["name"]
            )
            samples = sample_cache[benchmark_id]
            order = "\n".join(sample.sample_id for sample in samples).encode()
            dataset["sample_count"] = len(samples)
            dataset["sample_order_sha256"] = hashlib.sha256(order).hexdigest()
        write_json_atomic(manifest_path, manifest)
    for model in cfg["models"]:  # model-major；完成三個 benchmark 才 unload
        metadata = metadata_map[model["id"]]
        try:
            actual_digest = client.model_digest(model["tag"])
        except Exception as exc:
            actual_digest = f"ERROR:{exc}"
        if actual_digest != metadata["digest"]:
            for benchmark in cfg["benchmarks"]:
                combo_status.append(
                    {
                        "model": model["id"],
                        "benchmark": benchmark["id"],
                        "status": "failed",
                        "reason": (
                            "model digest 在 run 中改變；拒絕混用："
                            f"manifest={metadata['digest']} actual={actual_digest}"
                        ),
                    }
                )
            write_json_atomic(target / "combination_status.json", combo_status)
            continue
        for benchmark in cfg["benchmarks"]:
            try:
                prompt = (SUITE / benchmark["prompt"]).read_text(encoding="utf-8").strip()
                status = runner.run_combination(
                    model, metadata, benchmark, sample_cache[benchmark["id"]], prompt, smoke=smoke
                )
            except Exception as exc:  # 組合失敗不可阻止其他 14 組
                status = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            combo_status.append({"model": model["id"], "benchmark": benchmark["id"], **status})
            write_json_atomic(target / "combination_status.json", combo_status)
        try:
            client.unload(model["tag"])
        except Exception as exc:
            combo_status.append(
                {
                    "model": model["id"],
                    "benchmark": "_unload",
                    "status": "warning",
                    "reason": str(exc),
                }
            )
    failures = [c for c in combo_status if c["status"] == "failed"]
    return 1 if failures else 0


def score_omnidoc(cfg: dict[str, Any], run_dir: Path) -> None:
    benchmark = benchmark_by_id(cfg, "omnidocbench")
    cache = resolve_path(cfg, cfg["paths"]["cache_dir"])
    gt = omni_data.locate_ground_truth(cache / "hf" / "omnidocbench" / benchmark["revision"])
    evaluator = cache / "evaluators" / "omnidocbench"
    for model in cfg["models"]:
        predictions = run_dir / "predictions" / "omnidocbench_md" / model["id"]
        output = run_dir / "scores" / f"{model['id']}__omnidocbench_artifacts"
        try:
            result = run_official_evaluator(
                evaluator_dir=evaluator,
                gt_json=gt,
                predictions=predictions,
                output_dir=output,
                docker_image=benchmark["evaluator"]["docker_image"],
            )
        except Exception as exc:
            result = {"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"}
        (run_dir / "scores").mkdir(parents=True, exist_ok=True)
        (run_dir / "scores" / f"{model['id']}__omnidocbench.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def new_run(cfg: dict[str, Any], requested: str | None = None, resume: bool = False) -> Path:
    runs = resolve_path(cfg, cfg["paths"]["runs_dir"])
    if resume:
        previous = find_resumable_run(runs, config_hash(cfg))
        if previous is not None:
            return previous
    run_id = requested or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = runs / run_id
    for child in ("logs", "predictions", "scores"):
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def run_calibration(cfg: dict[str, Any], model_id: str, limit: int) -> Path:
    """只在 TC-STR train 比較 repeat penalty；永不寫入正式 runs/。"""
    model = next((item for item in cfg["models"] if item["id"] == model_id), None)
    if model is None:
        raise ValueError(f"未知 model id: {model_id}")
    benchmark = benchmark_by_id(cfg, "tc_str")
    samples = tc_data.load_samples(resolve_path(cfg, cfg["paths"]["data_dir"]) / "TC-STR", "train")[
        :limit
    ]
    prompt = (SUITE / benchmark["prompt"]).read_text(encoding="utf-8").strip()
    output = SUITE / "calibration" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    client = OllamaClient(cfg["ollama"]["host"])
    for penalty in (1.0, 1.1, 1.3):
        path = output / f"{model_id}__repeat_penalty_{penalty}.jsonl"
        options = dict(benchmark["options"])
        options["repeat_penalty"] = penalty
        with path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                result = client.chat(
                    model=model["tag"],
                    prompt=prompt,
                    image_path=sample.image_path,
                    options=options,
                    timeout=benchmark["timeout_seconds"],
                )
                prediction = result.text if result.status == "completed" else ""
                row = {
                    "informal_calibration": True,
                    "split": "train",
                    "model": model_id,
                    "repeat_penalty": penalty,
                    "sample_id": sample.sample_id,
                    "prediction": prediction,
                    "ground_truth": sample.ground_truth,
                    "metrics": score_tc(prediction, str(sample.ground_truth)),
                    "status": result.status,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    client.unload(model["tag"])
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SUITE / "configs/experiment.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--non-strict", action="store_true")
    start = sub.add_parser("init-run")
    start.add_argument("--run-id")
    start.add_argument("--resume", action="store_true")
    start.add_argument("--models-metadata", required=True)
    run = sub.add_parser("execute")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--models-metadata", required=True)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    score = sub.add_parser("score")
    score.add_argument("--run-dir", required=True)
    report = sub.add_parser("report")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--models-metadata", required=True)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--split", default="train", choices=["train"])
    calibrate.add_argument("--model", default="qwen3_vl_4b")
    calibrate.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg["_effective_hash"] = config_hash(cfg)
    if args.command == "preflight":
        errors = preflight(cfg, strict=not args.non_strict)
        if errors:
            print("\n".join(f"[FAIL] {error}" for error in errors), file=sys.stderr)
            print(
                f"建議低資源指令：{SUITE / 'run_all.sh'} --config "
                f"{SUITE / 'configs/constrained.yaml'}",
                file=sys.stderr,
            )
            return 1
        print("[OK] preflight")
        return 0
    if args.command == "init-run":
        models, _ = _load_model_metadata(Path(args.models_metadata))
        run_dir = new_run(cfg, args.run_id, args.resume)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            old_digests = {
                item.get("logical_id"): item.get("digest") for item in old.get("models", [])
            }
            new_digests = {item.get("logical_id"): item.get("digest") for item in models}
            if old_digests != new_digests:
                run_dir = new_run(cfg, args.run_id, resume=False)
                manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            manifest = create_manifest(cfg, REPO_ROOT, models, run_dir.name)
            write_json_atomic(manifest_path, manifest)
            (run_dir / "effective_config.yaml").write_text(
                yaml.safe_dump(canonical_config(cfg), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        print(run_dir)
        return 0
    if args.command == "execute":
        _, metadata = _load_model_metadata(Path(args.models_metadata))
        return execute(cfg, Path(args.run_dir), metadata, args.smoke, args.retry_failed)
    if args.command == "score":
        score_omnidoc(cfg, Path(args.run_dir))
        return 0
    if args.command == "report":
        _, metadata = _load_model_metadata(Path(args.models_metadata))
        summary = build_summary(cfg, Path(args.run_dir), metadata)
        write_pairwise(cfg, Path(args.run_dir))
        write_reports(summary, Path(args.run_dir))
        manifest_path = Path(args.run_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "completed"
        manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(manifest_path, manifest)
        return 0
    if args.command == "calibrate":
        if args.limit < 1:
            raise ValueError("--limit 必須大於 0")
        output = run_calibration(cfg, args.model, args.limit)
        print(
            "非正式 calibration 完成：train split，repeat_penalty=[1.0,1.1,1.3]；"
            f"不屬於 15 組正式結果。輸出：{output}"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
