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


def preflight(
    cfg: dict[str, Any],
    strict: bool = True,
    benchmark_ids: list[str] | None = None,
) -> list[str]:
    errors = []
    if platform.system() != "Linux":
        errors.append("只支援 Linux/Ubuntu")
    if sys.version_info < (3, 10):
        errors.append("需要 Python 3.10+")
    for command in ("curl", "git"):
        if shutil.which(command) is None:
            errors.append(f"找不到 {command}")
    if strict:
        required_commands = ["ollama"]
        if benchmark_ids is None or "omnidocbench" in benchmark_ids:
            required_commands.append("docker")
        for command in required_commands:
            if shutil.which(command) is None:
                errors.append(f"找不到 {command}")
        try:
            response = requests.get(f"{cfg['ollama']['host'].rstrip('/')}/api/tags", timeout=5)
            if not response.ok:
                errors.append(f"Ollama health HTTP {response.status_code}")
        except requests.RequestException as exc:
            errors.append(f"Ollama 無法連線: {exc}")
        if (benchmark_ids is None or "omnidocbench" in benchmark_ids) and shutil.which("docker"):
            proc = subprocess.run(["docker", "info"], text=True, capture_output=True, check=False)
            if proc.returncode:
                errors.append("Docker daemon 不可用（OmniDocBench 官方 evaluator 必需）")
    storage = {
        "data": resolve_path(cfg, cfg["paths"]["data_dir"]),
        "cache": resolve_path(cfg, cfg["paths"]["cache_dir"]),
        "results": resolve_path(cfg, cfg["paths"]["runs_dir"]),
    }
    usable: dict[str, Path] = {}
    for label, path in storage.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"{label} directory 無法建立: {path}: {exc}")
            continue
        if not os.access(path, os.W_OK):
            errors.append(f"{label} directory 不可寫: {path}")
            continue
        usable[label] = path

    # data/cache 是可重建 scratch；runs 是不可遺失的正式結果。若它們剛好位於
    # 同一檔案系統，只套用兩項門檻的最大值，避免重複計算相同 free space。
    requirements: dict[int, dict[str, Any]] = {}
    for label, path in usable.items():
        device = path.stat().st_dev
        required = float(
            cfg["minimum_results_free_disk_gb"]
            if label == "results"
            else cfg["minimum_free_disk_gb"]
        )
        entry = requirements.setdefault(device, {"path": path, "labels": [], "required_gb": 0.0})
        entry["labels"].append(label)
        entry["required_gb"] = max(entry["required_gb"], required)
    if strict:
        for entry in requirements.values():
            free_gb = shutil.disk_usage(entry["path"]).free / 1024**3
            if free_gb < entry["required_gb"]:
                labels = "/".join(entry["labels"])
                errors.append(
                    f"{labels} 磁碟僅剩 {free_gb:.1f} GiB，"
                    f"profile 要求 {entry['required_gb']:g} GiB"
                )
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
    benchmark_ids: list[str] | None = None,
    limit: int | None = None,
) -> int:
    selected = [
        benchmark
        for benchmark in cfg["benchmarks"]
        if benchmark_ids is None or benchmark["id"] in benchmark_ids
    ]
    if not selected:
        raise ValueError("至少必須選擇一個 benchmark")
    if limit is not None and limit < 1:
        raise ValueError("--limit 必須大於 0")
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
    status_path = target / "combination_status.json"
    try:
        combo_status = json.loads(status_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        combo_status = []

    def update_status(row: dict[str, Any]) -> None:
        key = (row.get("model"), row.get("benchmark"))
        combo_status[:] = [
            old for old in combo_status if (old.get("model"), old.get("benchmark")) != key
        ]
        combo_status.append(row)
        write_json_atomic(status_path, combo_status)

    sample_cache: dict[str, list[Any]] = {}
    for benchmark in selected:
        samples = list(_samples(cfg, benchmark))
        sample_cache[benchmark["id"]] = samples if limit is None else samples[:limit]
    for model in cfg["models"]:
        for benchmark in selected:
            update_status(
                {
                    "model": model["id"],
                    "benchmark": benchmark["id"],
                    "status": "pending",
                    "completed": 0,
                    "total": len(sample_cache[benchmark["id"]]),
                    "requested_limit": limit,
                    "reason": "",
                }
            )
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not smoke:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for dataset in manifest.get("datasets", []):
            benchmark_id = next(
                b["id"] for b in cfg["benchmarks"] if b["dataset"] == dataset["name"]
            )
            if benchmark_id not in sample_cache:
                continue
            samples = sample_cache[benchmark_id]
            order = "\n".join(sample.sample_id for sample in samples).encode()
            dataset.setdefault("executions", []).append(
                {
                    "sample_count": len(samples),
                    "requested_limit": limit,
                    "sample_order_sha256": hashlib.sha256(order).hexdigest(),
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        write_json_atomic(manifest_path, manifest)
    for model in cfg["models"]:  # model-major；所選 benchmark 完成後 unload
        metadata = metadata_map[model["id"]]
        try:
            actual_digest = client.model_digest(model["tag"])
        except Exception as exc:
            actual_digest = f"ERROR:{exc}"
        if actual_digest != metadata["digest"]:
            for benchmark in selected:
                update_status(
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
            continue
        for benchmark in selected:
            try:
                prompt = (SUITE / benchmark["prompt"]).read_text(encoding="utf-8").strip()
                status = runner.run_combination(
                    model, metadata, benchmark, sample_cache[benchmark["id"]], prompt, smoke=smoke
                )
            except Exception as exc:  # 組合失敗不可阻止其他 14 組
                status = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            update_status({"model": model["id"], "benchmark": benchmark["id"], **status})
        try:
            client.unload(model["tag"])
        except Exception as exc:
            update_status(
                {
                    "model": model["id"],
                    "benchmark": "_unload",
                    "status": "warning",
                    "reason": str(exc),
                }
            )
    selected_ids = {benchmark["id"] for benchmark in selected}
    failures = [
        c for c in combo_status if c.get("benchmark") in selected_ids and c["status"] == "failed"
    ]
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
    check.add_argument(
        "--benchmark",
        action="append",
        choices=["omnidocbench", "tc_str", "vistw_mcq"],
    )
    start = sub.add_parser("init-run")
    start.add_argument("--run-id")
    start.add_argument("--resume", action="store_true")
    start.add_argument("--models-metadata", required=True)
    run = sub.add_parser("execute")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--models-metadata", required=True)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument(
        "--benchmark",
        action="append",
        choices=["omnidocbench", "tc_str", "vistw_mcq"],
    )
    run.add_argument("--limit", type=int)
    score = sub.add_parser("score")
    score.add_argument("--run-dir", required=True)
    score.add_argument(
        "--benchmark",
        action="append",
        choices=["omnidocbench", "tc_str", "vistw_mcq"],
    )
    report = sub.add_parser("report")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--models-metadata", required=True)
    report.add_argument(
        "--benchmark",
        action="append",
        choices=["omnidocbench", "tc_str", "vistw_mcq"],
    )
    report.add_argument("--limit", type=int)
    failed = sub.add_parser("mark-failed")
    failed.add_argument("--run-dir", required=True)
    failed.add_argument("--reason", required=True)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--split", default="train", choices=["train"])
    calibrate.add_argument("--model", default="qwen3_vl_4b")
    calibrate.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg["_effective_hash"] = config_hash(cfg)
    if args.command == "preflight":
        errors = preflight(
            cfg,
            strict=not args.non_strict,
            benchmark_ids=args.benchmark,
        )
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
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "running"
            manifest["ended_at_utc"] = None
            manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
            manifest.pop("failure_reason", None)
            write_json_atomic(manifest_path, manifest)
        print(run_dir)
        return 0
    if args.command == "execute":
        _, metadata = _load_model_metadata(Path(args.models_metadata))
        return execute(
            cfg,
            Path(args.run_dir),
            metadata,
            args.smoke,
            args.retry_failed,
            args.benchmark,
            args.limit,
        )
    if args.command == "score":
        if args.benchmark is None or "omnidocbench" in args.benchmark:
            score_omnidoc(cfg, Path(args.run_dir))
        return 0
    if args.command == "report":
        _, metadata = _load_model_metadata(Path(args.models_metadata))
        summary = build_summary(cfg, Path(args.run_dir), metadata, sample_limit=args.limit)
        if args.benchmark:
            selected = set(args.benchmark)
            summary = [row for row in summary if row["benchmark"] in selected]
            label = "_".join(args.benchmark)
            size = "full" if args.limit is None else f"limit_{args.limit}"
            output_dir = Path(args.run_dir) / "partial" / label / size
            output_dir.mkdir(parents=True, exist_ok=True)
            write_pairwise(
                cfg,
                Path(args.run_dir),
                benchmark_ids=selected,
                output_path=output_dir / "pairwise_comparisons.json",
                sample_limit=args.limit,
            )
            write_reports(summary, output_dir, source_run_dir=Path(args.run_dir))
            manifest_path = Path(args.run_dir) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "partial"
            manifest.setdefault("partial_results", []).append(
                {
                    "benchmarks": args.benchmark,
                    "limit": args.limit,
                    "path": str(output_dir),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_json_atomic(manifest_path, manifest)
            print(output_dir)
            return 0
        write_pairwise(cfg, Path(args.run_dir))
        write_reports(summary, Path(args.run_dir))
        manifest_path = Path(args.run_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "completed"
        manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(manifest_path, manifest)
        return 0
    if args.command == "mark-failed":
        manifest_path = Path(args.run_dir) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "running":
                manifest["status"] = "failed"
                manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
                manifest["failure_reason"] = args.reason
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
