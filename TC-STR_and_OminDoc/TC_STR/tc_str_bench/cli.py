from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import load_settings
from .dataset import build_manifest
from .preflight import execute_preflight
from .reporting import write_smoke_report
from .runner import StopRequested, assess_smoke, build_reports, run_phase
from .storage import Checkpoint
from .util import append_log, atomic_write_json, utc_now


def _new_run_id() -> str:
    return "tcstr_" + utc_now().replace("-", "").replace(":", "").replace(".", "").replace("Z", "Z")


def _prepare(settings: Any, run_id: str | None = None) -> tuple[str, Path, dict[str, Any]]:
    run_id = run_id or os.environ.get("TC_STR_RUN_ID") or _new_run_id()
    if "/" in run_id or run_id in {".", ".."}:
        raise ValueError("run_id 不可包含 path separator")
    run_dir = settings.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(settings.raw["paths"]["scratch"]).mkdir(parents=True, exist_ok=True)
    Path(settings.raw["paths"]["volatile_outputs"]).mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(settings.dataset_dir, run_dir / "dataset_manifest.json", 3706)
    atomic_write_json(
        run_dir / "run_config.json",
        {"config": settings.raw, "models": settings.models, "source_fingerprint": settings.source_hash},
    )
    return run_id, run_dir, manifest


def _metadata(
    settings: Any,
    run_id: str,
    run_dir: Path,
    manifest: dict[str, Any],
    preflight: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    environment = preflight["environment"]["commands"]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": preflight["status"],
        "started_at": started_at,
        "ended_at": None,
        "run_signature": preflight["run_signature"],
        "models": [
            {
                key: model.get(key)
                for key in (
                    "id", "logical_name", "exact_tag", "digest", "quantization",
                    "parameter_count", "architecture", "context_limit", "capabilities",
                    "processor", "loaded_size", "loaded_context", "completion_policy",
                )
            }
            for model in preflight["models"]
        ],
        "ollama_version": environment["ollama_version"]["stdout"].strip(),
        "os": environment["python_version"]["stdout"].strip() + " / " + __import__("platform").platform(),
        "gpu": environment["nvidia_smi"]["stdout"],
        "generation_options": settings.options,
        "thinking": settings.options.get("think"),
        "kv_cache_type": "fp16 (recorded; service configuration unchanged)",
        "endpoint": settings.raw["ollama"]["host"] + settings.raw["ollama"]["endpoint"],
        "prompt": settings.prompt,
        "prompt_sha256": settings.prompt_hash,
        "dataset": {
            "name": settings.raw["benchmark"]["name"],
            "split": settings.raw["benchmark"]["split"],
            "sample_count": manifest["sample_count"],
            "source_url": settings.raw["benchmark"]["source_url"],
            "dataset_fingerprint": manifest["dataset_fingerprint"],
        },
        "versions": {
            **settings.raw["versions"],
            "program_git_sha": environment["git_head"]["stdout"].strip(),
            "program_git_dirty": bool(environment["git_status"]["stdout"].strip()),
            "source_fingerprint": settings.source_hash,
        },
        "execution_policy": {
            "timeout_seconds": settings.raw["ollama"]["timeout_seconds"],
            "max_attempts": settings.raw["ollama"]["max_attempts"],
            "retry_backoff_seconds": settings.raw["ollama"]["retry_backoff_seconds"],
            "warm_up": "one fixed smoke image per model; excluded from scores and latency",
            "batch_size": 1,
            "input_preprocessing": "none; original image bytes",
            "model_completion_policies": {
                model["id"]: model.get("completion_policy") or {"mode": "strict"}
                for model in settings.models
            },
            "evaluation_policy": settings.raw["evaluation_policy"],
        },
        "output_path": str(run_dir),
    }


def command_smoke(args: argparse.Namespace) -> int:
    settings = load_settings()
    run_id, run_dir, manifest = _prepare(settings, args.run_id)
    metadata_stub = {
        "run_id": run_id,
        "status": "PREFLIGHT_RUNNING",
        "started_at": utc_now(),
        "ended_at": None,
        "output_path": str(run_dir),
    }
    atomic_write_json(run_dir / "metadata.json", metadata_stub)
    append_log(run_dir / "run.log", "suite_start phase=preflight")
    preflight = execute_preflight(settings, run_id, run_dir, manifest)
    metadata = _metadata(settings, run_id, run_dir, manifest, preflight, metadata_stub["started_at"])
    if preflight["status"] != "READY":
        metadata.update(status="BLOCKED", ended_at=utc_now(), run_signature=preflight["run_signature"])
        atomic_write_json(run_dir / "metadata.json", metadata)
        atomic_write_json(
            run_dir / "status.json",
            {
                "run_id": run_id,
                "status": "BLOCKED",
                "phase": "preflight",
                "updated_at": utc_now(),
                "output_path": str(run_dir),
            },
        )
        append_log(run_dir / "error.log", "preflight BLOCKED; smoke not executed")
        append_log(run_dir / "run.log", "suite_stop status=BLOCKED phase=preflight")
        write_smoke_report(
            settings,
            run_dir,
            preflight,
            manifest,
            "BLOCKED",
            ["smoke 未執行：8 個模型尚未全部通過 exact tag / vision / 100% GPU preflight", *preflight["blockers"]],
        )
        build_reports(settings, run_dir)
        print(f"BLOCKED {run_dir}")
        return 2
    if args.preflight_only:
        metadata.update(status="PREFLIGHT_READY", ended_at=utc_now(), run_signature=preflight["run_signature"])
        atomic_write_json(run_dir / "metadata.json", metadata)
        print(f"PREFLIGHT_READY {run_dir}")
        return 0
    metadata.update(status="SMOKE_RUNNING", run_signature=preflight["run_signature"])
    atomic_write_json(run_dir / "metadata.json", metadata)
    try:
        runtime_blockers = run_phase(settings, run_dir, preflight, manifest, "smoke")
    except StopRequested:
        metadata.update(status="INTERRUPTED", ended_at=utc_now())
        atomic_write_json(run_dir / "metadata.json", metadata)
        return 130
    report = assess_smoke(settings, run_dir, preflight, manifest, runtime_blockers)
    metadata.update(status=report["status"], ended_at=utc_now())
    atomic_write_json(run_dir / "metadata.json", metadata)
    atomic_write_json(
        run_dir / "status.json",
        {
            "run_id": run_id,
            "status": report["status"],
            "phase": "smoke",
            "updated_at": utc_now(),
            "counts": {
                "terminal_results": sum(
                    int(model["scored_samples"]) for model in report["models"]
                ),
                "hard_blockers": len(report["blockers"]),
            },
            "output_path": str(run_dir),
        },
    )
    build_reports(settings, run_dir)
    print(f"{report['status']} {run_dir}")
    return 0 if report["status"] == "READY_FOR_FULL_RUN" else 3


def _ready_run(settings: Any, requested: str | None) -> Path:
    candidates = [settings.output_root / requested] if requested else sorted(settings.output_root.glob("tcstr_*"), reverse=True)
    for candidate in candidates:
        report_path = candidate / "smoke_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "READY_FOR_FULL_RUN":
            return candidate
    raise RuntimeError("找不到同 signature 且狀態 READY_FOR_FULL_RUN 的 smoke run；拒絕啟動 full")


def command_full(args: argparse.Namespace) -> int:
    settings = load_settings()
    run_dir = _ready_run(settings, args.run_id)
    manifest = build_manifest(settings.dataset_dir, run_dir / "dataset_manifest.current.json", 3706)
    old_manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    smoke = json.loads((run_dir / "smoke_report.json").read_text(encoding="utf-8"))
    if manifest["dataset_fingerprint"] != old_manifest["dataset_fingerprint"]:
        raise RuntimeError("dataset fingerprint 已變更；拒絕污染原 run")
    preflight = execute_preflight(settings, run_dir.name, run_dir, manifest)
    if preflight["status"] != "READY" or preflight["run_signature"] != smoke["run_signature"]:
        raise RuntimeError("full 前 preflight/signature 與 READY smoke 不一致；拒絕開始")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata.update(status="FULL_RUNNING", full_started_at=utc_now(), ended_at=None)
    atomic_write_json(run_dir / "metadata.json", metadata)
    try:
        blockers = run_phase(settings, run_dir, preflight, manifest, "full")
    except StopRequested:
        metadata.update(status="INTERRUPTED", ended_at=utc_now())
        atomic_write_json(run_dir / "metadata.json", metadata)
        return 130
    checkpoint = Checkpoint(run_dir / "results.sqlite")
    try:
        count = len(checkpoint.results(preflight["run_signature"], "full"))
    finally:
        checkpoint.close()
    expected = 3706 * 8
    status = "COMPLETED" if not blockers and count == expected else "BLOCKED"
    metadata.update(status=status, ended_at=utc_now(), full_results=count, full_expected=expected, blockers=blockers)
    atomic_write_json(run_dir / "metadata.json", metadata)
    build_reports(settings, run_dir)
    return 0 if status == "COMPLETED" else 4


def command_report(args: argparse.Namespace) -> int:
    settings = load_settings()
    run_dir = settings.output_root / args.run_id if args.run_id else max(settings.output_root.glob("tcstr_*"))
    build_reports(settings, run_dir)
    print(run_dir / "report.html")
    return 0


def command_status(args: argparse.Namespace) -> int:
    settings = load_settings()
    candidates = [settings.output_root / args.run_id] if args.run_id else sorted(settings.output_root.glob("tcstr_*"), reverse=True)
    if not candidates:
        print("尚無 TC-STR run")
        return 1
    run_dir = candidates[0]
    for name in ("status.json", "metadata.json"):
        path = run_dir / name
        if path.exists():
            print(path.read_text(encoding="utf-8"))
            break
    log = run_dir / "run.log"
    if log.exists():
        print("--- recent log ---")
        print("\n".join(log.read_text(encoding="utf-8").splitlines()[-20:]))
    print(f"EBS output: {run_dir}")
    return 0


def command_review_smoke(args: argparse.Namespace) -> int:
    """Apply a human/AI-authored per-item review; never creates a review automatically."""
    settings = load_settings()
    run_dir = settings.output_root / args.run_id
    preflight = json.loads((run_dir / "preflight_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    existing = json.loads((run_dir / "smoke_report.json").read_text(encoding="utf-8"))
    if existing["status"] not in {"AWAITING_AI_MANUAL_REVIEW", "BLOCKED"}:
        raise RuntimeError(
            "smoke 狀態必須是 AWAITING_AI_MANUAL_REVIEW 或 BLOCKED: "
            f"{existing['status']}"
        )
    review = json.loads(Path(args.review_file).read_text(encoding="utf-8"))
    if review.get("run_signature") != preflight["run_signature"]:
        raise RuntimeError("manual review signature 不符")
    expected = {
        (model["id"], int(index))
        for model in settings.models
        for index in manifest["smoke_selection"]["sample_indices"]
    }
    entries = review.get("reviews", [])
    actual = {(entry.get("model_id"), int(entry.get("sample_index", -1))) for entry in entries}
    if actual != expected or len(entries) != 160:
        raise RuntimeError("manual review 必須逐筆覆蓋 8×20 且不得重複")
    allowed = {"NORMAL", "MODEL_ABILITY_ERROR", "PIPELINE_ANOMALY"}
    if any(entry.get("verdict") not in allowed for entry in entries):
        raise RuntimeError(f"manual review verdict 僅允許 {sorted(allowed)}")
    pipeline = [entry for entry in entries if entry["verdict"] == "PIPELINE_ANOMALY"]
    blockers = list(existing.get("blockers", []))
    warnings = list(existing.get("warnings", []))
    if pipeline:
        note = (
            f"逐筆人工檢視發現 {len(pipeline)} 筆輸出異常；依核准政策視為模型在"
            "共同使用條件下的實際失敗，保留 raw response 並納入計分，不阻擋整套評測"
        )
        if note not in warnings:
            warnings.append(note)
    status = "BLOCKED" if blockers else "READY_FOR_FULL_RUN"
    report = write_smoke_report(
        settings, run_dir, preflight, manifest, status, blockers, review, warnings
    )
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata.update(status=report["status"], ended_at=utc_now(), manual_reviewed_at=utc_now())
    atomic_write_json(run_dir / "metadata.json", metadata)
    atomic_write_json(
        run_dir / "status.json",
        {
            "run_id": run_dir.name,
            "status": report["status"],
            "phase": "smoke_manual_review",
            "updated_at": utc_now(),
            "counts": {
                "terminal_results": len(report["results"]),
                "manual_reviews": len(entries),
                "pipeline_anomalies_reviewed": len(pipeline),
                "hard_blockers": len(blockers),
            },
            "output_path": str(run_dir),
        },
    )
    build_reports(settings, run_dir)
    print(f"{status} {run_dir}")
    return 0 if status == "READY_FOR_FULL_RUN" else 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--run-id")
    smoke.add_argument("--preflight-only", action="store_true")
    smoke.set_defaults(func=command_smoke)
    full = sub.add_parser("full")
    full.add_argument("--run-id")
    full.set_defaults(func=command_full)
    report = sub.add_parser("report")
    report.add_argument("--run-id")
    report.set_defaults(func=command_report)
    status = sub.add_parser("status")
    status.add_argument("--run-id")
    status.set_defaults(func=command_status)
    review = sub.add_parser("review-smoke")
    review.add_argument("--run-id", required=True)
    review.add_argument("--review-file", required=True)
    review.set_defaults(func=command_review_smoke)
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
