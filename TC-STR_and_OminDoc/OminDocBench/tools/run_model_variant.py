#!/usr/bin/env python3
"""Run preflight / smoke / inference-only for an ALTERNATE, independent set
of models, without ever touching config/models.yaml or any file under
omnidocbench/ or config/.

Why this exists: omnidocbench's own Settings.load() always reads
config/models.yaml, and its code_hash (part of run_signature -> run_id)
hashes every file under config/ plus every .py file under omnidocbench/. Any
edit to either -- including just adding a sibling models file into config/ --
shifts run_id for every run sharing this codebase, including
already-completed ones. This script builds a Settings object directly (same
frozen dataclass, same preflight/smoke/inference functions, unmodified) from
an arbitrary models file living OUTSIDE config/, so an independent model
subset gets its own run_id without disturbing any other run's identity.

Usage:
    python3 tools/run_model_variant.py preflight --models-file config_variants/models_e4b_12b_31b.yaml
    python3 tools/run_model_variant.py smoke      --models-file config_variants/models_e4b_12b_31b.yaml
    python3 tools/run_model_variant.py full-inference --models-file config_variants/models_e4b_12b_31b.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from omnidocbench.cli import (  # noqa: E402
    blocked_smoke,
    select_warmup_page,
)
from omnidocbench.core import (  # noqa: E402
    CONFIG_DIR,
    CheckpointDB,
    Settings,
    atomic_write,
    build_report,
    create_smoke_gt,
    export_raw_jsonl,
    install_signal_handlers,
    official_config,
    preflight,
    read_json,
    run_model_pages,
    run_official_evaluator,
    sha256_file,
    unload_model,
    utc_now,
    validate_smoke_selection,
    warmup_and_gpu_check,
    write_json,
)


def resolve_models_path(models_file: str) -> pathlib.Path:
    models_path = pathlib.Path(models_file)
    if not models_path.is_absolute():
        models_path = ROOT / models_path
    return models_path


def load_variant_settings(models_file: str) -> Settings:
    models_path = resolve_models_path(models_file)
    return Settings(
        raw=read_json(CONFIG_DIR / "benchmark.json"),
        models=read_json(models_path)["models"],
        smoke=read_json(CONFIG_DIR / "smoke_pages.json"),
    )


def unlink_stale_immutable_models_snapshot(settings: Settings) -> None:
    """core.write_immutable_inputs(), called from inside preflight(), always
    snapshots CONFIG_DIR/models.yaml (the live, main-run models file) into
    this run's immutable_config/models.yaml -- it has no idea this run was
    built from a different --models-file. If a prior invocation already
    restored the correct variant snapshot (see restore_immutable_models_snapshot
    below), write_immutable_inputs would see that existing, correct content
    differ from the live config/models.yaml bytes and raise "immutable config
    changed in existing run". Deleting the file first (only ever done
    immediately before a preflight() call, always restored right after) makes
    write_immutable_inputs see a first-time write and skip that guard, the
    same way it already does the very first time a given run_id is
    preflighted.
    """
    dest = settings.output_dir / "immutable_config" / "models.yaml"
    dest.unlink(missing_ok=True)


def restore_immutable_models_snapshot(settings: Settings, models_path: pathlib.Path) -> None:
    """Overwrite this run's immutable_config/models.yaml (just clobbered by
    write_immutable_inputs with the wrong, live config/models.yaml) with the
    actual --models-file content, and fix up its recorded hash in hashes.json
    to match -- so tools/evaluate_model.py, which trusts and hash-verifies
    that snapshot, sees the models this run actually used."""
    immutable_dir = settings.output_dir / "immutable_config"
    dest = immutable_dir / "models.yaml"
    content = models_path.read_bytes()
    atomic_write(dest, content)
    hashes_path = immutable_dir / "hashes.json"
    hashes = read_json(hashes_path)
    hashes["models.yaml"] = hashlib.sha256(content).hexdigest()
    write_json(hashes_path, hashes)


def verify_ready_variant(settings: Settings) -> dict:
    """Same checks as omnidocbench.cli.verify_ready, reproduced here (not
    imported and not patched in cli.py) because that function hardcodes an
    expected review page count of 40 -- correct only for exactly 2 models x
    20 smoke pages. This computes that count as
    len(settings.models) * smoke_page_count instead, which is the same check
    generalized to however many models this variant covers, not a weaker one.
    """
    path = settings.output_dir / "smoke_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"READY smoke manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("state") != "READY_FOR_FULL_RUN":
        raise RuntimeError(f"full-run gate closed: smoke state is {manifest.get('state')!r}")
    checks = {
        "run_signature": settings.run_signature,
        "prompt_hash": settings.prompt_hash,
        "options_hash": settings.options_hash,
        "dataset_gt_sha256": settings.raw["benchmark"]["ground_truth_sha256"],
        "evaluator_digest": settings.raw["evaluator"]["docker_digest"],
        "code_hash": settings.code_hash,
    }
    for key, value in checks.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"READY manifest {key} mismatch")
    review_path = settings.output_dir / "ai_review.json"
    if not review_path.is_file():
        raise RuntimeError("AI page review artifact missing")
    review_data = json.loads(review_path.read_text(encoding="utf-8"))
    expected_pages = len(settings.models) * settings.raw["smoke"]["page_count"]
    if review_data.get("state") != "READY" or review_data.get("reviewed_pages") != expected_pages:
        raise RuntimeError(
            f"AI page review did not approve all {expected_pages} smoke outputs "
            f"({len(settings.models)} models x {settings.raw['smoke']['page_count']} pages)"
        )
    return manifest


def cmd_preflight(settings: Settings, _args: argparse.Namespace) -> int:
    report = preflight(settings, hash_images=not _args.fast)
    print(json.dumps({"state": report["state"], "blockers": report["blockers"], "output": str(settings.output_dir)}, ensure_ascii=False, indent=2))
    return 0 if report["state"] == "PASS" else 2


def cmd_smoke(settings: Settings, _args: argparse.Namespace) -> int:
    report = preflight(settings, hash_images=True)
    if report["state"] != "PASS":
        return blocked_smoke(settings, report)
    manifest = report["dataset"]["manifest"]
    selected = validate_smoke_selection(settings, manifest)
    warmup_page = select_warmup_page(manifest)

    db = CheckpointDB(settings.output_dir / "results.sqlite")
    db.ensure_run(settings, "SMOKE_RUNNING")
    stop_requested = {"value": False}
    install_signal_handlers(stop_requested, db, settings)
    warmups, all_errors, run_stats = [], [], []
    for model in settings.models:
        warmup, errors = warmup_and_gpu_check(settings, model, warmup_page)
        warmups.append(warmup)
        all_errors.extend(errors)
        if errors:
            break
        model_stats = run_model_pages(settings, db, model, selected, stop_requested)
        run_stats.append(model_stats)
        unload_model(settings, model["ollama_tag"])
        if model_stats["failed"]:
            all_errors.append(f"{model['id']} had {model_stats['failed']} failed diagnostic/smoke pages")
        if stop_requested["value"]:
            all_errors.append("smoke interrupted")
            break
    write_json(settings.output_dir / "warmup_gpu_checks.json", warmups)
    export_raw_jsonl(settings, db)
    db.close()

    if all_errors:
        report["blockers"].extend(all_errors)
        report["state"] = "BLOCKED"
        write_json(settings.output_dir / "preflight_report.json", report)
        return blocked_smoke(settings, report)

    evaluations = {}
    gt = create_smoke_gt(settings, selected)
    for model in settings.models:
        evaluations[model["id"]] = run_official_evaluator(settings, model["id"], gt)
        if evaluations[model["id"]]["returncode"] != 0:
            all_errors.append(f"official smoke evaluation failed for {model['id']}")
    write_json(settings.output_dir / "smoke_official_evaluation_runs.json", evaluations)

    state = "BLOCKED" if all_errors else "SMOKE_REQUIRES_AI_REVIEW"
    smoke_manifest = {
        "state": state,
        "run_id": settings.run_id,
        "run_signature": settings.run_signature,
        "created_at": utc_now(),
        "prompt_hash": settings.prompt_hash,
        "options_hash": settings.options_hash,
        "code_hash": settings.code_hash,
        "dataset_gt_sha256": settings.raw["benchmark"]["ground_truth_sha256"],
        "evaluator_digest": settings.raw["evaluator"]["docker_digest"],
        "model_digests": {item["mapping"]["id"]: item.get("digest") for item in report["models"]["models"]},
        "models": settings.models,
        "pages": selected,
        "warmups": warmups,
        "run_stats": run_stats,
        "blockers": all_errors,
        "full_inference_started": False,
        "diagnostic_only": False,
        "diagnostic_page_count": None,
    }
    write_json(settings.output_dir / "smoke_manifest.json", smoke_manifest)
    build_report(settings, smoke=True)
    write_json(settings.output_dir / "status.json", {"state": state, "updated_at": utc_now(), "blockers": all_errors})
    print(f"run_id={settings.run_id}")
    print(f"state={state}")
    return 0 if not all_errors else 2


def ensure_official_configs(settings: Settings) -> None:
    """tools/evaluate_model.py requires immutable_config/{model_id}_official.yaml
    per model, hash-checked against hashes.json's official_run.yaml entry.
    Normally cmd_smoke's call to run_official_evaluator() writes this as a
    side effect during smoke. cmd_full_inference never calls
    run_official_evaluator() (full-inference does inference only, evaluation
    is a separate later step) -- so a run that goes straight from preflight to
    full-inference via --skip-review, never having gone through smoke, would
    reach evaluate_model.py later missing this file entirely. official_config()
    takes no model_id and is identical for every model, so this just writes
    that same already-hashed content under each model's expected filename;
    it does not run the Docker evaluator itself."""
    target = settings.output_dir / "immutable_config"
    expected = read_json(target / "hashes.json").get("official_run.yaml")
    content = official_config(settings)
    for model in settings.models:
        path = target / f"{model['id']}_official.yaml"
        if path.is_file() and sha256_file(path) == expected:
            continue
        atomic_write(path, content)


def cmd_full_inference(settings: Settings, args: argparse.Namespace) -> int:
    print(f"run_id={settings.run_id}")
    print(f"output_dir={settings.output_dir}")
    if args.skip_review:
        # Explicit operator opt-out: this ALWAYS defaults off. Skips both the
        # smoke/AI-review requirement and the resulting model-digest
        # cross-check (there is nothing to cross-check against without a
        # completed smoke run). Only affects config_variants runs invoked via
        # this flag -- never the default path, never the main config/models.yaml.
        print("WARNING: --skip-review set; bypassing the smoke/AI-review gate.", file=sys.stderr)
        ready = None
    else:
        ready = verify_ready_variant(settings)
    report = preflight(settings, hash_images=True)
    if report["state"] != "PASS":
        print("PREFLIGHT FAILED:", report["blockers"], file=sys.stderr)
        return 2
    ensure_official_configs(settings)
    if ready is not None:
        current_digests = {item["mapping"]["id"]: item.get("digest") for item in report["models"]["models"]}
        if current_digests != ready.get("model_digests"):
            print(f"model digests changed since READY smoke: {current_digests} != {ready.get('model_digests')}", file=sys.stderr)
            return 2

    pages = report["dataset"]["manifest"]
    warmup_page = select_warmup_page(pages)
    db = CheckpointDB(settings.output_dir / "results.sqlite")
    db.ensure_run(settings, "FULL_RUNNING")
    stop_requested = {"value": False}
    install_signal_handlers(stop_requested, db, settings)

    for model in settings.models:
        print(f"=== warmup: {model['id']} ===", flush=True)
        warmup, errors = warmup_and_gpu_check(settings, model, warmup_page)
        if errors:
            db.event(settings, "ERROR", "full warm-up gate failed", {"errors": errors})
            db.close()
            print("WARMUP FAILED:", errors, file=sys.stderr)
            return 2
        print(f"=== inference: {model['id']} ({len(pages)} pages) ===", flush=True)
        run_model_pages(settings, db, model, pages, stop_requested)
        unload_model(settings, model["ollama_tag"])
        if stop_requested["value"]:
            export_raw_jsonl(settings, db)
            db.close()
            print("interrupted", file=sys.stderr)
            return 130
        print(f"=== {model['id']} inference complete ===", flush=True)

    export_raw_jsonl(settings, db)
    db.close()
    write_json(settings.output_dir / "status.json", {"state": "INFERENCE_COMPLETE_EVALUATION_PENDING", "updated_at": utc_now()})
    print("\nAll models' inference complete. Evaluation intentionally NOT run here.")
    print("Next: use tools/evaluate_model.py + tools/auto_resume_evaluation.py per model.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run preflight/smoke/full-inference for an alternate models file")
    parser.add_argument("command", choices=["preflight", "smoke", "full-inference", "status"])
    parser.add_argument("--models-file", required=True, help="path to an alternate models.yaml-shaped JSON file, kept OUTSIDE config/")
    parser.add_argument("--fast", action="store_true", help="preflight only: skip per-image SHA-256")
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="full-inference only: bypass the smoke/AI-review gate (default is still the gate; this is an explicit operator opt-out)",
    )
    args = parser.parse_args()

    settings = load_variant_settings(args.models_file)
    if args.command == "status":
        from inference_status import print_progress

        print_progress(settings)
        return 0

    models_path = resolve_models_path(args.models_file)
    unlink_stale_immutable_models_snapshot(settings)
    try:
        if args.command == "preflight":
            return cmd_preflight(settings, args)
        if args.command == "smoke":
            return cmd_smoke(settings, args)
        return cmd_full_inference(settings, args)
    finally:
        if (settings.output_dir / "immutable_config").is_dir():
            restore_immutable_models_snapshot(settings, models_path)


if __name__ == "__main__":
    raise SystemExit(main())
