from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .core import (
    CheckpointDB,
    Settings,
    build_report,
    create_smoke_gt,
    export_raw_jsonl,
    install_signal_handlers,
    model_output_dirs,
    preflight,
    run_model_pages,
    run_official_evaluator,
    unload_model,
    utc_now,
    validate_smoke_selection,
    warmup_and_gpu_check,
    write_json,
    validate_dataset,
)


def command_preflight(settings: Settings, args: argparse.Namespace) -> int:
    report = preflight(settings, hash_images=not args.fast)
    print(json.dumps({"state": report["state"], "blockers": report["blockers"], "output": str(settings.output_dir)}, ensure_ascii=False, indent=2))
    return 0 if report["state"] == "PASS" else 2


def blocked_smoke(settings: Settings, report: dict) -> int:
    db = CheckpointDB(settings.output_dir / "results.sqlite")
    db.ensure_run(settings, "BLOCKED")
    db.event(settings, "ERROR", "smoke gate blocked before inference", {"blockers": report["blockers"]})
    db.close()
    smoke_manifest = {
        "state": "BLOCKED",
        "run_id": settings.run_id,
        "run_signature": settings.run_signature,
        "code_hash": settings.code_hash,
        "created_at": utc_now(),
        "blockers": report["blockers"],
        "full_inference_started": False,
        "smoke_inference_started": False,
    }
    write_json(settings.output_dir / "smoke_manifest.json", smoke_manifest)
    build_report(settings, smoke=True)
    write_json(
        settings.output_dir / "status.json",
        {"state": "BLOCKED", "updated_at": utc_now(), "blockers": report["blockers"]},
    )
    return 2


def select_warmup_page(manifest: list[dict]) -> dict:
    quick_path = pathlib.Path(__file__).parents[1] / "config" / "quick_load_pages.json"
    first = json.loads(quick_path.read_text(encoding="utf-8"))["pages"][0]
    for page in manifest:
        if page["filename"] == first["filename"] and page["index"] == first["index"]:
            return page
    raise ValueError(f"fixed warm-up page missing from v1.6 manifest: {first}")


def command_smoke(settings: Settings, args: argparse.Namespace) -> int:
    quick_specs = None
    if args.quick_load:
        quick_path = pathlib.Path(__file__).parents[1] / "config" / "quick_load_pages.json"
        quick_config = json.loads(quick_path.read_text(encoding="utf-8"))
        quick_specs = quick_config["pages"]
        # This is an explicitly non-scoring model-loading diagnostic. The
        # effective option is hashed into its own run signature and metadata.
        settings.raw["inference"]["options"]["num_predict"] = quick_config[
            "diagnostic_num_predict"
        ]
    report = preflight(settings, hash_images=True)
    if report["state"] != "PASS":
        return blocked_smoke(settings, report)
    manifest = report["dataset"]["manifest"]
    selected = validate_smoke_selection(settings, manifest)
    warmup_page = select_warmup_page(manifest)
    diagnostic = args.limit is not None or args.quick_load
    if args.quick_load:
        by_name = {page["filename"]: page for page in manifest}
        selected = []
        for wanted in quick_specs:
            found = by_name.get(wanted["filename"])
            if found is None or found["index"] != wanted["index"]:
                raise ValueError(f"fixed quick-load page mismatch: {wanted}")
            selected.append(found)
    if args.limit is not None:
        if not 1 <= args.limit < settings.raw["smoke"]["page_count"]:
            raise ValueError(
                f"--limit must be between 1 and "
                f"{settings.raw['smoke']['page_count'] - 1}"
            )
        selected = selected[:args.limit]
    db = CheckpointDB(settings.output_dir / "results.sqlite")
    db.ensure_run(settings, "SMOKE_RUNNING")
    stop_requested = {"value": False}
    install_signal_handlers(stop_requested, db, settings)
    warmups = []
    all_errors = []
    run_stats = []
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
            all_errors.append(
                f"{model['id']} had {model_stats['failed']} failed diagnostic/smoke pages"
            )
        if stop_requested["value"]:
            all_errors.append("smoke interrupted")
            break
    write_json(settings.output_dir / "warmup_gpu_checks.json", warmups)
    export_raw_jsonl(settings, db)
    db.close()
    if all_errors and not diagnostic:
        report["blockers"].extend(all_errors)
        report["state"] = "BLOCKED"
        write_json(settings.output_dir / "preflight_report.json", report)
        return blocked_smoke(settings, report)
    evaluations = {}
    if not diagnostic:
        gt = create_smoke_gt(settings, selected)
        for model in settings.models:
            evaluations[model["id"]] = run_official_evaluator(settings, model["id"], gt)
            if evaluations[model["id"]]["returncode"] != 0:
                all_errors.append(f"official smoke evaluation failed for {model['id']}")
    write_json(settings.output_dir / "smoke_official_evaluation_runs.json", evaluations)
    # A limited diagnostic can never satisfy or open the full-run gate.
    state = (
        "DIAGNOSTIC_BLOCKED"
        if diagnostic and all_errors
        else "BLOCKED"
        if all_errors
        else "DIAGNOSTIC_COMPLETE"
        if diagnostic
        else "SMOKE_REQUIRES_AI_REVIEW"
    )
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
        "model_digests": {
            item["mapping"]["id"]: item.get("digest")
            for item in report["models"]["models"]
        },
        "models": settings.models,
        "pages": selected,
        "warmups": warmups,
        "run_stats": run_stats,
        "blockers": all_errors,
        "full_inference_started": False,
        "diagnostic_only": diagnostic,
        "diagnostic_page_count": len(selected) if diagnostic else None,
    }
    write_json(settings.output_dir / "smoke_manifest.json", smoke_manifest)
    build_report(settings, smoke=True)
    write_json(settings.output_dir / "status.json", {"state": state, "updated_at": utc_now(), "blockers": all_errors})
    return 0 if not all_errors else 2


def verify_ready(settings: Settings) -> dict:
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
    review = settings.output_dir / "ai_review.json"
    if not review.is_file():
        raise RuntimeError("AI page review artifact missing")
    review_data = json.loads(review.read_text(encoding="utf-8"))
    if review_data.get("state") != "READY" or review_data.get("reviewed_pages") != 40:
        raise RuntimeError("AI page review did not approve all 40 smoke outputs")
    return manifest


def command_full(settings: Settings, _args: argparse.Namespace) -> int:
    ready = verify_ready(settings)
    report = preflight(settings, hash_images=True)
    if report["state"] != "PASS":
        return 2
    current_digests = {
        item["mapping"]["id"]: item.get("digest")
        for item in report["models"]["models"]
    }
    if current_digests != ready.get("model_digests"):
        raise RuntimeError(
            f"model digests changed since READY smoke: {current_digests} != "
            f"{ready.get('model_digests')}"
        )
    pages = report["dataset"]["manifest"]
    warmup_page = select_warmup_page(pages)
    db = CheckpointDB(settings.output_dir / "results.sqlite")
    db.ensure_run(settings, "FULL_RUNNING")
    stop_requested = {"value": False}
    install_signal_handlers(stop_requested, db, settings)
    for model in settings.models:
        warmup, errors = warmup_and_gpu_check(settings, model, warmup_page)
        if errors:
            db.event(settings, "ERROR", "full warm-up gate failed", {"errors": errors})
            db.close()
            return 2
        run_model_pages(settings, db, model, pages, stop_requested)
        unload_model(settings, model["ollama_tag"])
        if stop_requested["value"]:
            export_raw_jsonl(settings, db)
            db.close()
            return 130
    export_raw_jsonl(settings, db)
    db.close()
    for model in settings.models:
        result = run_official_evaluator(settings, model["id"], settings.gt_path)
        if result["returncode"] != 0:
            return 3
    build_report(settings, smoke=False)
    write_json(settings.output_dir / "status.json", {"state": "COMPLETE", "updated_at": utc_now()})
    return 0


def command_evaluate(settings: Settings, _args: argparse.Namespace) -> int:
    verify_ready(settings)
    dataset, errors = validate_dataset(settings, hash_images=False)
    if errors:
        print(json.dumps({"state": "BLOCKED", "errors": errors}, ensure_ascii=False, indent=2))
        return 2
    failures = 0
    for model in settings.models:
        pred = model_output_dirs(settings, model["id"])["predictions"]
        if not pred.is_dir():
            failures += 1
            continue
        result = run_official_evaluator(settings, model["id"], settings.gt_path)
        failures += int(result["returncode"] != 0)
    build_report(settings, smoke=False)
    return 3 if failures else 0


def command_report(settings: Settings, args: argparse.Namespace) -> int:
    path = build_report(settings, smoke=args.smoke)
    print(path)
    return 0


def command_status(settings: Settings, _args: argparse.Namespace) -> int:
    path = settings.output_dir / "status.json"
    if path.exists():
        print(path.read_text(encoding="utf-8"), end="")
    else:
        print(json.dumps({"state": "NOT_STARTED", "run_id": settings.run_id}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproducible OmniDocBench v1.6 runner")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--fast", action="store_true", help="skip per-image SHA-256")
    smoke = sub.add_parser("smoke")
    smoke.add_argument(
        "--limit",
        type=int,
        help="run only the first N fixed smoke pages as a diagnostic; never opens full-run gate",
    )
    smoke.add_argument(
        "--quick-load",
        action="store_true",
        help="run the fixed 10 short-page model-loading diagnostic; never opens full-run gate",
    )
    sub.add_parser("full")
    sub.add_parser("evaluate")
    report = sub.add_parser("report")
    report.add_argument("--smoke", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    settings = Settings.load()
    handlers = {
        "preflight": command_preflight,
        "smoke": command_smoke,
        "full": command_full,
        "evaluate": command_evaluate,
        "report": command_report,
        "status": command_status,
    }
    try:
        return handlers[args.command](settings, args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
