#!/usr/bin/env python3
"""Run the full 1,651-page inference pass for every configured model,
identical to `omnidocbench full`, but WITHOUT its trailing call into the
built-in official evaluator.

Why this exists: `command_full` in omnidocbench/cli.py always finishes by
calling `run_official_evaluator()` for every model, which invokes the same
pinned OmniDocBench docker evaluator this project spent a full session
hardening against (see tools/evaluate_model.py) -- but through a plain
`run_command(..., timeout=7200)` with no container naming, no stall
detection, and no cleanup on timeout. Even with match_workers=1 (validated
this session to remove the specific func_timeout/ThreadPoolExecutor
deadlock), that unprotected path is not worth the residual risk on a corpus
this large. This script does the identical inference work and then stops --
evaluation is done afterwards with evaluate_model.py + auto_resume_evaluation.py,
which already have that protection.

This is not a fork of the inference logic: every function called here
(verify_ready, preflight, warmup_and_gpu_check, run_model_pages,
unload_model, export_raw_jsonl, install_signal_handlers, CheckpointDB) is
imported unchanged from omnidocbench.cli / omnidocbench.core.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from omnidocbench.cli import select_warmup_page, verify_ready
from omnidocbench.core import (
    CheckpointDB,
    Settings,
    export_raw_jsonl,
    install_signal_handlers,
    preflight,
    run_model_pages,
    unload_model,
    warmup_and_gpu_check,
    write_json,
    utc_now,
)


def main() -> int:
    settings = Settings.load()
    print(f"run_id={settings.run_id}")
    print(f"output_dir={settings.output_dir}")

    ready = verify_ready(settings)
    report = preflight(settings, hash_images=True)
    if report["state"] != "PASS":
        print("PREFLIGHT FAILED:", report["blockers"], file=sys.stderr)
        return 2
    current_digests = {
        item["mapping"]["id"]: item.get("digest") for item in report["models"]["models"]
    }
    if current_digests != ready.get("model_digests"):
        print(
            f"model digests changed since READY smoke: {current_digests} != "
            f"{ready.get('model_digests')}",
            file=sys.stderr,
        )
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
    write_json(
        settings.output_dir / "status.json",
        {"state": "INFERENCE_COMPLETE_EVALUATION_PENDING", "updated_at": utc_now()},
    )
    print("\nAll models' inference complete. Evaluation intentionally NOT run here.")
    print("Next: use tools/evaluate_model.py + tools/auto_resume_evaluation.py per model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
