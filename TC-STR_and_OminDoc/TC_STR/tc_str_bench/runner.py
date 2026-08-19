from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .metrics import aligned_supplementary, anomaly_flags, normalize_primary, score
from .ollama import OllamaClient, ollama_ps, transport_accepted
from .reporting import build_main_report, write_smoke_report
from .storage import Checkpoint
from .util import append_log, atomic_write_json, utc_now


class StopRequested(Exception):
    pass


_STOP = False


def _signal_handler(signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _status(run_dir: Path, data: dict[str, Any]) -> None:
    atomic_write_json(run_dir / "status.json", data)


def _client(settings: Settings) -> OllamaClient:
    cfg = settings.raw["ollama"]
    return OllamaClient(cfg["host"], cfg["timeout_seconds"], cfg["max_attempts"], cfg["retry_backoff_seconds"])


def _processor_for(tag: str) -> tuple[bool, dict[str, Any]]:
    ps = ollama_ps()
    matching = [row for row in ps["rows"] if row.get("name") == tag]
    return bool(matching and matching[0].get("processor") == "100% GPU"), ps


def run_phase(
    settings: Settings,
    run_dir: Path,
    preflight: dict[str, Any],
    manifest: dict[str, Any],
    phase: str,
) -> list[str]:
    if phase not in {"smoke", "full"}:
        raise ValueError(phase)
    signature = preflight["run_signature"]
    samples = manifest["samples"]
    if phase == "smoke":
        wanted = set(manifest["smoke_selection"]["sample_indices"])
        samples = [sample for sample in samples if sample["index"] in wanted]
    client = _client(settings)
    checkpoint = Checkpoint(run_dir / "results.sqlite")
    log = run_dir / "run.log"
    error_log = run_dir / "error.log"
    blockers: list[str] = []
    timings_path = run_dir / "model_timings.json"
    model_timings = (
        json.loads(timings_path.read_text(encoding="utf-8")) if timings_path.exists() else {}
    )
    install_signal_handlers()
    model_metadata = {model["id"]: model for model in preflight["models"]}
    try:
        for model_position, configured in enumerate(settings.models):
            if _STOP:
                raise StopRequested()
            model = model_metadata[configured["id"]]
            tag = str(model["exact_tag"])
            model_started = utc_now()
            append_log(log, f"{phase} model_start id={configured['id']} tag={tag}")
            warm_sample = samples[0]
            warm_attempts = client.generate(
                tag,
                settings.prompt,
                settings.dataset_dir / warm_sample["image_relative_path"],
                settings.options,
                keep_alive="5m",
                completion_policy=configured.get("completion_policy"),
            )
            if not transport_accepted(warm_attempts[-1]):
                blockers.append(f"{configured['logical_name']}: warm-up 未取得 HTTP 200 模型回應")
                append_log(error_log, f"warmup_failed model={configured['id']} error={warm_attempts[-1].error_message}")
                try:
                    client.unload(tag)
                except Exception:
                    pass
                continue
            gpu_ok, ps = _processor_for(tag)
            if not gpu_ok:
                blockers.append(f"{configured['logical_name']}: smoke/full warm-up 後不是精確 100% GPU")
                append_log(error_log, f"processor_block model={configured['id']} ps={json.dumps(ps, ensure_ascii=False)}")
                try:
                    client.unload(tag)
                except Exception:
                    pass
                continue
            for position, sample in enumerate(samples, 1):
                if _STOP:
                    raise StopRequested()
                index = int(sample["index"])
                if checkpoint.successful(signature, phase, configured["id"], index):
                    continue
                initial_attempt = checkpoint.next_attempt(signature, phase, configured["id"], index)
                attempts = client.generate(
                    tag,
                    settings.prompt,
                    settings.dataset_dir / sample["image_relative_path"],
                    settings.options,
                    initial_attempt=initial_attempt,
                    keep_alive="5m",
                    completion_policy=configured.get("completion_policy"),
                )
                final = attempts[-1]
                for attempt in attempts:
                    checkpoint.record_attempt(
                        {
                            "run_signature": signature,
                            "phase": phase,
                            "model_id": configured["id"],
                            "sample_index": index,
                            "attempt_number": attempt.attempt_number,
                            "started_at": attempt.started_at,
                            "ended_at": attempt.ended_at,
                            "success": attempt.success,
                            "request": {
                                **attempt.request,
                                "image_sha256": sample["image_sha256"],
                                "prompt_sha256": settings.prompt_hash,
                            },
                            "response": {
                                key: value for key, value in (attempt.response or {}).items()
                                if key != "context"
                            } if attempt.response is not None else None,
                            "http_status": attempt.http_status,
                            "error_type": attempt.error_type,
                            "error_message": attempt.error_message,
                            "latency_seconds": attempt.latency_seconds,
                            "completion_validation": attempt.completion_validation,
                        }
                    )
                response = final.response or {}
                raw = str(response.get("response") or "")
                primary = normalize_primary(raw)
                flags = anomaly_flags(
                    raw,
                    settings.prompt,
                    response.get("done_reason"),
                    response.get("eval_count"),
                    int(settings.options["num_predict"]),
                )
                flags["truncation_unknown"] = not bool(
                    final.completion_validation.get("truncation_observable", True)
                )
                pipeline_anomaly = any(
                    flags[name]
                    for name in (
                        "prompt_echo", "refusal", "formatting", "repetition",
                        "thinking", "unrelated", "truncated",
                    )
                )
                if not transport_accepted(final):
                    record_status = "request_failure"
                elif flags["empty"] or flags["control_only"]:
                    record_status = "model_output_failure"
                elif pipeline_anomaly:
                    record_status = "completed_with_anomaly"
                else:
                    record_status = "completed"
                if record_status != "completed":
                    append_log(
                        error_log,
                        f"scored_failure phase={phase} model={configured['id']} index={index} "
                        f"status={record_status} error={final.error_message}",
                    )
                aligned, changes = aligned_supplementary(raw)
                checkpoint.record_success(
                    {
                        "run_signature": signature,
                        "phase": phase,
                        "model_id": configured["id"],
                        "sample_index": index,
                        "exact_tag": tag,
                        "model_digest": model.get("digest") or "",
                        "image_relative_path": sample["image_relative_path"],
                        "image_sha256": sample["image_sha256"],
                        "ground_truth": sample["ground_truth"],
                        "prediction_raw": raw,
                        "prediction_primary": primary,
                        "prediction_aligned": aligned,
                        "aligned_changes": changes,
                        "primary_metrics": score(primary, sample["ground_truth"]),
                        "aligned_metrics": score(aligned, sample["ground_truth"]),
                        "anomaly_flags": flags,
                        "ollama_metadata": {
                            key: value for key, value in response.items()
                            if key not in {"response", "context"}
                        },
                        "done_reason": response.get("done_reason"),
                        "prompt_eval_count": response.get("prompt_eval_count"),
                        "eval_count": response.get("eval_count"),
                        "completion_validation": final.completion_validation,
                        "record_status": record_status,
                        "http_status": final.http_status,
                        "error_type": final.error_type,
                        "error_message": final.error_message,
                        "latency_seconds": final.latency_seconds,
                        "attempt_count": final.attempt_number,
                    }
                )
                if position % int(settings.raw["checkpoint"]["snapshot_every"]) == 0 or position == len(samples):
                    counts = checkpoint.counts(signature, phase)
                    _status(
                        run_dir,
                        {
                            "run_id": run_dir.name,
                            "status": f"{phase.upper()}_RUNNING",
                            "phase": phase,
                            "model_id": configured["id"],
                            "model_position": model_position + 1,
                            "sample_position": position,
                            "sample_total": len(samples),
                            "counts": counts,
                            "updated_at": utc_now(),
                            "output_path": str(run_dir),
                        },
                    )
            try:
                client.unload(tag)
            except Exception as exc:
                blockers.append(f"{configured['logical_name']}: unload 失敗: {exc}")
            model_ended = utc_now()
            model_timings[f"{phase}:{configured['id']}"] = {
                "phase": phase,
                "model_id": configured["id"],
                "started_at": model_started,
                "ended_at": model_ended,
            }
            atomic_write_json(timings_path, model_timings)
            metadata_path = run_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            build_main_report(settings, run_dir, signature, phase, manifest, metadata)
            append_log(log, f"{phase} model_end id={configured['id']} started_at={model_started} ended_at={model_ended}")
    except StopRequested:
        _status(run_dir, {"run_id": run_dir.name, "status": "INTERRUPTED", "phase": phase, "updated_at": utc_now(), "output_path": str(run_dir)})
        append_log(log, f"{phase} interrupted; successful checkpoint rows preserved")
        raise
    finally:
        checkpoint.close()
    return blockers


def assess_smoke(
    settings: Settings,
    run_dir: Path,
    preflight: dict[str, Any],
    manifest: dict[str, Any],
    runtime_blockers: list[str],
) -> dict[str, Any]:
    checkpoint = Checkpoint(run_dir / "results.sqlite")
    try:
        rows = checkpoint.results(preflight["run_signature"], "smoke")
    finally:
        checkpoint.close()
    by_model: dict[str, list[dict[str, Any]]] = {model["id"]: [] for model in settings.models}
    for row in rows:
        by_model[row["model_id"]].append(row)
    blockers = list(runtime_blockers)
    warnings: list[str] = []
    for model in settings.models:
        model_rows = by_model[model["id"]]
        if len(model_rows) != 20:
            blockers.append(
                f"{model['logical_name']}: terminal smoke 結果 {len(model_rows)}/20；"
                "存在未形成可評分 checkpoint 的工程性缺口"
            )
        if model_rows and not any(row.get("http_status") == 200 for row in model_rows):
            blockers.append(
                f"{model['logical_name']}: 20 筆 smoke 均未取得 HTTP 200 模型回應"
            )
        flags = [row["anomaly_flags"] for row in model_rows]
        empty = sum(flag["empty"] or flag["control_only"] for flag in flags)
        anomaly_count = sum(
            any(flag[name] for name in ("prompt_echo", "refusal", "formatting", "repetition", "thinking", "unrelated"))
            for flag in flags
        )
        truncated = sum(flag["truncated"] for flag in flags)
        request_failures = sum(row.get("record_status") == "request_failure" for row in model_rows)
        if empty or anomaly_count or truncated or request_failures:
            warnings.append(
                f"{model['logical_name']}: 在共同使用條件下計入模型失敗／異常："
                f"request_failure={request_failures}, empty/control-only={empty}, "
                f"pipeline_anomalies={anomaly_count}, truncated={truncated}"
            )
    status = "BLOCKED" if blockers else "AWAITING_AI_MANUAL_REVIEW"
    return write_smoke_report(
        settings, run_dir, preflight, manifest, status, blockers, warnings=warnings
    )


def build_reports(settings: Settings, run_dir: Path) -> None:
    preflight = json.loads((run_dir / "preflight_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"run_id": run_dir.name}
    checkpoint = Checkpoint(run_dir / "results.sqlite")
    try:
        full_count = len(checkpoint.results(preflight["run_signature"], "full"))
    finally:
        checkpoint.close()
    phase = "full" if full_count else "smoke"
    build_main_report(settings, run_dir, preflight["run_signature"], phase, manifest, metadata)
