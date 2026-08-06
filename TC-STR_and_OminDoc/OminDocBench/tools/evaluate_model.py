#!/usr/bin/env python3
"""Run the pinned official evaluator for one model from an existing full run.

This tool intentionally lives outside the runner's code-hash inputs.  It is an
offline adapter for an already-created immutable run and never imports the
current Settings, calls Ollama, or changes the inference checkpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import pathlib
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _container_ops import (  # noqa: E402
    find_containers_with_mount_prefix,
    sanitize_container_name,
    stop_container,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = pathlib.Path("/opt/ocf-ai/outputs/omnidocbench_v1_6")
EXPECTED_RESULT_FRAGMENTS = (
    "metric_result",
    "text_block_result",
    "display_formula_result",
    "table_result",
    "reading_order_result",
)
# The end2end config's element names, fixed by the pinned/hash-verified
# official_config.yaml this pipeline already assumes elsewhere (see
# EXPECTED_RESULT_FRAGMENTS above); batching needs these to find and merge
# each batch's per-element raw result file.
ELEMENT_NAMES = ("text_block", "display_formula", "table", "reading_order")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_filename(image_filename: str) -> str:
    return pathlib.Path(image_filename).with_suffix(".md").name


def safe_run_dir(output_root: pathlib.Path, run_id: str) -> pathlib.Path:
    if not run_id or run_id in {".", ".."} or "/" in run_id:
        raise ValueError(f"invalid run id: {run_id!r}")
    root = output_root.resolve()
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root:
        raise ValueError("run directory escapes the configured output root")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    return run_dir


def load_immutable_run(run_dir: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    immutable = run_dir / "immutable_config"
    hashes = read_json(immutable / "hashes.json")
    for name in ("benchmark.json", "models.yaml"):
        path = immutable / name
        expected = hashes.get(name)
        if not expected or sha256_file(path) != expected:
            raise RuntimeError(f"immutable config hash mismatch: {path}")
    benchmark = read_json(immutable / "benchmark.json")
    models = read_json(immutable / "models.yaml")["models"]
    return benchmark, models


def load_ground_truth(benchmark: dict[str, Any]) -> tuple[pathlib.Path, list[dict[str, Any]]]:
    config = benchmark["benchmark"]
    gt = (
        pathlib.Path(benchmark["paths"]["dataset_root"])
        / config["ground_truth_filename"]
    )
    if not gt.is_file():
        raise FileNotFoundError(f"ground truth is missing: {gt}")
    actual_hash = sha256_file(gt)
    if actual_hash != config["ground_truth_sha256"]:
        raise RuntimeError(
            f"ground-truth SHA-256 mismatch: {actual_hash} != "
            f"{config['ground_truth_sha256']}"
        )
    pages = read_json(gt)
    if len(pages) != config["expected_pages"]:
        raise RuntimeError(
            f"ground-truth page count mismatch: {len(pages)} != "
            f"{config['expected_pages']}"
        )
    return gt, pages


def gt_filenames(pages: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for index, page in enumerate(pages):
        try:
            name = pathlib.Path(page["page_info"]["image_path"]).name
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"GT page {index} has no valid image_path") from exc
        names.append(name)
    if len(set(names)) != len(names):
        raise RuntimeError("ground-truth image filenames are not unique")
    pred_names = [prediction_filename(name) for name in names]
    if len(set(pred_names)) != len(pred_names):
        raise RuntimeError("ground-truth prediction filenames are not unique")
    return names


def load_model_rows(
    run_dir: pathlib.Path, model_id: str, expected_filenames: set[str]
) -> tuple[str, list[dict[str, Any]]]:
    db_path = run_dir / "results.sqlite"
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        run_rows = list(connection.execute("SELECT signature FROM run"))
        if len(run_rows) != 1:
            raise RuntimeError(f"expected exactly one run signature, found {len(run_rows)}")
        signature = run_rows[0]["signature"]
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT model_id, page_id, filename, status, prediction_path,
                       prediction_sha256, attempts, error_history_json
                FROM page_result
                WHERE signature = ? AND model_id = ?
                ORDER BY filename
                """,
                (signature, model_id),
            )
        ]
    finally:
        connection.close()
    filenames = [row["filename"] for row in rows]
    if len(filenames) != len(set(filenames)):
        raise RuntimeError(f"duplicate checkpoint filename for model {model_id}")
    actual = set(filenames)
    if actual != expected_filenames:
        missing = sorted(expected_filenames - actual)
        extra = sorted(actual - expected_filenames)
        raise RuntimeError(
            f"model does not have terminal coverage for the full GT; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return signature, rows


def eval_timeout_pages_path(model_base: pathlib.Path) -> pathlib.Path:
    return model_base / "evaluation_timeout_pages.json"


def load_eval_timeout_pages(model_base: pathlib.Path) -> dict[str, dict[str, Any]]:
    path = eval_timeout_pages_path(model_base)
    if not path.is_file():
        return {}
    data = read_json(path)
    return {
        entry["filename"]: entry
        for entry in data.get("pages", [])
        if entry.get("filename")
    }


def prepare_entries(
    rows: list[dict[str, Any]],
    *,
    failed_as_empty: bool,
    eval_timeout_pages: dict[str, dict[str, Any]] | None = None,
    apply_eval_timeout_as_empty: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    eval_timeout_pages = eval_timeout_pages or {}
    entries: list[dict[str, Any]] = []
    counts = {"SUCCESS": 0, "FAILED_AS_EMPTY": 0}
    for row in rows:
        pred_name = prediction_filename(row["filename"])
        if row["status"] == "SUCCESS":
            source = pathlib.Path(row["prediction_path"] or "")
            if not source.is_file():
                raise RuntimeError(f"successful prediction is missing: {source}")
            actual_hash = sha256_file(source)
            if actual_hash != row["prediction_sha256"]:
                raise RuntimeError(f"prediction hash mismatch: {source}")
            timeout_record = eval_timeout_pages.get(row["filename"])
            if timeout_record and apply_eval_timeout_as_empty:
                entry = {
                    "filename": row["filename"],
                    "prediction_filename": pred_name,
                    "checkpoint_status": "SUCCESS",
                    "evaluation_representation": "EMPTY_PREDICTION_EVAL_TIMEOUT",
                    "original_prediction_sha256": actual_hash,
                    "prediction_sha256": hashlib.sha256(b"").hexdigest(),
                    "attempts": row["attempts"],
                    "eval_timeout_reason": timeout_record.get("reason", ""),
                    "eval_timeout_recorded_at": timeout_record.get("recorded_at", ""),
                }
                counts["EVAL_TIMEOUT_AS_EMPTY"] = counts.get("EVAL_TIMEOUT_AS_EMPTY", 0) + 1
            else:
                entry = {
                    "filename": row["filename"],
                    "prediction_filename": pred_name,
                    "checkpoint_status": "SUCCESS",
                    "source_path": str(source),
                    "prediction_sha256": actual_hash,
                    "attempts": row["attempts"],
                }
                counts["SUCCESS"] += 1
        elif row["status"] == "FAILED":
            if not failed_as_empty:
                raise RuntimeError(
                    "terminal FAILED pages exist; rerun with --failed-as-empty "
                    "to include them transparently as empty predictions"
                )
            entry = {
                "filename": row["filename"],
                "prediction_filename": pred_name,
                "checkpoint_status": "FAILED",
                "evaluation_representation": "EMPTY_PREDICTION",
                "prediction_sha256": hashlib.sha256(b"").hexdigest(),
                "attempts": row["attempts"],
                "error_history": json.loads(row["error_history_json"] or "[]"),
            }
            counts["FAILED_AS_EMPTY"] += 1
        else:
            raise RuntimeError(
                f"unsupported checkpoint status {row['status']!r} for {row['filename']}"
            )
        entries.append(entry)
    return entries, counts


def validate_or_create_staging(
    model_base: pathlib.Path, entries: list[dict[str, Any]], input_hash: str
) -> tuple[pathlib.Path, pathlib.Path]:
    staging = model_base / f"official_predictions_full_1651_{input_hash[:12]}"
    manifest_path = model_base / f"{staging.name}_manifest.json"
    if staging.exists():
        if not manifest_path.is_file():
            raise RuntimeError(f"existing staging directory has no manifest: {staging}")
        manifest = read_json(manifest_path)
        if manifest.get("evaluation_input_hash") != input_hash:
            raise RuntimeError(f"existing staging manifest hash mismatch: {manifest_path}")
        actual_names = {path.name for path in staging.iterdir() if path.is_file()}
        expected_names = {entry["prediction_filename"] for entry in entries}
        if actual_names != expected_names:
            raise RuntimeError("existing staging prediction filenames do not match manifest")
        for entry in entries:
            path = staging / entry["prediction_filename"]
            if sha256_file(path) != entry["prediction_sha256"]:
                raise RuntimeError(f"existing staged prediction hash mismatch: {path}")
        return staging, manifest_path

    temporary = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{staging.name}.", dir=model_base)
    )
    try:
        for entry in entries:
            target = temporary / entry["prediction_filename"]
            source_value = entry.get("source_path")
            if source_value:
                source = pathlib.Path(source_value)
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
            else:
                target.write_bytes(b"")
        os.replace(temporary, staging)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return staging, manifest_path


def official_config_path(run_dir: pathlib.Path, model_id: str) -> pathlib.Path:
    path = run_dir / "immutable_config" / f"{model_id}_official.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"immutable official config is missing: {path}")
    immutable = run_dir / "immutable_config"
    expected = read_json(immutable / "hashes.json").get("official_run.yaml")
    if not expected or sha256_file(path) != expected:
        raise RuntimeError(f"immutable official config hash mismatch: {path}")
    return path


def image_reference(benchmark: dict[str, Any]) -> str:
    evaluator = benchmark["evaluator"]
    return f"{evaluator['docker_image']}@{evaluator['docker_digest']}"


def parse_results(result_dir: pathlib.Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(result_dir.glob("*.json")):
        if path.name in {"evaluation_manifest.json", "runner_invocation.json"}:
            continue
        try:
            files[path.name] = read_json(path)
        except Exception as exc:  # evaluator output remains preserved for inspection
            files[path.name] = {"parse_error": str(exc)}
    missing = [
        fragment
        for fragment in EXPECTED_RESULT_FRAGMENTS
        if not any(fragment in name for name in files)
    ]
    return {
        "status": "PASS" if not missing else "FAIL",
        "missing_expected": missing,
        "files": files,
    }


def write_summary_html(
    path: pathlib.Path,
    *,
    manifest: dict[str, Any],
    parsed: dict[str, Any],
) -> None:
    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>OmniDocBench v1.6 single-model official evaluation</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:1rem}}
.notice{{border-left:4px solid #b66a00;padding:.7rem 1rem;background:#fff8e8}}
</style></head><body>
<h1>OmniDocBench v1.6 single-model official evaluation</h1>
<p class="notice">這是指定模型的官方 evaluator 結果；FAILED_AS_EMPTY 頁面以空 prediction
納入完整 1,651-page 分母，原始 checkpoint 未被修改。</p>
<h2>Evaluation manifest</h2>
<pre>{html.escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre>
<h2>Parsed official artifacts</h2>
<pre>{html.escape(json.dumps(parsed, ensure_ascii=False, indent=2))}</pre>
</body></html>
"""
    path.write_text(document, encoding="utf-8")


def run_docker_evaluator(
    command: list[str],
    *,
    container_name: str,
    timeout_seconds: int,
    stall_seconds: int,
) -> dict[str, Any]:
    """Run the pinned evaluator, detecting both an overall timeout and a stall.

    A stall (no new stdout/stderr for ``stall_seconds``) is distinguished from
    ordinary slowness: the official evaluator streams a tqdm progress line
    continuously while genuinely working, so a long gap in output means the
    container is very likely hung, not merely processing a slow page. Either
    condition force-stops the container by name so no orphaned container is
    left behind, which a plain ``subprocess.run(timeout=...)`` does not
    guarantee (killing the ``docker run`` client does not stop the
    daemon-managed container).
    """
    started = utc_now()
    start_monotonic = time.monotonic()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    state = {"last_output": start_monotonic}
    state_lock = threading.Lock()

    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    def _reader(stream: Any, sink: list[str]) -> None:
        for line in iter(stream.readline, ""):
            sink.append(line)
            with state_lock:
                state["last_output"] = time.monotonic()
        stream.close()

    threads = [
        threading.Thread(target=_reader, args=(process.stdout, stdout_lines), daemon=True),
        threading.Thread(target=_reader, args=(process.stderr, stderr_lines), daemon=True),
    ]
    for thread in threads:
        thread.start()

    def _force_stop_and_wait() -> int:
        stop_container(container_name)
        try:
            return process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=10)

    def _handle_signal(signum: int, _frame: Any) -> None:
        stop_container(container_name)
        raise KeyboardInterrupt()

    previous_sigint = signal.signal(signal.SIGINT, _handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

    outcome = "OK"
    reason = ""
    try:
        while True:
            try:
                returncode = process.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            with state_lock:
                idle = now - state["last_output"]
            if now - start_monotonic > timeout_seconds:
                outcome = "TIMEOUT"
                reason = f"evaluator exceeded overall timeout of {timeout_seconds}s"
                returncode = _force_stop_and_wait()
                break
            if idle > stall_seconds:
                outcome = "STALLED"
                reason = (
                    f"no new stdout/stderr output for {int(idle)}s "
                    f"(stall threshold {stall_seconds}s); container force-stopped"
                )
                returncode = _force_stop_and_wait()
                break
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    for thread in threads:
        thread.join(timeout=10)

    if outcome == "TIMEOUT":
        effective_returncode = 124
    elif outcome == "STALLED":
        effective_returncode = 125
    else:
        effective_returncode = returncode

    return {
        "command": command,
        "container_name": container_name,
        "started_at": started,
        "ended_at": utc_now(),
        "returncode": effective_returncode,
        "outcome": outcome,
        "stdout": "".join(stdout_lines),
        "stderr": "".join(stderr_lines) + (f"\n{reason}" if reason else ""),
    }


def split_into_batches(
    pages: list[dict[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    return [pages[i : i + batch_size] for i in range(0, len(pages), batch_size)]


def batches_root_dir(model_base: pathlib.Path, batch_size: int) -> pathlib.Path:
    # Deliberately NOT keyed by the corpus-wide input_hash: that hash covers
    # all 1,651 pages' representation, so registering one new
    # evaluation-timeout page anywhere would otherwise invalidate every
    # batch, not just the one batch containing that page. Each batch's own
    # completion is instead tracked by a per-batch content hash (see
    # batch_content_hash), so unrelated batches survive a staging change.
    return model_base / "official_evaluation_batches" / f"bs{batch_size}"


def batch_content_hash(batch_pages: list[dict[str, Any]], entries_by_filename: dict[str, Any]) -> str:
    material = []
    for page in batch_pages:
        filename = pathlib.Path(page["page_info"]["image_path"]).name
        entry = entries_by_filename[filename]
        material.append(
            {
                "filename": entry["filename"],
                "prediction_filename": entry["prediction_filename"],
                "checkpoint_status": entry["checkpoint_status"],
                "evaluation_representation": entry.get("evaluation_representation"),
                "prediction_sha256": entry["prediction_sha256"],
            }
        )
    return hashlib.sha256(canonical_json(material).encode()).hexdigest()


def batch_state_path(batches_root: pathlib.Path) -> pathlib.Path:
    return batches_root / "batch_state.json"


def load_batch_state(batches_root: pathlib.Path) -> dict[str, Any] | None:
    path = batch_state_path(batches_root)
    if not path.is_file():
        return None
    return read_json(path)


def save_batch_state(batches_root: pathlib.Path, state: dict[str, Any]) -> None:
    write_json(batch_state_path(batches_root), state)


def batch_dir(batches_root: pathlib.Path, index: int) -> pathlib.Path:
    return batches_root / f"batch_{index:04d}"


def find_batch_element_result(batch_result_dir: pathlib.Path, element: str) -> pathlib.Path:
    matches = sorted(batch_result_dir.glob(f"*_{element}_result.json"))
    if not matches:
        raise RuntimeError(f"batch result missing {element}_result.json under {batch_result_dir}")
    if len(matches) > 1:
        raise RuntimeError(
            f"ambiguous {element}_result.json candidates under {batch_result_dir}: {matches}"
        )
    return matches[0]


def run_batched_evaluation(
    *,
    args: argparse.Namespace,
    model_base: pathlib.Path,
    benchmark: dict[str, Any],
    gt_path: pathlib.Path,
    pages: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    staging: pathlib.Path,
    staging_manifest: dict[str, Any],
    config_path: pathlib.Path,
    input_hash: str,
) -> tuple[pathlib.Path, dict[str, Any], dict[str, Any]]:
    """Run the pinned evaluator over the corpus split into resumable batches.

    Every batch runs the exact same pinned image and config as a single full
    run would, over a subset of ground-truth pages; matching and per-sample
    scoring are page-independent (verified against the pinned image's own
    source), so this changes nothing about how any individual page is
    scored. Once every batch has completed, a merge step -- executed inside
    the same pinned image via tools/_merge_driver.py -- reproduces the
    official corpus-wide aggregation over the concatenated per-sample
    results, using the official aggregation code unmodified.

    Each batch's completion is tracked by a hash of just that batch's own
    pages' representation (see batch_content_hash), not the corpus-wide
    input_hash: registering a new evaluation-timeout page anywhere only
    invalidates the one batch that actually contains it, so re-running this
    command after marking a newly-discovered hang does not throw away
    already-completed, unrelated batches. A batch that fails or stalls is
    recorded in batch_state.json; re-running this same command later
    retries only batches that are missing, failed, or whose own content
    changed.
    """
    batch_size = args.batch_size
    batches = split_into_batches(pages, batch_size)
    entries_by_filename = {entry["filename"]: entry for entry in entries}
    batches_root = batches_root_dir(model_base, batch_size)
    batches_root.mkdir(parents=True, exist_ok=True)

    shared_config_copy = batches_root / "official_config.yaml"
    if not shared_config_copy.is_file():
        shutil.copyfile(config_path, shared_config_copy)

    state = load_batch_state(batches_root)
    if (
        state is None
        or state.get("batch_size") != batch_size
        or state.get("total_pages") != len(pages)
        or state.get("batch_count") != len(batches)
    ):
        state = {
            "schema_version": 2,
            "batch_size": batch_size,
            "total_pages": len(pages),
            "batch_count": len(batches),
            "batches": [],
        }
    existing_by_index = {entry["index"]: entry for entry in state.get("batches", [])}
    state["batches"] = [
        existing_by_index.get(
            index,
            {
                "index": index,
                "page_count": len(batch_pages),
                "content_hash": None,
                "status": "pending",
                "outcome": None,
                "returncode": None,
                "started_at": None,
                "ended_at": None,
                "container_name": None,
                "last_error": None,
            },
        )
        for index, batch_pages in enumerate(batches)
    ]
    save_batch_state(batches_root, state)

    reused = sum(
        1
        for index, batch_pages in enumerate(batches)
        if state["batches"][index]["status"] == "done"
        and state["batches"][index].get("content_hash")
        == batch_content_hash(batch_pages, entries_by_filename)
    )
    print(
        f"batch_mode=on batch_size={batch_size} batch_count={len(batches)} "
        f"batches_root={batches_root} reusable_from_prior_run={reused}"
    )

    for index, batch_pages in enumerate(batches):
        entry = state["batches"][index]
        expected_hash = batch_content_hash(batch_pages, entries_by_filename)
        if entry["status"] == "done" and entry.get("content_hash") == expected_hash:
            continue
        if entry["status"] == "done" and entry.get("content_hash") != expected_hash:
            print(
                f"batch {index}: content changed since it last completed "
                "(e.g. a page in it was newly marked eval-timeout); re-running.",
                flush=True,
            )
        current_batch_dir = batch_dir(batches_root, index)
        current_batch_dir.mkdir(parents=True, exist_ok=True)
        gt_subset_path = current_batch_dir / "gt_subset.json"
        write_json(gt_subset_path, batch_pages)
        batch_result_dir = current_batch_dir / "result"
        batch_result_dir.mkdir(parents=True, exist_ok=True)

        for stale in find_containers_with_mount_prefix(current_batch_dir):
            print(
                f"WARNING: stopping a stale batch container: {stale['name']} ({stale['id']})",
                file=sys.stderr,
            )
            stop_container(stale["id"])

        container_name = sanitize_container_name(
            f"omnidocbench-eval-{args.run_id}-{args.model}-batch{index:04d}-"
            f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            container_name,
            "--mount",
            f"type=bind,src={gt_subset_path},dst=/workspace/gt/OmniDocBench.json,readonly",
            "--mount",
            f"type=bind,src={staging},dst=/workspace/data_md/predictions,readonly",
            "--mount",
            f"type=bind,src={batch_result_dir},dst=/workspace/result",
            "--mount",
            f"type=bind,src={shared_config_copy},dst=/workspace/configs/run.yaml,readonly",
            image_reference(benchmark),
            "--config",
            "configs/run.yaml",
        ]
        print(
            f"batch={index + 1}/{len(batches)} pages={len(batch_pages)} "
            f"container={container_name}",
            flush=True,
        )
        invocation = run_docker_evaluator(
            command,
            container_name=container_name,
            timeout_seconds=args.timeout_seconds,
            stall_seconds=args.stall_seconds,
        )
        write_json(current_batch_dir / "runner_invocation.json", invocation)
        (current_batch_dir / "docker_stdout.log").write_text(invocation["stdout"], encoding="utf-8")
        (current_batch_dir / "docker_stderr.log").write_text(invocation["stderr"], encoding="utf-8")

        entry["outcome"] = invocation.get("outcome")
        entry["returncode"] = invocation["returncode"]
        entry["started_at"] = invocation["started_at"]
        entry["ended_at"] = invocation["ended_at"]
        entry["container_name"] = container_name
        entry["content_hash"] = expected_hash
        if invocation["returncode"] == 0:
            batch_parsed = parse_results(batch_result_dir)
            if batch_parsed["status"] != "PASS":
                entry["status"] = "failed"
                entry["last_error"] = f"incomplete batch output: {batch_parsed['missing_expected']}"
            else:
                entry["status"] = "done"
                entry["last_error"] = None
        else:
            entry["status"] = "failed"
            entry["last_error"] = (
                f"returncode={invocation['returncode']} outcome={invocation.get('outcome')}; "
                f"see {current_batch_dir / 'docker_stderr.log'}"
            )
            if invocation.get("outcome") == "STALLED":
                print(
                    f"WARNING: batch {index} appears to have hung; inspect "
                    f"{current_batch_dir / 'docker_stdout.log'} for the page in flight, then "
                    "record it with tools/mark_eval_timeout_page.py and retry with "
                    "--eval-timeout-as-empty.",
                    file=sys.stderr,
                )
        save_batch_state(batches_root, state)

    pending = [entry for entry in state["batches"] if entry["status"] != "done"]
    if pending:
        summary = "; ".join(
            f"batch {entry['index']} ({entry['status']}: {entry.get('last_error', 'unknown')})"
            for entry in pending
        )
        raise RuntimeError(
            f"{len(pending)}/{len(batches)} batch(es) not complete: {summary}. Batch state "
            f"persisted at {batch_state_path(batches_root)}; re-run the same command to retry "
            "only the incomplete batches."
        )

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = model_base / "official_evaluation_full_1651" / f"{timestamp}_{input_hash[:12]}"
    result_dir.mkdir(parents=True, exist_ok=False)
    config_copy = result_dir / "official_config.yaml"
    shutil.copyfile(config_path, config_copy)

    merge_input_dir = result_dir / "merge_input"
    merge_input_dir.mkdir(parents=True, exist_ok=True)
    for element in ELEMENT_NAMES:
        merged_samples: list[Any] = []
        for index in range(len(batches)):
            per_element_path = find_batch_element_result(
                batch_dir(batches_root, index) / "result", element
            )
            merged_samples.extend(read_json(per_element_path))
        write_json(merge_input_dir / f"{element}_samples.json", merged_samples)

    evaluation_manifest = {
        **{key: value for key, value in staging_manifest.items() if key != "entries"},
        "evaluation_started_at": utc_now(),
        "official_config_source": str(config_path),
        "official_config_sha256": sha256_file(config_path),
        "docker_image": image_reference(benchmark),
        "ground_truth_path": str(gt_path),
        "ground_truth_sha256": benchmark["benchmark"]["ground_truth_sha256"],
        "result_directory": str(result_dir),
        "ollama_called": False,
        "batch_mode": {
            "batch_size": batch_size,
            "batch_count": len(batches),
            "batches_root": str(batches_root),
            "batches": state["batches"],
            "merge_policy": (
                "Every batch ran the identical pinned image/config over a page subset; "
                "matching and per-sample scoring are page-independent. The merge step "
                "(tools/_merge_driver.py) ran inside the same pinned image and reproduced "
                "the official corpus-wide aggregation, unmodified, over every batch's "
                "concatenated per-sample results."
            ),
        },
    }
    write_json(result_dir / "evaluation_manifest.json", evaluation_manifest)

    merge_container_name = sanitize_container_name(
        f"omnidocbench-merge-{args.run_id}-{args.model}-{timestamp}"
    )
    merge_driver_path = pathlib.Path(__file__).resolve().parent / "_merge_driver.py"
    merge_command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        merge_container_name,
        "--entrypoint",
        "python3",
        "--mount",
        f"type=bind,src={gt_path},dst=/workspace/gt/OmniDocBench.json,readonly",
        "--mount",
        f"type=bind,src={merge_input_dir},dst=/workspace/merge_input,readonly",
        "--mount",
        f"type=bind,src={result_dir},dst=/workspace/result",
        "--mount",
        f"type=bind,src={config_copy},dst=/workspace/configs/run.yaml,readonly",
        "--mount",
        f"type=bind,src={merge_driver_path},dst=/workspace/_merge_driver.py,readonly",
        image_reference(benchmark),
        "/workspace/_merge_driver.py",
    ]
    print(f"merging {len(batches)} batch(es); container={merge_container_name}", flush=True)
    invocation = run_docker_evaluator(
        merge_command,
        container_name=merge_container_name,
        timeout_seconds=args.timeout_seconds,
        stall_seconds=args.stall_seconds,
    )
    write_json(result_dir / "runner_invocation.json", invocation)
    (result_dir / "docker_stdout.log").write_text(invocation["stdout"], encoding="utf-8")
    (result_dir / "docker_stderr.log").write_text(invocation["stderr"], encoding="utf-8")
    shutil.rmtree(merge_input_dir, ignore_errors=True)
    return result_dir, evaluation_manifest, invocation


def run_evaluation(args: argparse.Namespace) -> pathlib.Path:
    if args.batch_size < 0:
        raise RuntimeError("--batch-size must be >= 0 (0 disables batching)")
    output_root = pathlib.Path(args.output_root)
    run_dir = safe_run_dir(output_root, args.run_id)
    benchmark, models = load_immutable_run(run_dir)
    model_by_id = {model["id"]: model for model in models}
    if args.model not in model_by_id:
        raise RuntimeError(
            f"unknown model id {args.model!r}; available: {sorted(model_by_id)}"
        )
    model = model_by_id[args.model]
    gt_path, pages = load_ground_truth(benchmark)
    filenames = gt_filenames(pages)
    signature, rows = load_model_rows(run_dir, args.model, set(filenames))
    model_base = run_dir / "models" / args.model
    eval_timeout_map = load_eval_timeout_pages(model_base)
    entries, counts = prepare_entries(
        rows,
        failed_as_empty=args.failed_as_empty,
        eval_timeout_pages=eval_timeout_map,
        apply_eval_timeout_as_empty=args.eval_timeout_as_empty,
    )
    if eval_timeout_map:
        if args.eval_timeout_as_empty:
            print(
                f"NOTE: {counts.get('EVAL_TIMEOUT_AS_EMPTY', 0)} page(s) substituted with "
                f"empty predictions per {eval_timeout_pages_path(model_base)} "
                "(--eval-timeout-as-empty)."
            )
        else:
            print(
                f"WARNING: {len(eval_timeout_map)} known evaluation-timeout page(s) recorded "
                f"in {eval_timeout_pages_path(model_base)} but --eval-timeout-as-empty was not "
                "passed; using their real predictions, which previously caused the official "
                "evaluator to hang.",
                file=sys.stderr,
            )
    input_material = {
        "run_signature": signature,
        "model_id": args.model,
        "gt_sha256": benchmark["benchmark"]["ground_truth_sha256"],
        "entries": [
            {
                "filename": entry["filename"],
                "prediction_filename": entry["prediction_filename"],
                "checkpoint_status": entry["checkpoint_status"],
                "evaluation_representation": entry.get("evaluation_representation"),
                "prediction_sha256": entry["prediction_sha256"],
            }
            for entry in entries
        ],
    }
    input_hash = hashlib.sha256(canonical_json(input_material).encode()).hexdigest()
    staging, staging_manifest_path = validate_or_create_staging(
        model_base, entries, input_hash
    )
    staging_manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "policy": (
            "All terminal pages are included. SUCCESS uses the hash-verified canonical "
            "prediction; FAILED is represented by an empty Markdown prediction only in "
            "this staging directory. EVAL_TIMEOUT_AS_EMPTY pages had a real SUCCESS "
            "prediction but are represented as empty in this staging directory only because "
            "they are recorded in evaluation_timeout_pages.json as causing the official "
            "evaluator to hang; the original prediction and its SHA-256 remain on record. "
            "The checkpoint and canonical predictions are unchanged."
        ),
        "run_id": args.run_id,
        "run_signature": signature,
        "model": model,
        "expected_pages": benchmark["benchmark"]["expected_pages"],
        "counts": counts,
        "evaluation_input_hash": input_hash,
        "staging_directory": str(staging),
        "entries": entries,
    }
    if not staging_manifest_path.exists():
        write_json(staging_manifest_path, staging_manifest)
    if args.prepare_only:
        print(f"run_id={args.run_id}")
        print(f"model={args.model}")
        print(
            f"pages={len(entries)} success={counts['SUCCESS']} "
            f"failed_as_empty={counts['FAILED_AS_EMPTY']} "
            f"eval_timeout_as_empty={counts.get('EVAL_TIMEOUT_AS_EMPTY', 0)}"
        )
        print(f"Prepared and validated evaluation input: {staging}")
        print("Docker and Ollama were not called.")
        return staging

    stale_containers = find_containers_with_mount_prefix(model_base)
    for stale in stale_containers:
        print(
            f"WARNING: stopping a stale evaluator container left running from a previous "
            f"attempt: {stale['name']} ({stale['id']}, status={stale['status']})",
            file=sys.stderr,
        )
        stop_container(stale["id"])

    config_path = official_config_path(run_dir, args.model)
    inspected = subprocess.run(
        ["docker", "image", "inspect", image_reference(benchmark)],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if inspected.returncode != 0:
        raise RuntimeError(
            "pinned evaluator image is not installed locally; refusing to pull or "
            f"substitute it: {inspected.stderr.strip()}"
        )
    print(f"run_id={args.run_id}")
    print(f"model={args.model}")
    print(f"pages={len(entries)} success={counts['SUCCESS']} "
          f"failed_as_empty={counts['FAILED_AS_EMPTY']} "
          f"eval_timeout_as_empty={counts.get('EVAL_TIMEOUT_AS_EMPTY', 0)}")

    if args.batch_size:
        result_dir, evaluation_manifest, invocation = run_batched_evaluation(
            args=args,
            model_base=model_base,
            benchmark=benchmark,
            gt_path=gt_path,
            pages=pages,
            entries=entries,
            staging=staging,
            staging_manifest=staging_manifest,
            config_path=config_path,
            input_hash=input_hash,
        )
    else:
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result_dir = (
            model_base
            / "official_evaluation_full_1651"
            / f"{timestamp}_{input_hash[:12]}"
        )
        result_dir.mkdir(parents=True, exist_ok=False)
        config_copy = result_dir / "official_config.yaml"
        shutil.copyfile(config_path, config_copy)
        evaluation_manifest = {
            **{key: value for key, value in staging_manifest.items() if key != "entries"},
            "evaluation_started_at": utc_now(),
            "official_config_source": str(config_path),
            "official_config_sha256": sha256_file(config_path),
            "docker_image": image_reference(benchmark),
            "ground_truth_path": str(gt_path),
            "ground_truth_sha256": benchmark["benchmark"]["ground_truth_sha256"],
            "result_directory": str(result_dir),
            "ollama_called": False,
        }
        write_json(result_dir / "evaluation_manifest.json", evaluation_manifest)

        container_name = sanitize_container_name(
            f"omnidocbench-eval-{args.run_id}-{args.model}-{timestamp}"
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            container_name,
            "--mount",
            f"type=bind,src={gt_path},dst=/workspace/gt/OmniDocBench.json,readonly",
            "--mount",
            f"type=bind,src={staging},dst=/workspace/data_md/predictions,readonly",
            "--mount",
            f"type=bind,src={result_dir},dst=/workspace/result",
            "--mount",
            f"type=bind,src={config_copy},dst=/workspace/configs/run.yaml,readonly",
            image_reference(benchmark),
            "--config",
            "configs/run.yaml",
        ]
        print(f"result_dir={result_dir}")
        print(f"container_name={container_name}")
        print("Starting pinned official Docker evaluator; Ollama is not called.", flush=True)
        invocation = run_docker_evaluator(
            command,
            container_name=container_name,
            timeout_seconds=args.timeout_seconds,
            stall_seconds=args.stall_seconds,
        )
        write_json(result_dir / "runner_invocation.json", invocation)
        (result_dir / "docker_stdout.log").write_text(
            invocation["stdout"], encoding="utf-8"
        )
        (result_dir / "docker_stderr.log").write_text(
            invocation["stderr"], encoding="utf-8"
        )

    if invocation["returncode"] != 0:
        detail = ""
        if invocation.get("outcome") == "STALLED":
            detail = (
                "\nThe evaluator container produced no new output for "
                f"{args.stall_seconds}s and was force-stopped; it was very likely hung, "
                f"not merely slow. Inspect {result_dir / 'docker_stdout.log'} for the last "
                "progress line to find which ground-truth page was in flight. If you can "
                "confirm it is a known/reproducible hang, record it with "
                "tools/mark_eval_timeout_page.py and retry with --eval-timeout-as-empty."
            )
        raise RuntimeError(
            f"official evaluator exited {invocation['returncode']} "
            f"({invocation.get('outcome', 'FAILED')}); see {result_dir / 'docker_stderr.log'}"
            f"{detail}"
        )
    parsed = parse_results(result_dir)
    write_json(result_dir / "summary_official_selected_model.json", parsed)
    write_summary_html(
        result_dir / "report_official_selected_model.html",
        manifest=evaluation_manifest,
        parsed=parsed,
    )
    if parsed["status"] != "PASS":
        raise RuntimeError(
            f"official evaluator output is incomplete: {parsed['missing_expected']}"
        )
    print(f"Official evaluation complete: {result_dir}")
    return result_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pinned OmniDocBench v1.6 official evaluation for exactly one "
            "model from an existing durable run"
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True, help="logical model id")
    parser.add_argument(
        "--failed-as-empty",
        action="store_true",
        help=(
            "represent terminal FAILED pages as empty .md files in an isolated "
            "evaluation staging directory"
        ),
    )
    parser.add_argument(
        "--eval-timeout-as-empty",
        action="store_true",
        help=(
            "represent ground-truth pages recorded in evaluation_timeout_pages.json "
            "(known to hang the official evaluator; see tools/mark_eval_timeout_page.py) "
            "as empty Markdown predictions in an isolated evaluation staging directory; "
            "never applied automatically"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="validate and stage the selected model without starting Docker",
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--stall-seconds",
        type=int,
        default=900,
        help=(
            "force-stop the evaluator container if it produces no new stdout/stderr "
            "output for this many seconds (default 900 = 15 minutes); distinguishes a "
            "hang from ordinary slow-page processing, which streams progress continuously"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "split the corpus into resumable batches of this many pages, each run "
            "through the identical pinned image/config; once every batch has "
            "completed, a merge step (tools/_merge_driver.py, run inside the same "
            "pinned image) reproduces the official corpus-wide aggregation over all "
            "batches' results unmodified. A batch that fails or stalls is recorded in "
            "batch_state.json under the run's official_evaluation_batches directory; "
            "re-running this same command retries only the incomplete batches. "
            "0 (default) disables batching and runs the corpus in one invocation."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        run_evaluation(build_parser().parse_args(argv))
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
