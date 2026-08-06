#!/usr/bin/env python3
"""Monitor one selected-model official evaluation without changing it.

Designed to be run either as its own polling loop (--watch) or as a single
snapshot wrapped by the standard `watch` utility, e.g.:

    watch -n 10 ./tools/evaluation_status.sh --run-id RUN --model MODEL

Each invocation persists a small state file (outside the evaluator's own
result directory, so it can never be picked up by evaluate_model.py's
official-artifact parsing) recording the last observed progress counter and
when it was first seen at that value. This is how a stall -- the container is
still "running" but has stopped making progress -- is told apart from a page
that is merely slow: the official evaluator streams a tqdm counter
continuously while genuinely working, so no change in that counter for a long
time, combined with near-zero container CPU, means it is very likely hung.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _container_ops import (  # noqa: E402
    container_cpu_percent,
    find_container_with_mount_source,
    find_containers_with_mount_prefix,
)


DEFAULT_OUTPUT_ROOT = pathlib.Path("/opt/ocf-ai/outputs/omnidocbench_v1_6")
DEFAULT_STALL_SECONDS = 900
PROGRESS_RE = re.compile(
    r"(?P<stage>[A-Za-z][A-Za-z0-9 _/-]{1,80}):\s*"
    r"(?P<percent>\d+)%\|[^\r\n]*?\|\s*"
    r"(?P<current>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<timing>[^\]\r\n]+)\]"
)


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_run_dir(output_root: pathlib.Path, run_id: str) -> pathlib.Path:
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError(f"invalid run id: {run_id!r}")
    root = output_root.resolve()
    target = (root / run_id).resolve()
    if target.parent != root or not target.is_dir():
        raise FileNotFoundError(f"run directory not found: {target}")
    return target


def latest_result_dir(run_dir: pathlib.Path, model: str) -> pathlib.Path | None:
    root = run_dir / "models" / model / "official_evaluation_full_1651"
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    ) if root.is_dir() else []
    return candidates[0] if candidates else None


def latest_batches_dir(run_dir: pathlib.Path, model: str) -> pathlib.Path | None:
    root = run_dir / "models" / model / "official_evaluation_batches"
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    ) if root.is_dir() else []
    return candidates[0] if candidates else None


def find_container(result_dir: pathlib.Path) -> dict[str, str] | None:
    return find_container_with_mount_source(result_dir)


def parse_progress(log_text: str) -> dict[str, Any] | None:
    matches = list(PROGRESS_RE.finditer(log_text.replace("\x1b[0m", "")))
    if not matches:
        return None
    match = matches[-1]
    data: dict[str, Any] = match.groupdict()
    data["percent"] = int(data["percent"])
    data["current"] = int(data["current"])
    data["total"] = int(data["total"])
    return data


def recent_interesting_logs(log_text: str, limit: int = 6) -> list[str]:
    records = [
        re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", value).strip()
        for value in log_text.replace("\r", "\n").splitlines()
    ]
    selected = [
        value
        for value in records
        if value
        and (
            "WARNING" in value
            or "ERROR" in value
            or "INFO" in value
            or re.search(r"\d+/\d+", value)
        )
    ]
    return selected[-limit:]


def elapsed_seconds(started_at: str) -> float:
    if not started_at:
        return 0.0
    parsed = dt.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    return max(0.0, (dt.datetime.now(dt.timezone.utc) - parsed).total_seconds())


def format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_bar(percent: int, width: int = 30) -> str:
    percent = max(0, min(100, percent))
    filled = int(width * percent / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def state_path(model_base: pathlib.Path, key: str) -> pathlib.Path:
    # Deliberately outside any result_dir: evaluate_model.py's parse_results()
    # globs a result_dir for *.json and treats every file as an official
    # evaluator artifact, so our own bookkeeping must never live inside one.
    return model_base / ".evaluation_status_state" / f"{key}.json"


def load_state(model_base: pathlib.Path, key: str) -> dict[str, Any]:
    path = state_path(model_base, key)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(model_base: pathlib.Path, key: str, state: dict[str, Any]) -> None:
    path = state_path(model_base, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def update_stall_tracking(
    model_base: pathlib.Path,
    key: str,
    progress: dict[str, Any] | None,
    cpu_percent: float | None,
) -> dict[str, Any]:
    """Track how long the progress counter has sat unchanged.

    Returns {"stalled_seconds": float, "current": int|None}. A near-zero CPU
    reading is not required to flag a stall -- the counter alone not moving
    for the threshold is already suspicious -- but it is reported alongside
    so the caller can phrase the warning with more or less confidence.
    """
    state = load_state(model_base, key)
    now = time.time()
    current = progress["current"] if progress else None
    if state.get("current") == current and current is not None:
        first_seen = state.get("first_seen_at", now)
    else:
        first_seen = now
    save_state(
        model_base,
        key,
        {"current": current, "first_seen_at": first_seen, "last_cpu": cpu_percent, "checked_at": now},
    )
    return {"stalled_seconds": max(0.0, now - first_seen), "current": current}


def render_running_container(
    container: dict[str, str],
    *,
    model_base: pathlib.Path,
    stall_key: str,
    stall_seconds: int,
    prefix_lines: list[str],
) -> tuple[str, list[str]]:
    started_seconds = elapsed_seconds(container["started_at"])
    cpu = container_cpu_percent(container["full_id"])
    logs = run(["docker", "logs", "--tail", "40", container["full_id"]])
    log_text = logs.stdout + logs.stderr
    progress = parse_progress(log_text)
    stall = update_stall_tracking(model_base, stall_key, progress, cpu)

    lines = list(prefix_lines)
    lines.append(
        f"container={container['name']} ({container['id']})  "
        f"elapsed={format_hms(started_seconds)}  "
        f"cpu={'?' if cpu is None else f'{cpu:.1f}%'}"
    )

    if progress:
        bar = render_bar(progress["percent"])
        lines.append(
            f"{progress['stage'].strip()}: {bar} "
            f"{progress['current']}/{progress['total']} ({progress['percent']}%)"
        )
        lines.append(f"  timing: {progress['timing']}")
    else:
        lines.append("progress: no parseable tqdm counter yet (still starting up)")

    is_stalled = stall["stalled_seconds"] >= stall_seconds
    if is_stalled:
        cpu_note = (
            "CPU is near zero, consistent with a hang"
            if (cpu is not None and cpu < 1.0)
            else "CPU is not near zero, so this may still be a very slow page rather than a hang"
        )
        lines.append(
            f"*** POSSIBLY STALLED: progress counter unchanged for "
            f"{format_hms(stall['stalled_seconds'])} (threshold {stall_seconds}s). {cpu_note}. "
            "If evaluate_model.py is not already going to force-stop it, consider "
            "tools/stop_evaluation.sh, then investigate the last log lines below. ***"
        )
        state = "RUNNING_STALLED"
    else:
        state = "RUNNING"

    interesting = recent_interesting_logs(log_text)
    if interesting:
        lines.append("recent_log:")
        lines.extend(f"  {value}" for value in interesting)
    return state, lines


BATCH_INDEX_RE = re.compile(r"-batch(\d{4})-")


def batch_progress_prefix(batches_dir: pathlib.Path, running_index: int | None) -> list[str]:
    state = read_json(batches_dir / "batch_state.json") if (batches_dir / "batch_state.json").is_file() else {}
    batches = state.get("batches", [])
    done = sum(1 for entry in batches if entry.get("status") == "done")
    failed = [entry for entry in batches if entry.get("status") == "failed"]
    total = state.get("batch_count", len(batches))
    lines = [f"batch_mode: {done}/{total} batch(es) done, batch_size={state.get('batch_size', '?')}"]
    if running_index is not None:
        lines.append(f"currently running: batch {running_index + 1}/{total}")
    if failed:
        lines.append(
            f"{len(failed)} batch(es) currently marked failed (will retry on next invocation): "
            + ", ".join(f"#{entry['index']}: {entry.get('last_error', '?')}" for entry in failed[:3])
        )
    return lines


def snapshot(run_dir: pathlib.Path, model: str, stall_seconds: int) -> tuple[str, list[str]]:
    model_base = run_dir / "models" / model
    header = (
        f"OmniDocBench v1.6 official evaluation status  "
        f"(run_id={run_dir.name}, model={model})"
    )
    base_lines = [
        header,
        f"time_utc={dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}",
    ]

    result_dir = latest_result_dir(run_dir, model)
    if result_dir is not None:
        container = find_container(result_dir)
        if container is not None:
            manifest_path = result_dir / "evaluation_manifest.json"
            manifest = read_json(manifest_path) if manifest_path.is_file() else {}
            lines = base_lines + [
                f"result_dir={result_dir.name}",
                (
                    "staged_input="
                    f"{manifest.get('counts', {}).get('SUCCESS', '?')} success + "
                    f"{manifest.get('counts', {}).get('FAILED_AS_EMPTY', '?')} failed_as_empty + "
                    f"{manifest.get('counts', {}).get('EVAL_TIMEOUT_AS_EMPTY', 0)} eval_timeout_as_empty"
                ),
            ]
            return render_running_container(
                container,
                model_base=model_base,
                stall_key=result_dir.name,
                stall_seconds=stall_seconds,
                prefix_lines=lines,
            )

    batches_dir = latest_batches_dir(run_dir, model)
    if batches_dir is not None:
        running = find_containers_with_mount_prefix(batches_dir)
        if running:
            container = running[0]
            match = BATCH_INDEX_RE.search(container["name"])
            running_index = int(match.group(1)) if match else None
            lines = base_lines + [f"batches_root={batches_dir.name}"]
            lines += batch_progress_prefix(batches_dir, running_index)
            stall_key = f"{batches_dir.name}_batch{running_index if running_index is not None else 'x'}"
            return render_running_container(
                container,
                model_base=model_base,
                stall_key=stall_key,
                stall_seconds=stall_seconds,
                prefix_lines=lines,
            )
        # Batch scratch exists but nothing is running right now: either between
        # batches, waiting to start the merge step, finished, or crashed.
        state_file = batches_dir / "batch_state.json"
        if state_file.is_file():
            state = read_json(state_file)
            batches = state.get("batches", [])
            done = sum(1 for entry in batches if entry.get("status") == "done")
            total = state.get("batch_count", len(batches))
            lines = base_lines + [f"batches_root={batches_dir.name}"]
            lines += batch_progress_prefix(batches_dir, None)
            if done == total and total > 0:
                lines.append(
                    "all batches complete; merge step should be starting (or a fresh "
                    "official_evaluation_full_1651 result_dir has already been created -- "
                    "re-run this status check)"
                )
                state_label = "BATCHES_DONE_AWAITING_MERGE"
            else:
                lines.append(
                    "no batch container currently running; the wrapper process may be "
                    "between batches, not running, or was interrupted. Re-running the same "
                    "evaluate_model.py command resumes only the incomplete batches."
                )
                state_label = "BATCH_IDLE"
            return state_label, lines

    if result_dir is not None:
        invocation_path = result_dir / "runner_invocation.json"
        summary_path = result_dir / "summary_official_selected_model.json"
        lines = base_lines + [f"result_dir={result_dir.name}"]
        if invocation_path.is_file():
            invocation = read_json(invocation_path)
            returncode = invocation.get("returncode")
            outcome = invocation.get("outcome", "")
            if returncode == 0 and summary_path.is_file():
                summary = read_json(summary_path)
                state = "COMPLETE" if summary.get("status") == "PASS" else "INCOMPLETE_OUTPUT"
            elif returncode == 0:
                state = "FINALIZING"
            elif outcome == "STALLED":
                state = "FAILED_STALLED"
            elif outcome == "TIMEOUT":
                state = "FAILED_TIMEOUT"
            else:
                state = "FAILED"
            lines.extend(
                [
                    f"state={state}",
                    f"returncode={returncode}",
                    f"started_at={invocation.get('started_at')}",
                    f"ended_at={invocation.get('ended_at')}",
                ]
            )
            report = result_dir / "report_official_selected_model.html"
            if report.is_file():
                lines.append(f"report={report}")
            if state.startswith("FAILED"):
                lines.append(f"error_log={result_dir / 'docker_stderr.log'}")
                lines.append(f"stdout_log={result_dir / 'docker_stdout.log'}")
            return state, lines

    evaluator = run(["pgrep", "-af", "tools/evaluate_model.py"])
    state = "STARTING" if evaluator.returncode == 0 and evaluator.stdout.strip() else "UNKNOWN"
    lines = base_lines + [
        f"state={state}",
        "progress=no running container, batch scratch, or final invocation file detected yet",
    ]
    return state, lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor a selected-model OmniDocBench official evaluation"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="one model id, or several separated by commas (e.g. "
        "gemma4_e4b,gemma4_12b,gemma4_31b) to show all their snapshots in one pass",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="loop internally instead of exiting after one snapshot; for most uses, "
        "prefer running this tool once per tick under the standard `watch` utility",
    )
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument(
        "--stall-seconds",
        type=int,
        default=DEFAULT_STALL_SECONDS,
        help="flag the run as possibly stalled if its progress counter has not moved "
        f"for this many seconds (default {DEFAULT_STALL_SECONDS})",
    )
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUT_ROOT), help=argparse.SUPPRESS
    )
    return parser


TERMINAL_STATES = {
    "COMPLETE",
    "INCOMPLETE_OUTPUT",
    "FAILED",
    "FAILED_STALLED",
    "FAILED_TIMEOUT",
    "UNKNOWN",
}
OK_STATES = {
    "COMPLETE",
    "RUNNING",
    "RUNNING_STALLED",
    "BATCH_IDLE",
    "BATCHES_DONE_AWAITING_MERGE",
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval < 2:
        print("ERROR: --interval must be at least 2 seconds", file=sys.stderr)
        return 2
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    try:
        run_dir = safe_run_dir(pathlib.Path(args.output_root), args.run_id)
        while True:
            states = []
            for index, model in enumerate(models):
                if len(models) > 1:
                    print(f"{'=' * 12} {model} {'=' * 12}")
                state, lines = snapshot(run_dir, model, args.stall_seconds)
                states.append(state)
                print("\n".join(lines), flush=True)
                if index < len(models) - 1:
                    print()
            all_terminal = all(state in TERMINAL_STATES for state in states)
            if not args.watch or all_terminal:
                return 0 if all(state in OK_STATES for state in states) else 1
            print(f"\n--- refresh in {args.interval}s; Ctrl+C stops monitoring only ---\n")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped; evaluator continues running.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
