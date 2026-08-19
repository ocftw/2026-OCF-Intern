#!/usr/bin/env python3
"""Supervise evaluate_model.py across the known repetition-loop hang.

This does not touch official_config.yaml or any pinned evaluator code. It
only automates the exact manual playbook already used three times this
session: run evaluate_model.py; if a batch stalls; confirm the stall matches
the specific signature already root-caused (a "[match-timeout] ... fallback
to chunked Hungarian" line for one page, with no subsequent
"[timeout-fallback] ..." diagnostic line -- meaning the official fallback
function itself never got to run) *and* that page's real prediction is
independently confirmed to be degenerate repeated content; if both hold,
register it via mark_eval_timeout_page.py (full evidence written, exactly
like the manual cases) and retry. Anything that does not match this exact,
already-confirmed signature stops the supervisor for a human to look at --
it never guesses on an unfamiliar failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
from collections import Counter
from typing import Any

TOOL_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = pathlib.Path("/opt/ocf-ai/outputs/omnidocbench_v1_6")

MATCH_TIMEOUT_RE = re.compile(
    r"\[match-timeout\]\s+(?P<filename>\S+\.(?:png|jpg|jpeg))\s*:\s*quick_match exceeded"
)
TIMEOUT_FALLBACK_RE = re.compile(r"\[timeout-fallback\]")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def log_event(model_base: pathlib.Path, event: dict[str, Any]) -> None:
    path = model_base / "auto_resume_supervisor_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"logged_at": utc_now(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[supervisor] {event.get('kind')}: {event.get('summary', '')}", flush=True)


def batches_root_dir(model_base: pathlib.Path, batch_size: int) -> pathlib.Path:
    return model_base / "official_evaluation_batches" / f"bs{batch_size}"


def batch_dir(batches_root: pathlib.Path, index: int) -> pathlib.Path:
    return batches_root / f"batch_{index:04d}"


def find_stalled_signature_page(batch_stdout_log: pathlib.Path) -> str | None:
    """Return the page filename if the log matches the known hang signature.

    The signature: the last "[match-timeout] <file>: quick_match exceeded
    ..." line in the log has no "[timeout-fallback] ..." line after it --
    meaning the official fallback function was entered but never actually
    ran, exactly as confirmed for the three known hangs this session. If a
    fallback line DOES follow (matching succeeded via the safe path, just
    slowly), or no match-timeout line exists at all, this returns None --
    the caller must not guess.
    """
    if not batch_stdout_log.is_file():
        return None
    text = batch_stdout_log.read_text(encoding="utf-8", errors="replace")
    matches = list(MATCH_TIMEOUT_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    if TIMEOUT_FALLBACK_RE.search(text, last.end()):
        return None  # fallback did run after this point; not the hang case
    return last.group("filename")


def is_degenerate_repetition(text: str) -> dict[str, Any] | None:
    """Independently confirm the page's real prediction is pathologically
    repetitive, using two shape-agnostic heuristics (whole-line repetition,
    and a repeated character n-gram) so either a page-59-style single huge
    line or a page-27-style many-repeated-lines pattern is caught."""
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) >= 20:
        counts = Counter(lines)
        top_line, top_count = counts.most_common(1)[0]
        ratio = top_count / len(lines)
        if ratio >= 0.4 and top_count >= 20:
            return {"method": "line_repetition", "ratio": ratio, "sample": top_line[:80]}

    # Catches the page-59-shape hang: content crammed into very few actual
    # newlines, so the line-based check above never fires. A repeated
    # "coverage fraction" derived from a single n-gram's count is
    # unreliable when stride < n (overlapping windows over- or
    # under-estimate coverage depending on the repeat period vs stride), so
    # this uses the raw count directly: no ordinary, non-degenerate prose
    # repeats the exact same 40-character sequence this many times.
    n = 40
    stride = 8
    if len(text) >= n * 10:
        gram_counts: Counter[str] = Counter()
        for i in range(0, len(text) - n, stride):
            gram_counts[text[i : i + n]] += 1
        if gram_counts:
            gram, count = gram_counts.most_common(1)[0]
            if count >= 15:
                return {
                    "method": "ngram_repeat_count",
                    "repeat_count": count,
                    "sample": gram[:80].replace("\n", "\\n"),
                }
    return None


def find_prediction_file(model_base: pathlib.Path, filename: str) -> pathlib.Path | None:
    pred_filename = pathlib.Path(filename).with_suffix(".md").name
    candidates = sorted(
        model_base.glob(f"official_predictions_full_1651_*/{pred_filename}"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    return candidates[0] if candidates else None


def confirm_and_register(
    model_base: pathlib.Path,
    filename: str,
    batch_index: int,
    batch_stdout_log: pathlib.Path,
    *,
    run_id: str,
    model: str,
) -> bool:
    prediction_path = find_prediction_file(model_base, filename)
    if prediction_path is None:
        log_event(
            model_base,
            {
                "kind": "STOP_UNCONFIRMED",
                "summary": f"batch {batch_index} stalled on {filename} but its prediction "
                "file could not be located; not auto-registering",
            },
        )
        return False

    content = prediction_path.read_text(encoding="utf-8", errors="replace")
    verdict = is_degenerate_repetition(content)
    if verdict is None:
        log_event(
            model_base,
            {
                "kind": "STOP_UNCONFIRMED",
                "summary": f"batch {batch_index} stalled on {filename}, matches the known hang "
                "log signature, but its prediction does NOT look like the known repeated-content "
                "pattern; this is a different, unfamiliar failure -- stopping for manual review",
            },
        )
        return False

    evidence_dir = model_base / "evaluation_timeout_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / f"{pathlib.Path(filename).stem}.md"
    log_excerpt = "\n".join(
        line
        for line in batch_stdout_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if "match-timeout" in line or "quick-match-timeout" in line
    )[-4000:]
    evidence_path.write_text(
        f"""# Evaluation-timeout evidence (auto-detected): {filename}

detected_at: {utc_now()}
batch_index: {batch_index}
batch_log: {batch_stdout_log}

## Signature match (same as the 3 manually-confirmed hangs this session)

A `[match-timeout] {filename}: quick_match exceeded ...` line was the last
such line in this batch's log, with no subsequent `[timeout-fallback] ...`
diagnostic line -- meaning the official chunked-Hungarian fallback function
was entered but never actually executed. This is the exact signature
root-caused earlier: a func_timeout thread-cancellation reliability issue in
the pinned evaluator, not a bug in this wrapper or the benchmark design.

Relevant log lines:
{log_excerpt}

## Independent content confirmation

Method: {verdict['method']}
Detail: {verdict}

This confirms the page's real prediction is pathologically repetitive
content (a degenerate LLM repetition-loop failure), matching the pattern of
every hang confirmed manually this session, not a false positive.

## Conclusion

Auto-registered by tools/auto_resume_evaluation.py per the standing
--eval-timeout-as-empty policy: this page's real prediction is not modified
anywhere; it is only substituted with an empty prediction in an isolated
evaluation staging directory when --eval-timeout-as-empty is passed.
""",
        encoding="utf-8",
    )

    mark_result = subprocess.run(
        [
            sys.executable,
            str(TOOL_DIR / "mark_eval_timeout_page.py"),
            "--run-id",
            run_id,
            "--model",
            model,
            "--filename",
            filename,
            "--reason",
            f"auto-detected degenerate repeated-text prediction ({verdict['method']}); "
            "matches the known func_timeout fallback hang signature confirmed manually "
            "3x this session",
            "--evidence",
            str(evidence_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    log_event(
        model_base,
        {
            "kind": "AUTO_REGISTERED",
            "summary": f"batch {batch_index}: confirmed and registered {filename} "
            f"({verdict['method']}); mark_eval_timeout_page rc={mark_result.returncode}",
            "evidence": str(evidence_path),
        },
    )
    return mark_result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Supervise evaluate_model.py across the confirmed repetition-loop hang, "
        "auto-registering only stalls that match the exact known signature."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--stall-seconds", type=int, default=900)
    parser.add_argument("--max-cycles", type=int, default=50)
    args = parser.parse_args(argv)

    model_base = pathlib.Path(args.output_root) / args.run_id / "models" / args.model
    batches_root = batches_root_dir(model_base, args.batch_size)

    for cycle in range(1, args.max_cycles + 1):
        log_event(model_base, {"kind": "CYCLE_START", "summary": f"cycle {cycle}/{args.max_cycles}"})
        command = [
            sys.executable,
            str(TOOL_DIR / "evaluate_model.py"),
            "--run-id",
            args.run_id,
            "--model",
            args.model,
            "--failed-as-empty",
            "--eval-timeout-as-empty",
            "--batch-size",
            str(args.batch_size),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--stall-seconds",
            str(args.stall_seconds),
        ]
        completed = subprocess.run(command, text=True, check=False)
        if completed.returncode == 0:
            log_event(model_base, {"kind": "SUCCESS", "summary": "evaluate_model.py completed successfully"})
            return 0

        state_path = batches_root / "batch_state.json"
        if not state_path.is_file():
            log_event(
                model_base,
                {
                    "kind": "STOP_UNCONFIRMED",
                    "summary": f"evaluate_model.py exited {completed.returncode} with no "
                    f"batch_state.json at {state_path}; not a batch-stall failure, stopping",
                },
            )
            return 1

        state = json.loads(state_path.read_text(encoding="utf-8"))
        stalled = [
            entry
            for entry in state.get("batches", [])
            if entry.get("status") == "failed" and entry.get("outcome") == "STALLED"
        ]
        if not stalled:
            log_event(
                model_base,
                {
                    "kind": "STOP_UNCONFIRMED",
                    "summary": f"evaluate_model.py exited {completed.returncode} but no batch is "
                    "marked STALLED (some other failure); stopping for manual review",
                },
            )
            return 1

        progressed = False
        for entry in stalled:
            index = entry["index"]
            log_path = batch_dir(batches_root, index) / "docker_stdout.log"
            filename = find_stalled_signature_page(log_path)
            if filename is None:
                log_event(
                    model_base,
                    {
                        "kind": "STOP_UNCONFIRMED",
                        "summary": f"batch {index} stalled but its log does not match the known "
                        "hang signature; stopping for manual review",
                    },
                )
                return 1
            if confirm_and_register(
                model_base, filename, index, log_path, run_id=args.run_id, model=args.model
            ):
                progressed = True
            else:
                return 1

        if not progressed:
            log_event(model_base, {"kind": "STOP_NO_PROGRESS", "summary": "no batch could be auto-resolved"})
            return 1

    log_event(model_base, {"kind": "STOP_MAX_CYCLES", "summary": f"reached --max-cycles={args.max_cycles}"})
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
