#!/usr/bin/env python3
"""Record a ground-truth page as a known official-evaluator hang.

This never happens automatically. A stall must first be observed (see
evaluate_model.py's stall detection, or a manual investigation) with concrete
evidence -- a log excerpt, a stall_report.json path, or similar -- before a
page is added here. Entries recorded here are only ever *used* when a run is
explicitly retried with --eval-timeout-as-empty; by default the evaluator
still uses the real prediction.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys


DEFAULT_OUTPUT_ROOT = pathlib.Path("/opt/ocf-ai/outputs/omnidocbench_v1_6")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {"schema_version": 1, "pages": []}
    return read_json(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record (or update) a ground-truth page filename as a known "
            "official-evaluator hang in evaluation_timeout_pages.json for one model."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--filename",
        required=True,
        help="ground-truth image filename exactly as it appears in OmniDocBench.json "
        "(e.g. page-d9a27df2-127e-4f76-bb61-e7e741078bef.png)",
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="short human-readable reason, e.g. 'degenerate repeated-text prediction; "
        "quick_match/timeout-fallback hang, confirmed 2026-07-31'",
    )
    parser.add_argument(
        "--evidence",
        required=True,
        help="path to a stall_report.json / docker_stdout.log / other artifact backing "
        "this call, so the entry is auditable later",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="remove the entry for --filename instead of adding/updating it",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_base = pathlib.Path(args.output_root) / args.run_id / "models" / args.model
    if not model_base.is_dir():
        print(f"ERROR: model directory does not exist: {model_base}", file=sys.stderr)
        return 1
    path = model_base / "evaluation_timeout_pages.json"
    data = load(path)
    pages = [entry for entry in data.get("pages", []) if entry.get("filename") != args.filename]

    if args.remove:
        removed = len(data.get("pages", [])) != len(pages)
        data["pages"] = pages
        write_json(path, data)
        print(f"{'removed' if removed else 'no matching entry for'} {args.filename} in {path}")
        return 0

    pages.append(
        {
            "filename": args.filename,
            "reason": args.reason,
            "evidence": args.evidence,
            "recorded_at": utc_now(),
        }
    )
    data["pages"] = pages
    write_json(path, data)
    print(f"recorded {args.filename} in {path}")
    print("this page is only substituted with an empty prediction when the next run "
          "passes --eval-timeout-as-empty; it is otherwise unaffected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
