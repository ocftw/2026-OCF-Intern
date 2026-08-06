#!/usr/bin/env python3
"""One-shot inference progress snapshot, meant to be wrapped with `watch`:

    watch -n 15 python3 tools/inference_status.py

Reads results.sqlite directly (read-only) -- safe to run alongside the live
inference process.
"""

from __future__ import annotations

import pathlib
import sqlite3
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from omnidocbench.core import Settings


def print_progress(settings: Settings) -> None:
    db_path = settings.output_dir / "results.sqlite"
    status_path = settings.output_dir / "status.json"

    print(f"run_id={settings.run_id}")
    if status_path.is_file():
        print(f"status={status_path.read_text(encoding='utf-8').strip()}")
    if not db_path.is_file():
        print("results.sqlite not created yet (still warming up)")
        return

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    run_rows = conn.execute("SELECT signature FROM run ORDER BY rowid DESC LIMIT 1").fetchall()
    if not run_rows:
        print("no run row yet")
        return
    signature = run_rows[0]["signature"]

    expected_total = settings.raw["benchmark"]["expected_pages"]

    for model in settings.models:
        model_id = model["id"]
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM page_result WHERE signature=? AND model_id=? GROUP BY status",
            (signature, model_id),
        ).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        done = sum(counts.values())
        success = counts.get("SUCCESS", 0)
        failed = counts.get("FAILED", 0)
        last = conn.execute(
            "SELECT filename, ended_at FROM page_result WHERE signature=? AND model_id=? "
            "ORDER BY rowid DESC LIMIT 1",
            (signature, model_id),
        ).fetchone()
        last_desc = f"{last['filename']} @ {last['ended_at']}" if last else "-"
        pct = (done / expected_total * 100) if expected_total else 0.0
        print(
            f"{model_id:<16} {done}/{expected_total} ({pct:5.1f}%)  "
            f"success={success} failed={failed}  last={last_desc}"
        )


def main() -> int:
    print_progress(Settings.load())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
