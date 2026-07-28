"""產生固定 15 列 summary、Markdown 與自包含 HTML。"""

from __future__ import annotations

import csv
import html
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

from .scorers import tc_str, vistw_mcq
from .statistics import (
    bootstrap_ci,
    paired_deltas,
    stratified_macro_bootstrap,
    stratified_paired_delta,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    latest = {}
    for row in rows:
        latest[str(row.get("sample_id"))] = row
    return list(latest.values())


def _latency(rows: list[dict[str, Any]]) -> tuple[float | None, float | None, float | None]:
    values = sorted(
        float(row["wall_clock_latency"])
        for row in rows
        if row.get("status") == "completed" and row.get("wall_clock_latency") is not None
    )
    if not values:
        return None, None, None
    p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
    return statistics.mean(values), statistics.median(values), p95


def build_summary(
    cfg: dict[str, Any], run_dir: Path, model_metadata: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for model in cfg["models"]:
        for benchmark in cfg["benchmarks"]:
            rows = _read_jsonl(run_dir / "predictions" / f"{model['id']}__{benchmark['id']}.jsonl")
            failures = sum(row.get("status") != "completed" for row in rows)
            truncated = sum(bool(row.get("truncated")) for row in rows)
            invalid = 0
            primary = None
            supporting: dict[str, Any] = {}
            ci = [None, None]
            successful = sum(row.get("status") == "completed" for row in rows)
            status = "completed" if successful else "failed"
            reason = (
                ""
                if successful
                else ("所有 prediction 均失敗" if rows else "沒有 prediction records")
            )
            if benchmark["id"] == "tc_str" and rows:
                supporting = tc_str.aggregate(rows)
                legacy_rows = [
                    {"metrics": row["legacy_metrics"]} for row in rows if row.get("legacy_metrics")
                ]
                if legacy_rows:
                    supporting["legacy_cleaned_supplement"] = tc_str.aggregate(legacy_rows)
                primary = supporting["exact_accuracy"]
                ci = list(bootstrap_ci([r["metrics"]["exact"] for r in rows]))
            elif benchmark["id"] == "vistw_mcq" and rows:
                supporting = vistw_mcq.aggregate(rows)
                primary = supporting["macro_accuracy"]
                invalid = sum(not r["metrics"]["valid"] for r in rows)
                subjects: dict[str, list[float]] = {}
                for row in rows:
                    subjects.setdefault(row["subject"], []).append(float(row["metrics"]["correct"]))
                ci = list(stratified_macro_bootstrap(subjects))
            elif benchmark["id"] == "omnidocbench":
                score_file = run_dir / "scores" / f"{model['id']}__omnidocbench.json"
                if score_file.exists():
                    official = json.loads(score_file.read_text(encoding="utf-8"))
                    status = official.get("status", "blocked")
                    reason = official.get("reason", "")
                    primary = official.get("overall")
                    supporting = official.get("official_metrics", {})
                    ci = official.get("ci95", [None, None])
                elif rows:
                    status, reason = "blocked", "官方 evaluator 尚未成功完成"
            mean_latency, median, p95 = _latency(rows)
            meta = model_metadata.get(model["id"], {})
            token_rates = [
                r["output_tokens_per_second"]
                for r in rows
                if r.get("output_tokens_per_second") is not None
            ]
            output.append(
                {
                    "model": model["id"],
                    "benchmark": benchmark["id"],
                    "status": status,
                    "reason": reason,
                    "primary_metric": primary,
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "supporting_metrics": supporting,
                    "n": len(rows),
                    "failed": failures,
                    "invalid": invalid,
                    "truncated": truncated,
                    "latency_mean": mean_latency,
                    "latency_median": median,
                    "latency_p95": p95,
                    "tokens_per_second": statistics.mean(token_rates) if token_rates else None,
                    "model_tag": model["tag"],
                    "model_digest": meta.get("digest", ""),
                    "quantization": meta.get("quantization", ""),
                    "config_hash": cfg["_effective_hash"],
                }
            )
    if len(output) != 15:
        raise AssertionError("summary 必須剛好 15 列")
    return output


def write_reports(summary: list[dict[str, Any]], run_dir: Path) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flat_rows = []
    for row in summary:
        flat = dict(row)
        flat["supporting_metrics"] = json.dumps(
            flat["supporting_metrics"], ensure_ascii=False, sort_keys=True
        )
        flat_rows.append(flat)
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    failures = []
    for path in sorted((run_dir / "predictions").glob("*.jsonl")):
        for row in _read_jsonl(path):
            if row.get("status") != "completed":
                failures.append(
                    {
                        "model": row.get("model_logical_id"),
                        "benchmark": row.get("benchmark"),
                        "sample_id": row.get("sample_id"),
                        "status": row.get("status"),
                        "error_type": row.get("error_type"),
                        "reason": row.get("error_message"),
                    }
                )
    existing = {(row["model"], row["benchmark"]) for row in failures}
    for row in summary:
        if row["status"] != "completed" and (row["model"], row["benchmark"]) not in existing:
            failures.append(
                {
                    "model": row["model"],
                    "benchmark": row["benchmark"],
                    "sample_id": "",
                    "status": row["status"],
                    "error_type": "",
                    "reason": row["reason"],
                }
            )
    with (run_dir / "failures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "benchmark", "sample_id", "status", "error_type", "reason"],
        )
        writer.writeheader()
        writer.writerows(failures)
    title = {
        "omnidocbench": "OmniDocBench Overall",
        "tc_str": "TC-STR Exact Accuracy",
        "vistw_mcq": "VisTW Macro Accuracy (deterministic parser)",
    }
    sections = []
    for benchmark, heading in title.items():
        rows = [row for row in summary if row["benchmark"] == benchmark]
        body = "\n".join(
            f"| {r['model']} | {r['status']} | {r['primary_metric']} | "
            f"[{r['ci95_low']}, {r['ci95_high']}] | {r['n']} | {r['failed']} | "
            f"{r['invalid']} | {r['truncated']} |"
            for r in rows
        )
        sections.append(
            f"## {heading}\n\n| Model | Status | Primary | 95% CI | N | Failed | "
            f"Invalid | Truncated |\n|---|---|---:|---|---:|---:|---:|---:|\n{body}"
        )
    markdown = (
        "# Benchmark Results\n\n三個 benchmark 分開排名；未計算跨任務 raw-score 平均。\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    (run_dir / "RESULTS.md").write_text(markdown, encoding="utf-8")
    escaped = html.escape(markdown)
    data = html.escape(json.dumps(summary, ensure_ascii=False))
    document = f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>OCF VLM Benchmark</title><style>
body{{font:16px/1.55 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}
pre{{white-space:pre-wrap;background:#f6f8fa;padding:1rem;border-radius:8px}}
</style><h1>OCF VLM Benchmark</h1><pre>{escaped}</pre>
<details><summary>Machine-readable summary</summary><pre>{data}</pre></details></html>"""
    (run_dir / "report.html").write_text(document, encoding="utf-8")


def write_pairwise(cfg: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """每個 benchmark 10 組；Omni 無 page metric 時明確 N/A。"""
    model_ids = [model["id"] for model in cfg["models"]]
    results: list[dict[str, Any]] = []
    tc_by_model = {}
    vistw_by_model = {}
    for model in model_ids:
        tc_rows = _read_jsonl(run_dir / "predictions" / f"{model}__tc_str.jsonl")
        tc_by_model[model] = {
            str(row["sample_id"]): float(row["metrics"]["exact"]) for row in tc_rows
        }
        vistw_rows = _read_jsonl(run_dir / "predictions" / f"{model}__vistw_mcq.jsonl")
        nested: dict[str, dict[str, float]] = {}
        for row in vistw_rows:
            nested.setdefault(str(row["subject"]), {})[str(row["sample_id"])] = float(
                row["metrics"]["correct"]
            )
        vistw_by_model[model] = nested
    for row in paired_deltas(
        tc_by_model, iterations=cfg["bootstrap_iterations"], seed=cfg["bootstrap_seed"]
    ):
        results.append({"benchmark": "tc_str", **row})
    for left, right in itertools.combinations(model_ids, 2):
        results.append(
            {
                "benchmark": "vistw_mcq",
                "model_a": left,
                "model_b": right,
                **stratified_paired_delta(
                    vistw_by_model[left],
                    vistw_by_model[right],
                    cfg["bootstrap_iterations"],
                    cfg["bootstrap_seed"],
                ),
            }
        )
        results.append(
            {
                "benchmark": "omnidocbench",
                "model_a": left,
                "model_b": right,
                "status": "N/A",
                "reason": "官方 evaluator 未保證提供可對齊的 page-level Overall components",
            }
        )
    scores = run_dir / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    (scores / "pairwise_comparisons.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results
