from __future__ import annotations

import csv
import html
import json
import os
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
from datetime import datetime

from .config import Settings
from .metrics import aggregate, aligned_supplementary, anomaly_flags, normalize_primary, score
from .storage import Checkpoint
from .util import atomic_write_json, atomic_write_text


def _taipei(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(ZoneInfo("Asia/Taipei")).isoformat()
    except ValueError:
        return iso


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _summary(rows: list[dict[str, Any]], models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model_id"]].append(row)
    summary = []
    for model in models:
        values = grouped[model["id"]]
        clean = [row for row in values if row["record_status"] == "completed"]
        anomalous = [row for row in values if row["record_status"] == "completed_with_anomaly"]
        output_failures = [row for row in values if row["record_status"] == "model_output_failure"]
        request_failures = [row for row in values if row["record_status"] == "request_failure"]
        latencies = [
            float(row["latency_seconds"])
            for row in values
            if row.get("http_status") == 200
        ]
        flags = [row["anomaly_flags"] for row in values]
        summary.append(
            {
                "model_id": model["id"],
                "logical_name": model["logical_name"],
                "exact_tag": model.get("exact_tag"),
                "digest": values[0]["model_digest"] if values else "",
                "scored_samples": len(values),
                "successful": len(clean) + len(anomalous),
                "clean_completed": len(clean),
                "completed_with_anomaly": len(anomalous),
                "model_output_failures": len(output_failures),
                "request_failures": len(request_failures),
                "failed_or_incomplete": len(output_failures) + len(request_failures),
                "empty": sum(flag["empty"] or flag["control_only"] for flag in flags),
                "truncated": sum(flag["truncated"] for flag in flags),
                "truncation_unknown": sum(flag.get("truncation_unknown", False) for flag in flags),
                "completion_policy_exceptions": sum(
                    bool(row.get("completion_validation", {}).get("accepted_by_exception"))
                    for row in values
                ),
                "pipeline_anomalies": sum(
                    any(flag[name] for name in ("prompt_echo", "refusal", "formatting", "repetition", "thinking", "unrelated"))
                    for flag in flags
                ),
                "primary": aggregate([row["primary_metrics"] for row in values]),
                "aligned_supplementary": aggregate([row["aligned_metrics"] for row in values]),
                "latency_mean": statistics.mean(latencies) if latencies else None,
                "latency_median": statistics.median(latencies) if latencies else None,
                "latency_p95": _percentile(latencies, 0.95),
                "retries": sum(max(0, int(row["attempt_count"]) - 1) for row in values),
            }
        )
    return summary


def _combined_rows(
    settings: Settings,
    checkpoint: Checkpoint,
    signature: str,
    phase: str,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    results = checkpoint.results(signature, phase)
    for row in results:
        row.setdefault("record_status", "completed")
    result_keys = {(row["model_id"], int(row["sample_index"])) for row in results}
    samples = {int(sample["index"]): sample for sample in manifest["samples"]}
    model_map = {model["id"]: model for model in settings.models}
    for attempt in checkpoint.latest_attempts(signature, phase):
        key = (attempt["model_id"], int(attempt["sample_index"]))
        if key in result_keys:
            continue
        sample = samples[key[1]]
        response = attempt.get("response") or {}
        raw = str(response.get("response") or "")
        primary = normalize_primary(raw)
        aligned, changes = aligned_supplementary(raw)
        flags = anomaly_flags(
            raw,
            settings.prompt,
            response.get("done_reason"),
            response.get("eval_count"),
            int(settings.options["num_predict"]),
        )
        completion_validation = attempt.get("completion_validation") or {}
        flags["truncation_unknown"] = not bool(
            completion_validation.get("truncation_observable", True)
        )
        results.append(
            {
                "run_signature": signature,
                "phase": phase,
                "model_id": key[0],
                "sample_index": key[1],
                "exact_tag": model_map[key[0]].get("exact_tag") or "",
                "model_digest": "",
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
                    name: value
                    for name, value in response.items()
                    if name not in {"response", "context"}
                },
                "done_reason": response.get("done_reason"),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "completion_validation": completion_validation,
                "latency_seconds": attempt["latency_seconds"],
                "attempt_count": attempt["attempt_number"],
                "completed_at": attempt["ended_at"],
                "record_status": (
                    "request_failure"
                    if attempt.get("http_status") != 200 or attempt.get("error_type")
                    else "model_output_failure"
                ),
                "error_type": attempt.get("error_type"),
                "error_message": attempt.get("error_message"),
                "http_status": attempt.get("http_status"),
            }
        )
    return sorted(results, key=lambda row: (int(row["sample_index"]), row["model_id"]))


def _copy_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def export_results(
    settings: Settings,
    run_dir: Path,
    signature: str,
    phase: str,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint = Checkpoint(run_dir / "results.sqlite")
    try:
        rows = _combined_rows(settings, checkpoint, signature, phase, manifest)
    finally:
        checkpoint.close()
    summaries = _summary(rows, settings.models)
    atomic_write_json(run_dir / "summary.json", {"phase": phase, "models": summaries})
    long_fields = [
        "sample_index", "image_relative_path", "image_sha256", "ground_truth", "model_id",
        "record_status", "http_status", "error_type", "error_message",
        "exact_tag", "model_digest", "prediction_raw", "prediction_primary",
        "prediction_aligned", "primary_metrics", "aligned_metrics", "anomaly_flags",
        "latency_seconds", "done_reason", "prompt_eval_count", "eval_count",
        "completion_validation", "attempt_count",
    ]
    long_path = run_dir / "predictions_long.csv"
    temporary = long_path.with_name(f".{long_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), (dict, list)) else row.get(key)
                    for key in long_fields
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, long_path)
    jsonl_path = run_dir / "predictions.jsonl"
    atomic_write_text(
        jsonl_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )
    by_index: dict[int, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        by_index[int(row["sample_index"])][row["model_id"]] = row
    wide_fields = ["sample_index", "image_relative_path", "image_sha256", "ground_truth"] + [
        f"{model['id']}__prediction" for model in settings.models
    ]
    wide_path = run_dir / "predictions_wide.csv"
    temporary = wide_path.with_name(f".{wide_path.name}.tmp")
    samples = {int(sample["index"]): sample for sample in manifest["samples"]}
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fields)
        writer.writeheader()
        for index in sorted(by_index):
            sample = samples[index]
            output = {
                "sample_index": index,
                "image_relative_path": sample["image_relative_path"],
                "image_sha256": sample["image_sha256"],
                "ground_truth": sample["ground_truth"],
            }
            for model in settings.models:
                result = by_index[index].get(model["id"])
                output[f"{model['id']}__prediction"] = result["prediction_primary"] if result else ""
            writer.writerow(output)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, wide_path)
    return rows, summaries


def build_main_report(
    settings: Settings,
    run_dir: Path,
    signature: str,
    phase: str,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    rows, summaries = export_results(settings, run_dir, signature, phase, manifest)
    metadata_models = {model.get("id"): model for model in metadata.get("models", [])}
    for item in summaries:
        model_metadata = metadata_models.get(item["model_id"], {})
        item["quantization"] = model_metadata.get("quantization") or "—"
        item["parameter_count"] = model_metadata.get("parameter_count") or "—"
        item["processor"] = model_metadata.get("processor") or "—"
        if not item["digest"]:
            item["digest"] = model_metadata.get("digest") or ""
    by_index: dict[int, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        by_index[int(row["sample_index"])][row["model_id"]] = row
    sample_map = {int(sample["index"]): sample for sample in manifest["samples"]}
    for index in by_index:
        sample = sample_map[index]
        suffix = Path(sample["image_relative_path"]).suffix.lower()
        _copy_asset(settings.dataset_dir / sample["image_relative_path"], run_dir / "assets" / f"{index:04d}{suffix}")
    summary_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['logical_name']))}</td><td><code>{html.escape(str(item['exact_tag']))}</code></td>"
        f"<td>{html.escape(str(item['digest']))}</td>"
        f"<td>{html.escape(str(item['quantization']))}</td><td>{html.escape(str(item['parameter_count']))}</td>"
        f"<td>{html.escape(str(item['processor']))}</td>"
        + "".join(f"<td>{item['primary'][metric]:.4f}</td>" for metric in ("em", "cm", "anls", "f1"))
        + "".join(f"<td>{item['aligned_supplementary'][metric]:.4f}</td>" for metric in ("em", "cm", "anls", "f1"))
        + f"<td>{item['successful']}</td><td>{item['failed_or_incomplete']}</td>"
        f"<td>{item['empty']}</td><td>{item['truncated']}</td>"
        f"<td>{item['truncation_unknown']}</td><td>{item['completion_policy_exceptions']}</td>"
        f"<td>{item['latency_mean'] if item['latency_mean'] is not None else '—'}</td>"
        f"<td>{item['latency_median'] if item['latency_median'] is not None else '—'}</td>"
        f"<td>{item['latency_p95'] if item['latency_p95'] is not None else '—'}</td></tr>"
        for item in summaries
    )
    detail_rows = []
    for index in sorted(by_index):
        sample = sample_map[index]
        suffix = Path(sample["image_relative_path"]).suffix.lower()
        cards = []
        searchable = [sample["ground_truth"], sample["image_relative_path"]]
        for model in settings.models:
            row = by_index[index].get(model["id"])
            if not row:
                cards.append(
                    f"<article class='model missing' data-model='{model['id']}' data-em='' data-flags='missing'>"
                    f"<h4>{html.escape(model['logical_name'])}</h4><p>無成功結果</p></article>"
                )
                continue
            flags = [name for name, active in row["anomaly_flags"].items() if active]
            searchable.extend([row["prediction_raw"], *flags])
            raw = html.escape(row["prediction_raw"])
            primary = html.escape(row["prediction_primary"])
            aligned = html.escape(row["prediction_aligned"])
            metrics = row["primary_metrics"]
            all_flags = flags + ([str(row["error_type"])] if row.get("error_type") else [])
            cards.append(
                f"<article class='model {html.escape(row['record_status'])}' data-model='{model['id']}' "
                f"data-em='{metrics['em']}' data-flags='{html.escape(' '.join(all_flags))}'>"
                f"<h4>{html.escape(model['logical_name'])}</h4>"
                f"<p><b>Status:</b> {html.escape(row['record_status'])} · "
                f"HTTP {html.escape(str(row.get('http_status')))} · "
                f"error {html.escape(str(row.get('error_type') or 'none'))}: "
                f"{html.escape(str(row.get('error_message') or ''))}</p>"
                f"<p><b>Primary:</b> <span class='answer'>{primary}</span></p>"
                f"<p><b>Aligned supplement:</b> {aligned}</p>"
                f"<p>EM {metrics['em']:.3f} · CM {metrics['cm']:.3f} · ANLS {metrics['anls']:.3f} · F1 {metrics['f1']:.3f}</p>"
                f"<p>latency {row['latency_seconds']:.3f}s · done {html.escape(str(row['done_reason']))} · "
                f"tokens {row['prompt_eval_count']}/{row['eval_count']} · attempt {row['attempt_count']}</p>"
                f"<p><b>Completion validation:</b> "
                f"{html.escape(json.dumps(row.get('completion_validation') or {}, ensure_ascii=False, sort_keys=True))}</p>"
                f"<p class='flags'>{html.escape(', '.join(flags) or 'none')}</p>"
                f"<details><summary>Raw response</summary><pre>{raw}</pre></details></article>"
            )
        searchable_text = html.escape(" ".join(searchable).lower(), quote=True)
        detail_rows.append(
            f"<section class='sample' data-index='{index}' "
            f"data-gt='{html.escape(sample['ground_truth'], quote=True)}' "
            f"data-search='{searchable_text}'><header><h3>#{index} · "
            f"{html.escape(sample['image_relative_path'])}</h3><p>SHA-256 <code>{sample['image_sha256']}</code> · "
            f"{sample['width']}×{sample['height']}</p></header>"
            f"<div class='sample-head'><img loading='lazy' src='assets/{index:04d}{suffix}' alt='TC-STR sample {index}'>"
            f"<p><b>Ground truth:</b> <span class='ground-truth'>{html.escape(sample['ground_truth'])}</span></p></div>"
            f"<div class='cards'>{''.join(cards)}</div></section>"
        )
    warning = (
        "Aligned supplementary 可能移除包裝、壓縮連續四次以上重複字元並截斷至 200 字；"
        "不得與 primary 主榜混稱。四項指標是本研究統一補充指標，不是 TC-STR 官方唯一 leaderboard protocol。"
        " 依使用者核准的共同條件政策，空輸出、重複、prompt echo、拒答、格式失控、"
        "thinking 污染與 token-limit 等輸出異常均保留並納入該模型分數，不再作為整套評測的硬阻擋；"
        "prompt、原圖與 generation options 對所有模型維持一致。"
        " GLM OCR BF16 經使用者核准採 completion metadata 例外：HTTP 200 且 response 非空即可評分；"
        "缺少的 done_reason/prompt_eval_count/eval_count 保留 null，因此其 token 使用與自動截斷診斷不可和其他模型等量比較。"
    )
    disagreements = []
    for index in sorted(by_index):
        completed = [
            row for row in by_index[index].values() if row["record_status"] == "completed"
        ]
        predictions = {row["prediction_primary"] for row in completed}
        if len(predictions) > 1:
            mean_anls = statistics.mean(row["primary_metrics"]["anls"] for row in completed)
            disagreements.append((mean_anls, index, completed))
    representative_rows = []
    for mean_anls, index, completed in sorted(disagreements)[:10]:
        answers = " · ".join(
            f"{row['model_id']}={row['prediction_primary']}" for row in completed
        )
        representative_rows.append(
            f"<li>#{index} · GT={html.escape(sample_map[index]['ground_truth'])} · "
            f"mean ANLS={mean_anls:.3f} · {html.escape(answers)}</li>"
        )
    descriptive_summary = (
        f"<p>目前有 {len(disagreements)} 題在至少兩個已完成模型間產生不同 primary prediction。"
        "下列最多 10 題依已完成模型的平均 ANLS 由低到高列出，只作描述，不取代完整逐題資料。</p>"
        f"<ol>{''.join(representative_rows) or '<li>目前沒有足夠的跨模型完成結果可比較。</li>'}</ol>"
    )
    document = f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>TC-STR × 8 Models</title><style>
body{{font:14px/1.5 system-ui;margin:1.5rem;color:#18202a}}code,pre{{font-family:ui-monospace,monospace}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ccd2d8;padding:.35rem;vertical-align:top}}
.warning{{background:#fff3cd;border-left:4px solid #b77900;padding:1rem}}.sample{{border-top:2px solid #506070;padding:1rem 0}}
.sample-head{{display:flex;gap:1rem;align-items:center}}img{{max-width:260px;max-height:120px;object-fit:contain;background:#eee}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:.6rem}}article{{border:1px solid #ccd2d8;padding:.6rem;min-width:0}}
.answer,.ground-truth{{font-size:1.15em}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:22rem;overflow:auto}}
.flags{{color:#9b2c2c}}.missing{{background:#fee}}.hidden{{display:none}}.toolbar{{position:sticky;top:0;background:white;padding:.7rem;border:1px solid #ccc;z-index:2}}
</style><h1>TC-STR × 8 Models — {html.escape(str(metadata.get('status', phase)))}</h1>
<p>Run <code>{html.escape(str(metadata.get('run_id', run_dir.name)))}</code> · signature <code>{html.escape(signature)}</code><br>
UTC: {html.escape(str(metadata.get('started_at', '—')))} → {html.escape(str(metadata.get('ended_at', '—')))}<br>
Asia/Taipei: {html.escape(_taipei(metadata.get('started_at')))} → {html.escape(_taipei(metadata.get('ended_at')))}</p>
<p class="warning">{html.escape(warning)}</p>
<h2>代表性錯例與模型差異（描述性）</h2>{descriptive_summary}
<h2>Primary leaderboard 與 aligned supplementary</h2>
<table><thead><tr><th>Model</th><th>Exact tag</th><th>Digest</th><th>Quantization</th><th>Parameters</th><th>Processor</th>
<th>EM</th><th>CM</th><th>ANLS</th><th>F1</th>
<th>Aligned EM</th><th>Aligned CM</th><th>Aligned ANLS</th><th>Aligned F1</th>
<th>Success</th><th>Failed/incomplete</th><th>Empty</th><th>Truncated</th>
<th>Truncation unknown</th><th>Completion exceptions</th>
<th>Mean s</th><th>Median s</th><th>P95 s</th></tr></thead><tbody>{summary_rows}</tbody></table>
<h2>共同設定</h2><pre>{html.escape(json.dumps({'prompt': settings.prompt, 'options': settings.options}, ensure_ascii=False, indent=2))}</pre>
<div class="toolbar"><label>搜尋 <input id="q"></label> <label>Model <select id="model"><option value="">全部</option>
{''.join(f"<option value='{m['id']}'>{html.escape(m['logical_name'])}</option>" for m in settings.models)}</select></label>
<label>結果 <select id="correct"><option value="">全部</option><option value="1">EM 正確</option><option value="0">EM 錯誤</option></select></label>
<label>異常 <select id="anomaly"><option value="">全部</option><option>empty</option><option>control_only</option>
<option>prompt_echo</option><option>refusal</option><option>formatting</option><option>repetition</option>
<option>thinking</option><option>unrelated</option><option>truncated</option><option>truncation_unknown</option><option>timeout</option>
<option>transport</option><option>http_terminal</option><option>http_retryable</option><option>api_error</option></select></label>
<label>排序 <select id="sort"><option value="index-asc">Index ↑</option><option value="index-desc">Index ↓</option>
<option value="gt">Ground truth</option></select></label></div>
<div id="details">{''.join(detail_rows) or '<p>目前沒有 prediction。</p>'}</div>
<script>
const q=document.querySelector('#q'), model=document.querySelector('#model'), correct=document.querySelector('#correct'),
 anomaly=document.querySelector('#anomaly'), sort=document.querySelector('#sort'), details=document.querySelector('#details');
function filter(){{document.querySelectorAll('.sample').forEach(s=>{{
 const text=s.dataset.search, query=q.value.toLowerCase(); let visible=!query||text.includes(query);
 s.querySelectorAll('article.model').forEach(c=>{{const show=(!model.value||c.dataset.model===model.value)&&
  (!correct.value||c.dataset.em===correct.value)&&(!anomaly.value||c.dataset.flags.split(' ').includes(anomaly.value));
  c.classList.toggle('hidden',!show)}});
 if(model.value||correct.value||anomaly.value) visible=visible&&[...s.querySelectorAll('article.model')].some(c=>!c.classList.contains('hidden'));
 s.classList.toggle('hidden',!visible);
}});
 const rows=[...document.querySelectorAll('.sample')];
 rows.sort((a,b)=>sort.value==='index-desc'?+b.dataset.index-+a.dataset.index:
  sort.value==='gt'?a.dataset.gt.localeCompare(b.dataset.gt,'zh-Hant'):+a.dataset.index-+b.dataset.index);
 rows.forEach(row=>details.appendChild(row));
}}; q.oninput=filter;model.onchange=filter;correct.onchange=filter;anomaly.onchange=filter;sort.onchange=filter;
</script></html>"""
    atomic_write_text(run_dir / "report.html", document)


def write_smoke_report(
    settings: Settings,
    run_dir: Path,
    preflight: dict[str, Any],
    manifest: dict[str, Any],
    status: str,
    blockers: list[str],
    manual_review: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    signature = preflight["run_signature"]
    checkpoint = Checkpoint(run_dir / "results.sqlite")
    try:
        rows = _combined_rows(settings, checkpoint, signature, "smoke", manifest)
    finally:
        checkpoint.close()
    summaries = _summary(rows, settings.models)
    report = {
        "schema_version": 1,
        "run_id": preflight["run_id"],
        "status": status,
        "run_signature": signature,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "smoke_sample_indices": manifest["smoke_selection"]["sample_indices"],
        "models": summaries,
        "results": rows,
        "blockers": blockers,
        "warnings": list(dict.fromkeys(preflight.get("warnings", []) + (warnings or []))),
        "evaluation_policy": settings.raw["evaluation_policy"],
        "protocol_exceptions": [
            {
                "model_id": model["id"],
                "exact_tag": model.get("exact_tag"),
                "completion_policy": model.get("completion_policy"),
            }
            for model in settings.models
            if (model.get("completion_policy") or {}).get("mode") != "strict"
        ],
        "manual_review": manual_review,
    }
    atomic_write_json(run_dir / "smoke_report.json", report)
    rows_html = "".join(
        f"<tr><td>{html.escape(item['logical_name'])}</td><td>{item['scored_samples']}/20</td>"
        f"<td>{item['successful']}</td>"
        f"<td>{item['failed_or_incomplete']}</td><td>{item['empty']}</td>"
        f"<td>{item['pipeline_anomalies']}</td><td>{item['truncated']}</td>"
        f"<td>{item['truncation_unknown']}</td><td>{item['completion_policy_exceptions']}</td>"
        f"<td>{item['primary']['em']:.4f}</td><td>{item['primary']['cm']:.4f}</td>"
        f"<td>{item['primary']['anls']:.4f}</td><td>{item['primary']['f1']:.4f}</td></tr>"
        for item in summaries
    )
    blocker_html = "".join(f"<li>{html.escape(item)}</li>" for item in blockers) or "<li>無</li>"
    warning_html = "".join(
        f"<li>{html.escape(item)}</li>" for item in report["warnings"]
    ) or "<li>無</li>"
    raw = html.escape(json.dumps(report, ensure_ascii=False, indent=2))
    document = f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>TC-STR Smoke</title>
<style>body{{font:15px/1.5 system-ui;max-width:1200px;margin:2rem auto}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:.5rem}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:1rem}}</style>
<h1>TC-STR 全模型 Smoke：{html.escape(status)}</h1><p>Signature <code>{signature}</code></p>
<h2>Blockers</h2><ul>{blocker_html}</ul><h2>Warnings / protocol exceptions</h2>
<ul>{warning_html}</ul><pre>{html.escape(json.dumps(report['protocol_exceptions'], ensure_ascii=False, indent=2))}</pre>
<p>輸出異常依核准政策視為模型在共同使用條件下的實際失敗，保留並納入 20 筆分母；
共同 prompt 與 generation options 不因模型變更。</p>
<table><thead><tr><th>Model</th><th>Scored</th><th>Nonempty response</th>
<th>Failed/incomplete</th><th>Empty</th><th>Pipeline anomalies</th><th>Truncated</th>
<th>Truncation unknown</th><th>Completion exceptions</th>
<th>EM</th><th>CM</th><th>ANLS</th><th>F1</th>
</tr></thead><tbody>{rows_html}</tbody></table><details><summary>完整逐題資料</summary><pre>{raw}</pre></details></html>"""
    atomic_write_text(run_dir / "smoke_report.html", document)
    return report
