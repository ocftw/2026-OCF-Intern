#!/usr/bin/env python3
"""Run paired, one-factor-at-a-time OCR ablations and build an HTML report."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import os
import re
import statistics
import string
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE
RUNTIME_OUTPUT_DIR = HERE / "output"
sys.path.insert(0, str(PROJECT_DIR))

from metrics import score_all as aligned_score_all  # noqa: E402
from postprocess import clean_prediction as aligned_postprocess  # noqa: E402
import config as aligned_config  # noqa: E402


SHORT_PROMPT = "請辨識圖片中的文字。只輸出辨識出的文字，不要有任何其他贅字、標點、說明或格式。"
BASE_OPTIONS = {"temperature": 0, "repeat_penalty": 1.6, "num_predict": 80}

INFERENCE_VARIANTS = {
    "baseline": {
        "label": "Aligned baseline",
        "description": "合作者完整設定：/api/generate、長 prompt、repeat_penalty=1.6、num_predict=80。",
        "endpoint": "generate",
        "prompt": aligned_config.OCR_PROMPT,
        "options": BASE_OPTIONS,
    },
    "endpoint_chat": {
        "label": "Endpoint: /api/chat",
        "description": "只把 /api/generate 改成 /api/chat；prompt 與 generation options 不變。",
        "endpoint": "chat",
        "prompt": aligned_config.OCR_PROMPT,
        "options": BASE_OPTIONS,
    },
    "short_prompt": {
        "label": "Prompt: Sixhuang short",
        "description": "只把 171 字 aligned prompt 改成 Sixhuang 39 字短 prompt。",
        "endpoint": "generate",
        "prompt": SHORT_PROMPT,
        "options": BASE_OPTIONS,
    },
    "no_repeat_penalty": {
        "label": "No repeat_penalty",
        "description": "只移除 repeat_penalty=1.6；保留 temperature=0、num_predict=80。",
        "endpoint": "generate",
        "prompt": aligned_config.OCR_PROMPT,
        "options": {"temperature": 0, "num_predict": 80},
    },
    "num_predict_25": {
        "label": "num_predict: 25",
        "description": "只把 num_predict=80 改成 25；其他設定不變。",
        "endpoint": "generate",
        "prompt": aligned_config.OCR_PROMPT,
        "options": {"temperature": 0, "repeat_penalty": 1.6, "num_predict": 25},
    },
}

DERIVED_VARIANTS = {
    "minimal_postprocess": {
        "label": "Minimal postprocess",
        "description": "不重跑模型；只把 baseline raw response 改用 Sixhuang 的最小後處理。",
    },
    "sixhuang_scorer": {
        "label": "Sixhuang scorer",
        "description": "不重跑模型；保留 baseline processed prediction，只改用 Sixhuang scorer。",
    },
}

FIELDNAMES = [
    "index", "image_filename", "image_sha256", "ground_truth", "variant",
    "model_tag", "endpoint", "prompt_sha256", "options_json",
    "prediction_raw", "prediction", "em", "cm", "anls", "f1",
    "latency_sec", "done_reason", "prompt_eval_count", "eval_count", "error",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_samples(labels_path: Path, images_dir: Path, limit: int) -> list[dict]:
    samples = []
    with labels_path.open("r", encoding="utf-8") as f:
        for source_index, line in enumerate(f):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid label line {source_index + 1}: {line!r}")
            rel_path, ground_truth = parts
            filename = Path(rel_path).name
            image_path = images_dir / filename
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image: {image_path}")
            image_bytes = image_path.read_bytes()
            samples.append({
                "index": len(samples),
                "image_filename": filename,
                "image_path": image_path,
                "image_bytes": image_bytes,
                "image_sha256": sha256_bytes(image_bytes),
                "ground_truth": ground_truth.strip(),
            })
            if limit > 0 and len(samples) >= limit:
                break
    return samples


def minimal_postprocess(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        if len(lines) > 2 and lines[0].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        text = text[1:-1].strip()
    return text


def sixhuang_clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    zh_punc = "，。！？；：（）［］【】「」『』〈〉《》——……——、•·「」『』"
    return text.translate(str.maketrans("", "", zh_punc))


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def sixhuang_score_all(pred: str, gt: str) -> dict[str, float]:
    p, g = sixhuang_clean_text(pred), sixhuang_clean_text(gt)
    em = 1.0 if p == g else 0.0
    cm = 0.0 if not p or not g else (1.0 if g in p or p in g else 0.0)
    if not p and not g:
        anls = 1.0
    elif not p or not g:
        anls = 0.0
    else:
        nls = 1.0 - edit_distance(p, g) / max(len(p), len(g))
        anls = nls if nls >= 0.5 else 0.0
    if not p and not g:
        f1 = 1.0
    elif not p or not g:
        f1 = 0.0
    else:
        common = sum((Counter(p) & Counter(g)).values())
        precision, recall = common / len(p), common / len(g)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"em": em, "cm": cm, "anls": anls, "f1": f1}


def request_ollama(
    host: str,
    model: str,
    variant: dict,
    image_bytes: bytes,
    timeout: int,
) -> tuple[str, dict, float]:
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    endpoint = variant["endpoint"]
    common = {
        "model": model,
        "images": [image_b64],
        "options": variant["options"],
        "stream": False,
    }
    if endpoint == "generate":
        payload = {**common, "prompt": variant["prompt"]}
        url = f"{host.rstrip('/')}/api/generate"
    elif endpoint == "chat":
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": variant["prompt"],
                "images": [image_b64],
            }],
            "options": variant["options"],
            "stream": False,
        }
        url = f"{host.rstrip('/')}/api/chat"
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")

    started = time.monotonic()
    response = requests.post(url, json=payload, timeout=timeout)
    latency = time.monotonic() - started
    response.raise_for_status()
    data = response.json()
    if endpoint == "generate":
        text = data.get("response", "")
    else:
        text = (data.get("message") or {}).get("content", "")
    meta = {
        "done_reason": data.get("done_reason", ""),
        "prompt_eval_count": data.get("prompt_eval_count", ""),
        "eval_count": data.get("eval_count", ""),
    }
    return text, meta, latency


def read_csv_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    # Interrupted/resumed runs may contain an old error followed by a successful
    # retry for the same sample. Reports always use the latest attempt.
    latest = {}
    for row in rows:
        try:
            latest[int(row["index"])] = row
        except (KeyError, ValueError):
            continue
    return [latest[idx] for idx in sorted(latest)]


def completed_indexes(path: Path) -> set[int]:
    done = set()
    for row in read_csv_records(path):
        try:
            if not row.get("error"):
                done.add(int(row["index"]))
        except (KeyError, ValueError):
            continue
    return done


def write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def run_inference_variant(
    key: str,
    variant: dict,
    samples: list[dict],
    args: argparse.Namespace,
    output_dir: Path,
    status_path: Path,
) -> None:
    out_path = output_dir / f"{key}.csv"
    done = completed_indexes(out_path)
    print(f"\n=== {key}: {variant['label']} ({len(done)}/{len(samples)} already complete) ===", flush=True)
    prompt_hash = sha256_text(variant["prompt"])
    options_json = json.dumps(variant["options"], ensure_ascii=False, sort_keys=True)

    for sample in samples:
        idx = sample["index"]
        if idx in done:
            continue
        prediction_raw, meta, latency, error = "", {}, 0.0, ""
        for attempt in range(1, args.max_retries + 1):
            try:
                prediction_raw, meta, latency = request_ollama(
                    args.host, args.model, variant, sample["image_bytes"], args.timeout
                )
                error = ""
                break
            except Exception as exc:  # keep overnight run moving and preserve the error
                error = f"{type(exc).__name__}: {exc}"
                print(f"[{key}] {idx + 1}/{len(samples)} attempt {attempt}/{args.max_retries} failed: {error}", flush=True)
                if attempt < args.max_retries:
                    time.sleep(args.retry_delay)

        prediction = aligned_postprocess(prediction_raw) if not error else ""
        scores = aligned_score_all(prediction, sample["ground_truth"])
        row = {
            "index": idx,
            "image_filename": sample["image_filename"],
            "image_sha256": sample["image_sha256"],
            "ground_truth": sample["ground_truth"],
            "variant": key,
            "model_tag": args.model,
            "endpoint": variant["endpoint"],
            "prompt_sha256": prompt_hash,
            "options_json": options_json,
            "prediction_raw": prediction_raw,
            "prediction": prediction,
            "em": scores["em"],
            "cm": scores["cm"],
            "anls": scores["anls"],
            "f1": scores["f1"],
            "latency_sec": round(latency, 6),
            "done_reason": meta.get("done_reason", ""),
            "prompt_eval_count": meta.get("prompt_eval_count", ""),
            "eval_count": meta.get("eval_count", ""),
            "error": error,
        }
        write_row(out_path, row)
        done.add(idx)
        if (idx + 1) % args.progress_every == 0 or idx + 1 == len(samples):
            print(f"[{key}] progress {idx + 1}/{len(samples)}", flush=True)
            atomic_json(status_path, {
                "state": "running",
                "updated_at": utc_now(),
                "variant": key,
                "completed": len(done),
                "total": len(samples),
            })


def write_derived_variants(output_dir: Path) -> None:
    baseline_path = output_dir / "baseline.csv"
    baseline = read_csv_records(baseline_path)
    if not baseline:
        raise RuntimeError("Cannot derive offline ablations without baseline.csv")
    for key in DERIVED_VARIANTS:
        out_path = output_dir / f"{key}.csv"
        if out_path.exists():
            out_path.unlink()
        for source in baseline:
            row = dict(source)
            row["variant"] = key
            if key == "minimal_postprocess":
                row["prediction"] = minimal_postprocess(source["prediction_raw"])
                scores = aligned_score_all(row["prediction"], row["ground_truth"])
            elif key == "sixhuang_scorer":
                scores = sixhuang_score_all(row["prediction"], row["ground_truth"])
            else:
                raise ValueError(key)
            for metric, value in scores.items():
                row[metric] = value
            row["endpoint"] = "offline"
            row["latency_sec"] = 0
            row["done_reason"] = "derived"
            row["prompt_eval_count"] = ""
            row["eval_count"] = ""
            write_row(out_path, row)


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize(records: list[dict], baseline_by_index: dict[int, dict] | None) -> dict:
    metrics = {name: [] for name in ("em", "cm", "anls", "f1")}
    lengths, latencies = [], []
    errors = 0
    exact_prediction = improved = worsened = em_gain = em_loss = 0
    comparable = 0
    for row in records:
        for name in metrics:
            metrics[name].append(as_float(row.get(name)))
        prediction = row.get("prediction", "")
        lengths.append(len(prediction))
        latencies.append(as_float(row.get("latency_sec")))
        errors += bool(row.get("error"))
        if baseline_by_index is not None:
            base = baseline_by_index.get(int(row["index"]))
            if base is not None:
                comparable += 1
                exact_prediction += prediction == base.get("prediction", "")
                delta = as_float(row.get("anls")) - as_float(base.get("anls"))
                improved += delta > 0
                worsened += delta < 0
                em_delta = as_float(row.get("em")) - as_float(base.get("em"))
                em_gain += em_delta > 0
                em_loss += em_delta < 0
    n = len(records)
    return {
        "n": n,
        **{name: (sum(values) / n if n else 0.0) for name, values in metrics.items()},
        "errors": errors,
        "mean_latency": sum(latencies) / n if n else 0.0,
        "median_length": statistics.median(lengths) if lengths else 0,
        "max_length": max(lengths, default=0),
        "exact_prediction": exact_prediction,
        "comparable": comparable,
        "anls_improved": improved,
        "anls_worsened": worsened,
        "em_gain": em_gain,
        "em_loss": em_loss,
    }


def build_report(output_dir: Path, manifest: dict) -> Path:
    all_keys = list(INFERENCE_VARIANTS) + list(DERIVED_VARIANTS)
    records_by_key = {key: read_csv_records(output_dir / f"{key}.csv") for key in all_keys}
    baseline_records = records_by_key["baseline"]
    baseline_by_index = {int(row["index"]): row for row in baseline_records}
    summaries = {
        key: summarize(rows, None if key == "baseline" else baseline_by_index)
        for key, rows in records_by_key.items()
    }
    baseline_summary = summaries["baseline"]

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    def delta(value: float, base: float) -> str:
        d = (value - base) * 100
        cls = "positive" if d > 0 else ("negative" if d < 0 else "neutral")
        return f'<span class="{cls}">{d:+.2f} pp</span>'

    summary_rows = []
    for key in all_keys:
        spec = INFERENCE_VARIANTS.get(key) or DERIVED_VARIANTS[key]
        s = summaries[key]
        exact = "baseline" if key == "baseline" else (
            f"{s['exact_prediction']}/{s['comparable']} "
            f"({(s['exact_prediction'] / s['comparable'] * 100 if s['comparable'] else 0):.2f}%)"
        )
        summary_rows.append(f"""
        <tr>
          <td><strong>{html.escape(spec['label'])}</strong><br><code>{key}</code></td>
          <td>{s['n']}</td><td>{pct(s['em'])}<br>{delta(s['em'], baseline_summary['em'])}</td>
          <td>{pct(s['cm'])}<br>{delta(s['cm'], baseline_summary['cm'])}</td>
          <td>{pct(s['anls'])}<br>{delta(s['anls'], baseline_summary['anls'])}</td>
          <td>{pct(s['f1'])}<br>{delta(s['f1'], baseline_summary['f1'])}</td>
          <td>{exact}</td><td>{s['errors']}</td><td>{s['mean_latency']:.3f}</td>
        </tr>""")

    sections = []
    for key in all_keys[1:]:
        spec = INFERENCE_VARIANTS.get(key) or DERIVED_VARIANTS[key]
        s = summaries[key]
        variant_by_index = {int(row["index"]): row for row in records_by_key[key]}
        changed = []
        for idx, base in baseline_by_index.items():
            row = variant_by_index.get(idx)
            if not row or row.get("prediction") == base.get("prediction"):
                continue
            changed.append((
                abs(as_float(row.get("anls")) - as_float(base.get("anls"))),
                idx, base, row,
            ))
        changed.sort(reverse=True, key=lambda item: item[0])
        examples = []
        for _, _, base, row in changed[:20]:
            examples.append(f"""
            <tr><td>{html.escape(row['image_filename'])}</td>
            <td>{html.escape(row['ground_truth'])}</td>
            <td>{html.escape(base.get('prediction', ''))}</td>
            <td>{html.escape(row.get('prediction', ''))}</td>
            <td>{as_float(base.get('anls')):.2f} → {as_float(row.get('anls')):.2f}</td></tr>""")
        sections.append(f"""
        <section><h2>{html.escape(spec['label'])}</h2>
        <p>{html.escape(spec['description'])}</p>
        <ul>
          <li>與 baseline 預測完全相同：{s['exact_prediction']}/{s['comparable']}</li>
          <li>ANLS 改善／變差：{s['anls_improved']}／{s['anls_worsened']}</li>
          <li>EM 由錯轉對／由對轉錯：{s['em_gain']}／{s['em_loss']}</li>
          <li>輸出長度中位數／最大值：{s['median_length']}／{s['max_length']}</li>
        </ul>
        <details><summary>查看 ANLS 差異最大的案例（最多 20 筆）</summary>
        <table><thead><tr><th>圖片</th><th>GT</th><th>Baseline</th><th>Variant</th><th>ANLS</th></tr></thead>
        <tbody>{''.join(examples) if examples else '<tr><td colspan="5">無預測差異</td></tr>'}</tbody></table>
        </details></section>""")

    manifest_text = html.escape(json.dumps(manifest, ensure_ascii=False, indent=2))
    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>OCR Ablation Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:auto;padding:24px;color:#17202a;background:#f6f8fa}}
h1,h2{{color:#102a43}} section{{background:white;border:1px solid #d8dee4;border-radius:10px;padding:18px;margin:20px 0}}
table{{width:100%;border-collapse:collapse;background:white}}th,td{{border:1px solid #d8dee4;padding:8px;text-align:left;vertical-align:top}}
th{{background:#eaf2f8}}td:nth-child(n+2){{font-variant-numeric:tabular-nums}}code,pre{{background:#eef1f4;border-radius:4px}}
pre{{padding:12px;white-space:pre-wrap}}.positive{{color:#087f23}}.negative{{color:#b42318}}.neutral{{color:#667085}}
.note{{border-left:5px solid #f0ad4e;padding:10px 14px;background:#fff8e6}}details{{margin-top:12px}}summary{{cursor:pointer;font-weight:600}}
</style></head><body>
<h1>GLM-OCR 單一變量消融實驗</h1>
<p>產生時間：{html.escape(utc_now())}；baseline 為合作者 aligned pipeline。</p>
<p class="note">每一列只改一個變量。Delta 以 baseline 為準；正值不一定代表設定普遍較好，只代表在本資料集與模型上的結果。此實驗測量單一變量主效應，不涵蓋變量之間的交互作用。</p>
<section><h2>總覽</h2><table><thead><tr><th>實驗</th><th>n</th><th>EM</th><th>CM</th><th>ANLS</th><th>F1</th><th>與 baseline 預測相同</th><th>Errors</th><th>平均延遲(s)</th></tr></thead>
<tbody>{''.join(summary_rows)}</tbody></table></section>
{''.join(sections)}
<section><h2>Run manifest</h2><details><summary>查看完整設定</summary><pre>{manifest_text}</pre></details></section>
</body></html>"""
    report_path = output_dir / "ablation_report.html"
    report_path.write_text(document, encoding="utf-8")
    atomic_json(output_dir / "summary.json", summaries)
    return report_path


def create_manifest(args: argparse.Namespace, samples: list[dict], selected: list[str]) -> dict:
    dataset_fingerprint = sha256_text("\n".join(
        f"{s['index']}\t{s['image_filename']}\t{s['image_sha256']}\t{s['ground_truth']}" for s in samples
    ))
    variants = {}
    for key in selected:
        spec = INFERENCE_VARIANTS[key]
        variants[key] = {
            "label": spec["label"],
            "description": spec["description"],
            "endpoint": f"/api/{spec['endpoint']}",
            "prompt": spec["prompt"],
            "prompt_sha256": sha256_text(spec["prompt"]),
            "options": spec["options"],
        }
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "model": args.model,
        "host": args.host,
        "labels": str(Path(args.labels).resolve()),
        "images_dir": str(Path(args.images_dir).resolve()),
        "sample_count": len(samples),
        "dataset_fingerprint": dataset_fingerprint,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "inference_variants": variants,
        "derived_variants": DERIVED_VARIANTS,
    }


def manifest_signature(manifest: dict) -> str:
    stable = dict(manifest)
    stable.pop("created_at", None)
    return sha256_text(json.dumps(stable, ensure_ascii=False, sort_keys=True))


def main() -> int:
    global RUNTIME_OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Run paired GLM-OCR one-factor-at-a-time ablations")
    parser.add_argument("--limit", type=int, default=0, help="0 = all 3706 TC-STR test samples")
    parser.add_argument("--model", default="ggml-org/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf:latest")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--labels", default=aligned_config.TEST_LABELS)
    parser.add_argument("--images-dir", default=aligned_config.IMAGES_DIR)
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--variants", nargs="*", choices=list(INFERENCE_VARIANTS), default=list(INFERENCE_VARIANTS))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    RUNTIME_OUTPUT_DIR = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    samples = load_samples(Path(args.labels), Path(args.images_dir), args.limit)
    selected = list(dict.fromkeys(args.variants))
    if "baseline" not in selected:
        selected.insert(0, "baseline")
    manifest = create_manifest(args, samples, selected)
    manifest["signature"] = manifest_signature(manifest)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("signature") != manifest["signature"]:
            raise RuntimeError(
                f"Existing output has a different run signature: {output_dir}. "
                "Use a new --output-dir or remove the old ablation output."
            )
        manifest = previous
        print(f"Resuming compatible run: {output_dir}", flush=True)
    else:
        atomic_json(manifest_path, manifest)

    atomic_json(output_dir / "dataset_manifest.json", [{
        "index": s["index"], "image_filename": s["image_filename"],
        "image_sha256": s["image_sha256"], "ground_truth": s["ground_truth"],
    } for s in samples])
    atomic_json(status_path, {"state": "starting", "updated_at": utc_now(), "total": len(samples)})

    print(f"Samples: {len(samples)}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    for key in selected:
        run_inference_variant(key, INFERENCE_VARIANTS[key], samples, args, output_dir, status_path)
        build_report(output_dir, manifest)

    write_derived_variants(output_dir)
    report = build_report(output_dir, manifest)
    remaining_errors = {
        key: sum(bool(row.get("error")) for row in read_csv_records(output_dir / f"{key}.csv"))
        for key in selected
    }
    final_state = "complete" if not any(remaining_errors.values()) else "complete_with_errors"
    atomic_json(status_path, {
        "state": final_state, "updated_at": utc_now(), "total": len(samples),
        "errors_by_variant": remaining_errors, "report": str(report),
    })
    print(f"\n{final_state}. Report: {report}", flush=True)
    if final_state != "complete":
        print("Some requests still failed. Run start_ablation.sh again to retry only failed indexes.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        try:
            atomic_json(RUNTIME_OUTPUT_DIR / "status.json", {
                "state": "failed", "updated_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            })
        except Exception:
            pass
        raise
    raise SystemExit(exit_code)
