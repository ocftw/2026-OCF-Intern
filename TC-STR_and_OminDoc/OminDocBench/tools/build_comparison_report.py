#!/usr/bin/env python3
"""Build a self-contained, interactively-sortable HTML report comparing
every model's raw prediction side by side, page image + ground truth
included, for a configurable subset of pages (best / worst / random by one
baseline model's per-page score).

Unlike build_visual_report.py this script:
  - never needs a --run-id: it auto-discovers, per model id, the most
    recent complete `official_evaluation_full_1651` result across ALL run
    directories under paths.output_root (models evaluated under different
    run ids -- e.g. after a config change -- are found automatically).
  - embeds every requested model's per-page score into the HTML, so the
    page order can be re-sorted client-side (by any model, ascending or
    descending) without regenerating the file.
  - never touches any run's official evaluation output -- read-only.

Quick start:
    # see which models currently have a complete result to pick from
    python3 tools/build_comparison_report.py --list-models

    # baseline = gemma4_e2b: its 5 best + 5 worst pages, all discovered
    # models shown side by side, output to report.html
    python3 tools/build_comparison_report.py \
        --baseline gemma4_e2b --best 5 --worst 5 --random 5 \
        --out report.html

Open the resulting HTML in a browser: a toolbar at the top lets you
re-sort the selected pages by any model's score, filter by
best/worst/random, and search by filename -- all without touching this
script again.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omnidocbench.core import Settings, canonical_visible_text  # noqa: E402

MAX_TEXT_CHARS_DEFAULT = 2500
MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

# metric -> (per-page score filename, "lower is better?" -- all four are
# edit-distance style scores, so lower always means closer to the ground
# truth for every one of them)
METRIC_FILES = {
    "text_block": "predictions_quick_match_text_block_per_page_edit.json",
    "table": "predictions_quick_match_table_per_page_edit.json",
    "display_formula": "predictions_quick_match_display_formula_per_page_edit.json",
    "reading_order": "predictions_quick_match_reading_order_per_page_edit.json",
}


# --------------------------------------------------------------------------
# Discovery: find each model's latest complete result, wherever it lives.
# --------------------------------------------------------------------------

def discover_models(output_root: pathlib.Path) -> dict[str, dict]:
    """Return {model_id: {run_id, result_dir, predictions_dir, mtime}} using
    the newest `official_evaluation_full_1651/*` result found for that model
    id across every run directory under output_root."""
    found: dict[str, dict] = {}
    for run_dir in sorted(output_root.glob("v1_6_*")):
        models_root = run_dir / "models"
        if not models_root.is_dir():
            continue
        for model_dir in sorted(models_root.iterdir()):
            full_root = model_dir / "official_evaluation_full_1651"
            if not full_root.is_dir():
                continue
            candidates = [p for p in full_root.iterdir() if p.is_dir()]
            if not candidates:
                continue
            latest = max(candidates, key=lambda p: p.stat().st_mtime_ns)
            mtime = latest.stat().st_mtime_ns
            prev = found.get(model_dir.name)
            if prev is None or mtime > prev["mtime"]:
                found[model_dir.name] = {
                    "run_id": run_dir.name,
                    "result_dir": latest,
                    "predictions_dir": model_dir / "predictions",
                    "mtime": mtime,
                }
    return found


def diagnose_missing_model(output_root: pathlib.Path, model_id: str) -> str:
    """Best-effort explanation for why a model has no full_1651 result yet,
    by inspecting any in-progress batched evaluation for it."""
    notes = []
    for run_dir in sorted(output_root.glob("v1_6_*")):
        for state_path in sorted(
            (run_dir / "models" / model_id / "official_evaluation_batches").glob("bs*/batch_state.json")
        ):
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            counts = collections.Counter(b.get("status") for b in data.get("batches", []))
            notes.append(f"{run_dir.name}: {dict(counts)} / {data.get('batch_count')} 批")
    if notes:
        return "尚未有完整的 official_evaluation_full_1651 結果，偵測到批次評測進度 -- " + "; ".join(notes)
    return "找不到任何評測結果，請確認 model id 是否正確，或評測尚未執行。"


def load_logical_names(root: pathlib.Path) -> dict[str, str]:
    names: dict[str, str] = {}
    config_files = [root / "config" / "models.yaml"] + sorted((root / "config_variants").glob("*.yaml"))
    for path in config_files:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for m in data.get("models", []):
            if "id" in m:
                names.setdefault(m["id"], m.get("logical_name", m["id"]))
    return names


# --------------------------------------------------------------------------
# Per-model scores / summaries / predictions
# --------------------------------------------------------------------------

def load_per_page_scores(result_dir: pathlib.Path, metric: str) -> dict[str, float]:
    path = result_dir / METRIC_FILES[metric]
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary_row(result_dir: pathlib.Path) -> dict[str, float | None]:
    try:
        m = json.loads((result_dir / "predictions_quick_match_metric_result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    def g(*path):
        cur = m
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    return {
        "text_block_edit": g("text_block", "all", "Edit_dist", "ALL_page_avg"),
        "table_edit": g("table", "all", "Edit_dist", "ALL_page_avg"),
        "table_teds": g("table", "all", "TEDS", "all"),
        "formula_edit": g("display_formula", "all", "Edit_dist", "ALL_page_avg"),
        "formula_cdm": g("display_formula", "all", "CDM", "all"),
        "reading_order_edit": g("reading_order", "all", "Edit_dist", "ALL_page_avg"),
    }


def encode_image(image_path: pathlib.Path) -> tuple[str, str]:
    if not image_path.is_file():
        return "", ""
    mime = MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/png")
    return base64.b64encode(image_path.read_bytes()).decode("ascii"), mime


def truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = text[: int(limit * 0.7)]
    tail = text[-int(limit * 0.2):]
    return f"{head}\n\n... [截斷，原長 {len(text)} 字元] ...\n\n{tail}", True


def html_escape(text: str) -> str:
    # &quot; is escaped too (not just <, >) because this is also used to embed
    # JSON inside double-quoted HTML attributes (data-scores="..."); browsers
    # decode &quot; back to a literal " in text content, so this is safe
    # everywhere html_escape is used, not just in attributes.
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def score_pill_class(score: float | None) -> str:
    if score is None:
        return "warn"
    return "ok" if score < 0.3 else "warn" if score < 0.6 else "danger"


# --------------------------------------------------------------------------
# Page selection
# --------------------------------------------------------------------------

def select_pages(scores: dict[str, float], best: int, worst: int, rnd: int, seed: int) -> dict[str, list[str]]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    picked: dict[str, list[str]] = {"best": [], "worst": [], "random": []}
    used: set[str] = set()
    for name, _ in ranked[:best]:
        picked["best"].append(name)
        used.add(name)
    for name, _ in ranked[::-1][:worst]:
        if name not in used:
            picked["worst"].append(name)
            used.add(name)
    remaining = [name for name in scores if name not in used]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    picked["random"] = remaining[:rnd]
    return picked


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------

def build_page_card(
    filename: str,
    group: str,
    gt_pages_by_name: dict[str, dict],
    models: list[str],
    names: dict[str, str],
    model_scores: dict[str, dict[str, float]],
    model_pred_dirs: dict[str, pathlib.Path],
    image_dir: pathlib.Path,
    max_chars: int,
) -> str:
    image_path = image_dir / filename
    image_b64, image_mime = encode_image(image_path)
    gt_page = gt_pages_by_name.get(filename)
    gt_text = canonical_visible_text(gt_page) if gt_page else "(找不到 GT 對應頁面)"
    gt_text, gt_trunc = truncate(gt_text, max_chars)

    group_label = {"best": "最佳", "worst": "最差", "random": "隨機"}[group]
    group_class = {"best": "ok", "worst": "danger", "random": "info"}[group]

    scores_for_js = {m: model_scores[m].get(filename) for m in models}

    model_panels = []
    for model_id in models:
        score = scores_for_js[model_id]
        pred_path = model_pred_dirs[model_id] / pathlib.Path(filename).with_suffix(".md").name
        try:
            pred_text = pred_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pred_text = "(找不到預測檔案)"
        pred_text, pred_trunc = truncate(pred_text, max_chars)
        score_str = f"{score:.3f}" if score is not None else "N/A"
        model_panels.append(f"""
        <div class="panel" data-model="{html_escape(model_id)}">
          <div class="panel-head">
            <span class="model-name">{html_escape(names.get(model_id, model_id))}</span>
            <span class="pill {score_pill_class(score)}">{score_str}</span>
          </div>
          <pre class="text-block">{html_escape(pred_text)}</pre>
          {'<p class="trunc-note">已截斷，僅顯示部分內容</p>' if pred_trunc else ''}
        </div>""")

    scores_json = html_escape(json.dumps(scores_for_js))
    return f"""
    <section class="card" data-group="{group}" data-filename="{html_escape(filename)}" data-scores="{scores_json}">
      <header class="card-head">
        <span class="pill {group_class}">{group_label}</span>
        <span class="filename">{html_escape(filename)}</span>
      </header>
      <div class="card-body">
        <div class="panel image-panel">
          <div class="panel-head"><span class="model-name">原始頁面</span></div>
          {f'<img src="data:{image_mime};base64,{image_b64}" alt="{html_escape(filename)}" loading="lazy" />' if image_b64 else '<p class="trunc-note">找不到圖片</p>'}
        </div>
        <div class="panel">
          <div class="panel-head"><span class="model-name">Ground Truth(依閱讀順序重建)</span></div>
          <pre class="text-block">{html_escape(gt_text)}</pre>
          {'<p class="trunc-note">已截斷，僅顯示部分內容</p>' if gt_trunc else ''}
        </div>
        {''.join(model_panels)}
      </div>
    </section>"""


def compute_overall(s: dict) -> float | None:
    """Official leaderboard formula: ((1 - text edit) * 100 + table TEDS + formula CDM) / 3,
    with TEDS/CDM expressed on the same 0-100 scale as the text-edit term (see
    opendatalab/OmniDocBench README's "End-to-End Evaluation" section); our stored
    TEDS/CDM are 0-1 fractions, so they're scaled by 100 to match."""
    text_edit, table_teds, formula_cdm = s.get("text_block_edit"), s.get("table_teds"), s.get("formula_cdm")
    if text_edit is None or table_teds is None or formula_cdm is None:
        return None
    return ((1 - text_edit) * 100 + table_teds * 100 + formula_cdm * 100) / 3


# (label, data-key) for every sortable column, in display order; data-key matches
# both summaries' dict keys (except "overall", computed) and each row's data-metrics JSON.
SUMMARY_TABLE_COLUMNS = [
    ("Overall", "overall"),
    ("text_block edit", "text_block_edit"),
    ("table edit", "table_edit"),
    ("table TEDS", "table_teds"),
    ("formula edit", "formula_edit"),
    ("formula CDM", "formula_cdm"),
    ("reading_order edit", "reading_order_edit"),
]


def build_summary_table(models: list[str], names: dict[str, str], summaries: dict[str, dict]) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"

    rows = []
    for m in models:
        s = summaries.get(m, {})
        metrics = {key: s.get(key) for _, key in SUMMARY_TABLE_COLUMNS if key != "overall"}
        metrics["overall"] = compute_overall(s)
        metrics_json = html_escape(json.dumps(metrics))
        cells = "".join(
            f'<td class="overall-cell">{fmt(metrics["overall"])}</td>' if key == "overall"
            else f"<td>{fmt(metrics[key])}</td>"
            for _, key in SUMMARY_TABLE_COLUMNS
        )
        rows.append(f"""
        <tr data-metrics="{metrics_json}">
          <td class="rowname">{html_escape(names.get(m, m))}</td>
          {cells}
        </tr>""")

    header_cells = "".join(
        f'<th class="sortable" data-key="{key}">{html_escape(label)}<span class="sort-arrow"></span></th>'
        for label, key in SUMMARY_TABLE_COLUMNS
    )
    return f"""
    <table class="summary-table" id="summary-table">
      <caption>全語料庫整體分數(供比對用，非本報告子集平均) -- Overall/TEDS/CDM 越高越好，Edit_dist 越低越好 -- 點欄位標題可依該指標排序</caption>
      <thead>
        <tr>
          <th>模型</th>{header_cells}
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    {SUMMARY_TABLE_JS}"""


# One entry per column of build_summary_table(), in the same order. "key"
# matches load_summary_row()'s dict keys; "selected_by" is the --metric value
# (see METRIC_FILES) whose page-selection/sort this metric drives in THIS
# report, so the active one can be flagged for the reader.
METRIC_EXPLAINERS = [
    {
        "key": "text_block_edit",
        "label": "text_block edit",
        "direction": "↓ 越低越好(0 = 完全一致)",
        "selected_by": "text_block",
        "blurb": "內文(一般段落文字，不含表格、公式)辨識結果跟 GT 的整體差異程度。",
        "algorithm": "把整頁除了表格、公式以外的本文段落，依官方比對後攤平串接，計算預測與 GT 的正規化 Levenshtein 編輯距離：edit_distance ÷ max(len(pred), len(gt))。",
    },
    {
        "key": "table_edit",
        "label": "table edit",
        "direction": "↓ 越低越好",
        "selected_by": "table",
        "blurb": "表格的 HTML 原始碼(含標籤與文字)整體差異程度，字元層級的粗略指標。",
        "algorithm": "跟 text_block edit 同一套算法，只是比對對象換成 GT 與預測表格的 HTML 字串，一樣是正規化 Levenshtein 編輯距離。",
    },
    {
        "key": "table_teds",
        "label": "table TEDS",
        "direction": "↑ 越高越好(1 = 結構與內容完全一致)",
        "selected_by": None,
        "blurb": "專門評估表格「結構」(列、欄、合併儲存格)與儲存格內容是否正確，比單純比 HTML 字串更能反映表格有沒有被真正解析對。",
        "algorithm": "Tree-Edit-Distance-based Similarity(IBM TEDS)：把 GT 與預測的表格轉成樹狀結構(tr/td，含 rowspan/colspan/內容)，計算兩棵樹的編輯距離，再正規化成 0–1 的相似度。官方另外還有只看結構、不看儲存格文字的 TEDS-structure-only，這份報告沒有另外列出。",
    },
    {
        "key": "formula_edit",
        "label": "formula edit",
        "direction": "↓ 越低越好",
        "selected_by": "display_formula",
        "blurb": "公式 LaTeX 文字本身跟 GT 的差異程度，跟 text_block edit 演算法相同，只是比對對象換成每個獨立/行內公式。",
        "algorithm": "GT 與預測公式 LaTeX 字串的正規化 Levenshtein 編輯距離。缺點是同一條公式可以有很多種寫法都算對(例如 \\frac{a}{b} 跟 a/b)，這種純文字比對會誤判成「不一樣」，因此官方另外提供 formula CDM 作為更公平的替代指標。",
    },
    {
        "key": "formula_cdm",
        "label": "formula CDM",
        "direction": "↑ 越高越好",
        "selected_by": None,
        "blurb": "不比較 LaTeX 文字，而是把 GT 與預測的公式各自「畫出來」再比對圖像上的符號，比較不會被公式寫法差異誤傷，也是官方 Overall 總分採用的公式指標。",
        "algorithm": "Character Detection Matching(CDM)：把 GT 與預測的 LaTeX 各自用 pdflatex 渲染成圖片，偵測圖片裡每個字元/符號的位置，依視覺特徵與座標做逐字元比對，算出 precision/recall 後回報 F1 分數。",
    },
    {
        "key": "reading_order_edit",
        "label": "reading_order edit",
        "direction": "↓ 越低越好",
        "selected_by": "reading_order",
        "blurb": "模型輸出內容的「先後順序」跟人類真正閱讀順序是否一致(例如多欄版面、跨頁表格有沒有讀對順序)。",
        "algorithm": "把 GT 與預測各個內容區塊依官方比對結果重新排成一組索引序列，計算兩序列之間的正規化 Levenshtein 編輯距離。",
    },
]


def build_metrics_explainer(active_metric: str) -> str:
    rows = []
    for m in METRIC_EXPLAINERS:
        active_note = (
            ' <span class="pill ok">本報告排序/選頁依據</span>' if m["selected_by"] == active_metric else ""
        )
        rows.append(f"""
        <tr>
          <td class="rowname">{html_escape(m['label'])}{active_note}</td>
          <td><span class="pill info">{html_escape(m['direction'])}</span></td>
          <td>{html_escape(m['blurb'])}<span class="algo-note">{html_escape(m['algorithm'])}</span></td>
        </tr>""")
    return f"""
    <details class="metrics-explainer">
      <summary>📖 指標說明(意義 / 算法 / 方向) -- 點此展開或收合</summary>
      <table class="summary-table explainer-table">
        <thead><tr><th>指標</th><th>方向</th><th>意義與算法</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <p class="explainer-footnote">上面整體分數表的 Overall 欄位是官方 Leaderboard 公式：
        <code>Overall = ((1 − text edit) × 100 + table TEDS + formula CDM) ÷ 3</code>，數字越高越好，只吃這三個子指標，不含 table edit / reading_order edit。</p>
    </details>"""


CSS = """
:root{
  --bg:#f7f6fb; --surface:#ffffff; --surface-2:#f0eef8; --border:#e0dfec;
  --text:#1c1c28; --muted:#6b6b80;
  --accent:#7a6dc4; --accent-soft:#efe9fb;
  --ok:#2f8f57; --ok-soft:#e4f5ea;
  --warn:#a06a13; --warn-soft:#faf0dd;
  --danger:#c53f4f; --danger-soft:#fbe8ea;
  --info:#3167ad; --info-soft:#e6eefb;
  --mono: ui-monospace, "Cascadia Code", "SF Mono", Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", "PingFang TC", sans-serif;
}
:root[data-theme="dark"]{
  --bg:#14141c; --surface:#1b1b26; --surface-2:#20202e; --border:#2c2c3e;
  --text:#e9e9f2; --muted:#8d8da3;
  --accent:#a294e8; --accent-soft:#26223a;
  --ok:#5fbf7a; --ok-soft:#1c2b21;
  --warn:#e8a33d; --warn-soft:#332818;
  --danger:#e0616f; --danger-soft:#341f22;
  --info:#6fa3e0; --info-soft:#1c2733;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#14141c; --surface:#1b1b26; --surface-2:#20202e; --border:#2c2c3e;
    --text:#e9e9f2; --muted:#8d8da3;
    --accent:#a294e8; --accent-soft:#26223a;
    --ok:#5fbf7a; --ok-soft:#1c2b21;
    --warn:#e8a33d; --warn-soft:#332818;
    --danger:#e0616f; --danger-soft:#341f22;
    --info:#6fa3e0; --info-soft:#1c2733;
  }
}
*{box-sizing:border-box;}
body{background:var(--bg); color:var(--text); font-family:var(--sans); max-width:1600px; margin:0 auto; padding:2rem 1.5rem 4rem;}
h1{font-size:1.6rem; margin:0 0 .3rem; text-wrap:balance;}
.sub{color:var(--muted); margin:0 0 1rem; font-family:var(--mono); font-size:.85rem;}
.summary-table{border-collapse:collapse; width:100%; margin:0 0 1.5rem; font-size:.8rem; background:var(--surface); border:1px solid var(--border); border-radius:8px; overflow:hidden;}
.summary-table caption{text-align:left; caption-side:top; color:var(--muted); font-size:.75rem; padding:.5rem .8rem; background:var(--surface-2);}
.summary-table th, .summary-table td{padding:.5rem .8rem; border-bottom:1px solid var(--border); text-align:right;}
.summary-table th:first-child, .summary-table td:first-child{text-align:left;}
.summary-table .rowname{font-weight:600; color:var(--accent);}
.summary-table .overall-cell{font-weight:700; color:var(--text);}
.metrics-explainer{background:var(--surface); border:1px solid var(--border); border-radius:10px; margin:0 0 1.5rem; overflow:hidden;}
.metrics-explainer summary{cursor:pointer; padding:.7rem 1rem; font-weight:600; font-size:.85rem; color:var(--accent); background:var(--surface-2); list-style:none;}
.metrics-explainer summary::-webkit-details-marker{display:none;}
.metrics-explainer summary::before{content:"▸ "; display:inline-block;}
.metrics-explainer[open] summary::before{content:"▾ ";}
.metrics-explainer .explainer-table{margin:1rem; width:calc(100% - 2rem); border:1px solid var(--border);}
.metrics-explainer .explainer-table td{text-align:left; vertical-align:top; font-size:.78rem; line-height:1.5;}
.metrics-explainer .explainer-table th:nth-child(2), .metrics-explainer .explainer-table td:nth-child(2){white-space:nowrap;}
.algo-note{display:block; margin-top:.3rem; color:var(--muted); font-size:.74rem;}
.explainer-footnote{margin:0 1rem 1rem; font-size:.76rem; color:var(--muted);}
.explainer-footnote code{font-family:var(--mono); background:var(--surface-2); padding:.1em .4em; border-radius:4px;}
.toolbar{
  position:sticky; top:0; z-index:10; display:flex; flex-wrap:wrap; gap:1rem; align-items:center;
  background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:.8rem 1rem; margin-bottom:1.5rem;
  box-shadow:0 2px 10px rgba(0,0,0,.06);
}
.toolbar .group{display:flex; align-items:center; gap:.5rem; font-size:.82rem;}
.toolbar label{color:var(--muted);}
.toolbar select, .toolbar input[type=text]{
  font-family:var(--mono); font-size:.8rem; padding:.35em .6em; border-radius:6px; border:1px solid var(--border);
  background:var(--surface-2); color:var(--text);
}
.toolbar button{
  font-family:var(--mono); font-size:.8rem; padding:.35em .8em; border-radius:6px; border:1px solid var(--border);
  background:var(--accent-soft); color:var(--accent); cursor:pointer; font-weight:600;
}
.toolbar button:hover{filter:brightness(0.95);}
#sort-label{font-size:.8rem; color:var(--accent); font-weight:600;}
.legend{display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem; font-size:.85rem; color:var(--muted);}
.card{background:var(--surface); border:1px solid var(--border); border-radius:12px; margin-bottom:1.5rem; overflow:hidden;}
.card-head{display:flex; align-items:center; gap:.7rem; padding:.8rem 1.2rem; border-bottom:1px solid var(--border); background:var(--surface-2);}
.filename{font-family:var(--mono); font-size:.85rem; color:var(--muted); word-break:break-all;}
.card-body{display:grid; grid-template-columns:repeat(auto-fit, minmax(320px,1fr)); gap:1px; background:var(--border);}
.panel{background:var(--surface); padding:1rem; display:flex; flex-direction:column; gap:.6rem; transition:background .15s;}
.panel.sort-active{background:var(--accent-soft);}
.panel-head{display:flex; justify-content:space-between; align-items:center; gap:.5rem;}
.model-name{font-family:var(--mono); font-size:.78rem; letter-spacing:.03em; text-transform:uppercase; color:var(--accent); font-weight:600;}
.image-panel img{max-width:100%; height:auto; border-radius:6px; border:1px solid var(--border);}
.text-block{
  font-family:var(--mono); font-size:.76rem; line-height:1.5; white-space:pre-wrap; word-break:break-word;
  background:var(--surface-2); border:1px solid var(--border); border-radius:6px; padding:.7rem;
  max-height:420px; overflow-y:auto; margin:0;
}
.trunc-note{font-size:.72rem; color:var(--muted); margin:0; font-style:italic;}
.pill{
  display:inline-flex; align-items:center; font-family:var(--mono); font-size:.72rem; font-weight:600;
  padding:.25em .6em; border-radius:999px; white-space:nowrap;
}
.pill.ok{background:var(--ok-soft); color:var(--ok);}
.pill.warn{background:var(--warn-soft); color:var(--warn);}
.pill.danger{background:var(--danger-soft); color:var(--danger);}
.pill.info{background:var(--info-soft); color:var(--info);}
.pill.rank-badge{background:var(--accent); color:#fff;}
.summary-table th.sortable{cursor:pointer; user-select:none;}
.summary-table th.sortable:hover{color:var(--accent);}
.summary-table th.sort-active{color:var(--accent);}
.sort-arrow{display:inline-block; margin-left:.3em; font-size:.7em; opacity:.8;}
"""

# Sorts the #summary-table (per-model overall scores) by clicking a column
# header; each <tr data-metrics='{...}'> carries the raw numeric value per
# column key so sorting doesn't need to re-parse formatted cell text. Default
# on load: Overall, descending (highest/best model first).
SUMMARY_TABLE_JS = """
<script>
(function(){
  var table = document.getElementById('summary-table');
  if (!table) return;
  var tbody = table.querySelector('tbody');
  var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
  var headers = Array.prototype.slice.call(table.querySelectorAll('th.sortable'));
  var state = {key: null, dir: 'desc'};

  function rowValue(row, key){ return JSON.parse(row.dataset.metrics)[key]; }

  function sortByKey(key, dir){
    state = {key: key, dir: dir};
    var ranked = rows.slice().sort(function(a, b){
      var av = rowValue(a, key), bv = rowValue(b, key);
      var aMissing = (av === undefined || av === null);
      var bMissing = (bv === undefined || bv === null);
      if (aMissing && bMissing) return 0;
      if (aMissing) return 1;
      if (bMissing) return -1;
      return dir === 'asc' ? av - bv : bv - av;
    });
    ranked.forEach(function(r){ tbody.appendChild(r); });
    headers.forEach(function(h){
      var arrow = h.querySelector('.sort-arrow');
      if (h.dataset.key === key) {
        h.classList.add('sort-active');
        arrow.textContent = dir === 'asc' ? '▲' : '▼';
      } else {
        h.classList.remove('sort-active');
        arrow.textContent = '';
      }
    });
  }

  headers.forEach(function(h){
    h.addEventListener('click', function(){
      var key = h.dataset.key;
      var dir = (state.key === key && state.dir === 'desc') ? 'asc' : 'desc';
      sortByKey(key, dir);
    });
  });

  sortByKey('overall', 'desc');
})();
</script>
"""

TOOLBAR_JS = """
<script>
(function(){
  var container = document.getElementById('cards');
  var cards = Array.prototype.slice.call(container.querySelectorAll('.card'));
  var originalOrder = cards.slice();
  var sortSelect = document.getElementById('sort-model');
  var dirBtn = document.getElementById('sort-dir');
  var sortLabel = document.getElementById('sort-label');
  var filterInput = document.getElementById('filename-filter');
  var groupBoxes = Array.prototype.slice.call(document.querySelectorAll('.group-filter'));
  var legendModelNames = document.querySelectorAll('.legend-model-name');
  var originalBaselineText = legendModelNames.length ? legendModelNames[0].textContent : '';
  var dir = 'asc'; // edit-distance metrics: lower is better, so asc = best-first

  function cardScores(card){ return JSON.parse(card.dataset.scores); }

  function cmpBy(modelId){
    return function(a, b){
      var av = cardScores(a)[modelId], bv = cardScores(b)[modelId];
      var aMissing = (av === undefined || av === null);
      var bMissing = (bv === undefined || bv === null);
      if (aMissing && bMissing) return 0;
      if (aMissing) return 1;
      if (bMissing) return -1;
      return dir === 'asc' ? av - bv : bv - av;
    };
  }

  function clearRankBadges(){
    cards.forEach(function(c){
      var b = c.querySelector('.rank-badge');
      if (b) b.remove();
    });
  }

  function highlightPanels(modelId){
    document.querySelectorAll('.panel[data-model]').forEach(function(p){
      p.classList.toggle('sort-active', !!modelId && p.dataset.model === modelId);
    });
  }

  function applySort(){
    var modelId = sortSelect.value;
    clearRankBadges();
    if (modelId === '__original__') {
      originalOrder.forEach(function(c){ container.appendChild(c); });
      sortLabel.textContent = '目前順序: 產生時的分組(最差 → 隨機 → 最佳)';
      legendModelNames.forEach(function(el){ el.textContent = originalBaselineText; });
      highlightPanels(null);
      recomputeVisibility();
      return;
    }
    var ranked = cards.slice().sort(cmpBy(modelId));
    ranked.forEach(function(c, i){
      container.appendChild(c);
      var head = c.querySelector('.card-head');
      var badge = document.createElement('span');
      badge.className = 'pill rank-badge';
      badge.textContent = '#' + (i + 1);
      head.insertBefore(badge, head.firstChild);
    });
    var displayName = sortSelect.options[sortSelect.selectedIndex].text;
    sortLabel.textContent = '目前順序: 依 ' + displayName +
      ' 分數 ' + (dir === 'asc' ? '由低到高(較佳優先)' : '由高到低(較差優先)') + ' 排序';
    legendModelNames.forEach(function(el){ el.textContent = displayName; });
    highlightPanels(modelId);
    recomputeVisibility();
  }

  function recomputeVisibility(){
    var activeGroups = groupBoxes.filter(function(b){ return b.checked; }).map(function(b){ return b.value; });
    var q = filterInput.value.trim().toLowerCase();
    cards.forEach(function(c){
      var groupOk = activeGroups.indexOf(c.dataset.group) !== -1;
      var textOk = !q || c.dataset.filename.toLowerCase().indexOf(q) !== -1;
      c.style.display = (groupOk && textOk) ? '' : 'none';
    });
  }

  sortSelect.addEventListener('change', applySort);
  dirBtn.addEventListener('click', function(){
    dir = dir === 'asc' ? 'desc' : 'asc';
    dirBtn.textContent = dir === 'asc' ? '↑ 由低到高' : '↓ 由高到低';
    if (sortSelect.value !== '__original__') applySort();
  });
  filterInput.addEventListener('input', recomputeVisibility);
  groupBoxes.forEach(function(b){ b.addEventListener('change', recomputeVisibility); });
})();
</script>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an interactive, client-side-sortable multi-model comparison report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--models", default=None, help="comma-separated model ids (default: every model with a complete result)")
    parser.add_argument("--baseline", default=None, help="model id to rank best/worst/random by (default: first of --models)")
    parser.add_argument("--metric", choices=sorted(METRIC_FILES), default="text_block", help="per-page metric to select/sort by (default: text_block)")
    parser.add_argument("--best", type=int, default=5)
    parser.add_argument("--worst", type=int, default=5)
    parser.add_argument("--random", type=int, default=5, dest="rnd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=MAX_TEXT_CHARS_DEFAULT, help="truncate each text panel beyond this many characters")
    parser.add_argument("--out", default="comparison_report.html")
    parser.add_argument("--list-models", action="store_true", help="print discovered models and exit")
    args = parser.parse_args()

    settings = Settings.load()
    output_root = pathlib.Path(settings.raw["paths"]["output_root"])
    discovered = discover_models(output_root)
    names = load_logical_names(ROOT)

    if args.list_models:
        if not discovered:
            print(f"沒有在 {output_root} 底下找到任何 official_evaluation_full_1651 結果")
            return 0
        print(f"{'model_id':<18} {'logical_name':<20} {'run_id':<20} result_dir")
        for model_id, info in sorted(discovered.items()):
            print(f"{model_id:<18} {names.get(model_id, ''):<20} {info['run_id']:<20} {info['result_dir']}")
        for known_id in names:
            if known_id not in discovered:
                print(f"\n[缺少] {known_id}: {diagnose_missing_model(output_root, known_id)}")
        return 0

    models = [m.strip() for m in args.models.split(",")] if args.models else sorted(discovered)
    if not models:
        print("找不到任何有完整結果的模型；先用 --list-models 檢查狀態", file=sys.stderr)
        return 1

    missing = [m for m in models if m not in discovered]
    if missing:
        for m in missing:
            print(f"錯誤: model '{m}' 沒有完整的 official_evaluation_full_1651 結果 -- {diagnose_missing_model(output_root, m)}", file=sys.stderr)
        return 1

    baseline = args.baseline or models[0]
    if baseline not in models:
        print(f"錯誤: --baseline {baseline!r} 不在 --models 列表中: {models}", file=sys.stderr)
        return 1

    gt_pages = json.loads(settings.gt_path.read_text(encoding="utf-8"))
    gt_pages_by_name = {pathlib.Path(p["page_info"]["image_path"]).name: p for p in gt_pages}

    model_scores = {m: load_per_page_scores(discovered[m]["result_dir"], args.metric) for m in models}
    model_pred_dirs = {m: discovered[m]["predictions_dir"] for m in models}
    summaries = {m: load_summary_row(discovered[m]["result_dir"]) for m in models}

    selection = select_pages(model_scores[baseline], args.best, args.worst, args.rnd, args.seed)
    total_selected = sum(len(v) for v in selection.values())

    cards = []
    for group in ("worst", "random", "best"):
        for filename in selection[group]:
            cards.append(
                build_page_card(
                    filename, group, gt_pages_by_name, models, names,
                    model_scores, model_pred_dirs, settings.image_dir, args.max_chars,
                )
            )

    model_options = "".join(
        f'<option value="{html_escape(m)}">{html_escape(names.get(m, m))}</option>' for m in models
    )
    group_checkboxes = "".join(
        f'<label class="group"><input type="checkbox" class="group-filter" value="{g}" checked>{label}</label>'
        for g, label in (("worst", "最差"), ("random", "隨機"), ("best", "最佳"))
    )

    html = f"""<title>OmniDocBench 多模型比較報告</title>
<style>{CSS}</style>
<h1>OmniDocBench v1.6 多模型比較報告</h1>
<p class="sub">baseline={html_escape(baseline)} · metric={args.metric} · models={','.join(models)} · 共 {total_selected} 頁(最佳 {len(selection['best'])} / 最差 {len(selection['worst'])} / 隨機 {len(selection['random'])})</p>
{build_summary_table(models, names, summaries)}
{build_metrics_explainer(args.metric)}
<div class="toolbar">
  <div class="group">
    <label for="sort-model">排序依據:</label>
    <select id="sort-model">
      <option value="__original__">產生時的分組順序</option>
      {model_options}
    </select>
    <button id="sort-dir" type="button">↑ 由低到高</button>
  </div>
  <div class="group">{group_checkboxes}</div>
  <div class="group">
    <label for="filename-filter">篩選檔名:</label>
    <input type="text" id="filename-filter" placeholder="輸入部分檔名...">
  </div>
  <span id="sort-label">目前順序: 產生時的分組(最差 → 隨機 → 最佳)</span>
</div>
<div class="legend">
  <span class="pill danger">最差</span> 依 <span class="legend-model-name">{html_escape(names.get(baseline, baseline))}</span> 的 {args.metric} 由高到低
  <span class="pill info">隨機</span> 排除最佳/最差後隨機抽樣(seed={args.seed})
  <span class="pill ok">最佳</span> 依 <span class="legend-model-name">{html_escape(names.get(baseline, baseline))}</span> 的 {args.metric} 由低到高
  · 每個模型面板的分數即該模型在此頁的 {args.metric} edit distance(越低越好)
</div>
<div id="cards">{''.join(cards)}</div>
{TOOLBAR_JS}
"""
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({len(cards)} pages, {out_path.stat().st_size / 1024:.0f} KB, models={models})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
