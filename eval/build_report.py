# -*- coding: utf-8 -*-
"""
讀取 run_eval.py 產生的 raw_results.csv，彙總指標，
輸出一個獨立、自包含的一頁式 HTML 報告（圖表 + 可篩選的錯誤圖片 gallery）。

用法：
    python build_report.py
    python build_report.py --raw eval/results/raw_results.csv --out eval/results/report.html
"""

import argparse
import base64
import difflib
import html
import io
import json
import os
import re

import pandas as pd
from PIL import Image

from config import RAW_RESULTS_CSV, REPORT_HTML, IMAGES_DIR, TEXT_LEN_BUCKETS, MODELS


# ---------- 配色（來自 dataviz skill 的驗證過色板，固定順序指派） ----------
PALETTE = {
    "blue":   {"light": "#2a78d6", "dark": "#3987e5"},
    "aqua":   {"light": "#1baf7a", "dark": "#199e70"},
    "yellow": {"light": "#eda100", "dark": "#c98500"},
    "green":  {"light": "#008300", "dark": "#008300"},
}
METRIC_ORDER = [("em", "EM"), ("cm", "CM"), ("anls", "ANLS"), ("f1", "F1")]
METRIC_HUES = ["blue", "aqua", "yellow", "green"]
MODEL_KEYS_ORDER = [m["key"] for m in MODELS]
MODEL_HUES = ["blue", "aqua", "yellow", "green"]
MODEL_TAG = {m["key"]: m["tag"] for m in MODELS}

_CATEGORY_RE = re.compile(r"^([A-Za-z]+)_\d")


def hue_pair(hue_name):
    h = PALETTE[hue_name]
    return h["light"], h["dark"]


def model_hue(model_key):
    idx = MODEL_KEYS_ORDER.index(model_key) if model_key in MODEL_KEYS_ORDER else 0
    return MODEL_HUES[idx % len(MODEL_HUES)]


# ---------- 小工具 ----------

def text_len_bucket(text):
    n = len(text or "")
    for lo, hi, label in TEXT_LEN_BUCKETS:
        if lo <= n < hi:
            return label
    return TEXT_LEN_BUCKETS[-1][2]


def category_label(filename):
    """從檔名前綴猜類別，例如 billboard_00000_010_雜貨舖.jpg -> billboard。
    猜不出來就歸類成 other。"""
    m = _CATEGORY_RE.match(filename or "")
    return m.group(1) if m else "other"


def char_diff_html(a, b):
    """回傳 (ground_truth_html, prediction_html)，用 span 標示差異字元"""
    a, b = a or "", b or ""
    sm = difflib.SequenceMatcher(None, a, b)
    a_parts, b_parts = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        seg_a, seg_b = html.escape(a[i1:i2]), html.escape(b[j1:j2])
        if tag == "equal":
            a_parts.append(seg_a)
            b_parts.append(seg_b)
        else:
            if seg_a:
                a_parts.append(f'<span class="diff-del">{seg_a}</span>')
            if seg_b:
                b_parts.append(f'<span class="diff-add">{seg_b}</span>')
    return "".join(a_parts), "".join(b_parts)


def thumbnail_data_uri(path, max_width=320):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > max_width:
                ratio = max_width / im.width
                im = im.resize((max_width, max(1, int(im.height * ratio))))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


# ---------- SVG 長條圖 ----------

def _rounded_top_path(x, y, w, h, r=4):
    if h <= 0:
        return ""
    r = min(r, w / 2, h)
    return (f"M{x:.1f},{y+h:.1f} L{x:.1f},{y+r:.1f} "
            f"Q{x:.1f},{y:.1f} {x+r:.1f},{y:.1f} "
            f"L{x+w-r:.1f},{y:.1f} Q{x+w:.1f},{y:.1f} {x+w:.1f},{y+r:.1f} "
            f"L{x+w:.1f},{y+h:.1f} Z")


def svg_grouped_bar(categories, series, series_hue, y_max=1.0, width=680, height=300,
                     value_fmt="{:.2f}", counts=None):
    """
    categories: x 軸分組標籤 list
    series: list of (series_key, series_label, values) — values 長度 = len(categories)
    series_hue: {series_key: hue_name}
    counts: 可選 {(series_key, cat_index): n}，用於 tooltip 顯示樣本數
    """
    margin_left, margin_bottom, margin_top, margin_right = 42, 46, 16, 16
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    n_cat = max(len(categories), 1)
    n_series = max(len(series), 1)
    group_w = plot_w / n_cat
    bar_gap = 3
    outer_gap = 10
    bar_w = max(4.0, (group_w - outer_gap - bar_gap * (n_series - 1)) / n_series)

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
              f'class="chart" role="img">']

    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = margin_top + plot_h * (1 - frac)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
                      f'class="gridline" />')
        parts.append(f'<text x="{margin_left - 6}" y="{y + 4:.1f}" class="axis-label" '
                      f'text-anchor="end">{y_max * frac:.1f}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h:.1f}" '
                  f'x2="{width - margin_right}" y2="{margin_top + plot_h:.1f}" class="baseline" />')

    for ci, cat in enumerate(categories):
        gx = margin_left + ci * group_w + outer_gap / 2
        for si, (skey, slabel, values) in enumerate(series):
            v = values[ci] if values[ci] is not None else 0.0
            bh = plot_h * (max(0.0, min(1.0, v)) / y_max if y_max else 0)
            bx = gx + si * (bar_w + bar_gap)
            by = margin_top + plot_h - bh
            light, dark = hue_pair(series_hue.get(skey, "blue"))
            n = counts.get((skey, ci)) if counts else None
            n_txt = f" (n={n})" if n is not None else ""
            tip = f"{slabel} · {cat}: {value_fmt.format(v)}{n_txt}"
            path_d = _rounded_top_path(bx, by, bar_w, bh, r=3)
            parts.append(
                f'<path d="{path_d}" data-tip="{html.escape(tip)}" '
                f'style="fill:{light}" data-dark="{dark}" class="bar" />'
            )
        parts.append(f'<text x="{gx + (n_series * bar_w + (n_series - 1) * bar_gap) / 2:.1f}" '
                      f'y="{height - margin_bottom + 18}" class="axis-label" '
                      f'text-anchor="middle">{html.escape(str(cat))}</text>')

    parts.append("</svg>")
    return "".join(parts)


def legend_html(series_hue, labels):
    """labels: [(key, label), ...]"""
    items = []
    for key, label in labels:
        light, dark = hue_pair(series_hue.get(key, "blue"))
        items.append(
            f'<span class="legend-item"><span class="swatch" style="background:{light}" '
            f'data-dark="{dark}"></span>{html.escape(label)}</span>'
        )
    return f'<div class="legend">{"".join(items)}</div>'


def table_html(headers, rows):
    thead = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body_rows = []
    for r in rows:
        cells = "".join(f"<td>{html.escape(str(c))}</td>" for c in r)
        body_rows.append(f"<tr>{cells}</tr>")
    return (f'<table class="data-table"><thead><tr>{thead}</tr></thead>'
            f'<tbody>{"".join(body_rows)}</tbody></table>')


# ---------- 依模型分組的長條圖（x 軸=某個維度的分組，series=模型） ----------

def model_series_chart(df, group_col, group_order, model_keys, model_tags):
    by_group = df.groupby([group_col, "model_key"])["anls"].mean().unstack("model_key")
    by_group_n = df.groupby([group_col, "model_key"])["anls"].count().unstack("model_key")
    series, hue = [], {}
    for k in model_keys:
        vals = [
            by_group.loc[g, k] if (g in by_group.index and k in by_group.columns
                                    and not pd.isna(by_group.loc[g, k])) else None
            for g in group_order
        ]
        series.append((k, model_tags[k], vals))
        hue[k] = model_hue(k)
    counts = {}
    for k in model_keys:
        for gi, g in enumerate(group_order):
            if g in by_group_n.index and k in by_group_n.columns and not pd.isna(by_group_n.loc[g, k]):
                counts[(k, gi)] = int(by_group_n.loc[g, k])
    svg = svg_grouped_bar(group_order, series, hue, counts=counts)
    legend = legend_html(hue, [(k, model_tags[k]) for k in model_keys])
    return svg, legend


# ---------- 主流程 ----------

def build(raw_csv, out_html, images_dir):
    # keep_default_na=False：避免空字串預測（例如呼叫失敗或模型真的沒有輸出）被 pandas
    # 讀成 NaN，導致報告中顯示成字面 "nan"
    df = pd.read_csv(raw_csv, keep_default_na=False, na_values=[])
    # 多次 --resume 補跑之後，同一張圖同一個模型可能同時留下舊的失敗紀錄跟新的成功紀錄
    # （附加寫入不會刪掉舊的錯誤列）。只保留每組 (image, model) 最後一次的嘗試結果。
    df = df.drop_duplicates(subset=["image_filename", "model_key"], keep="last")
    for col in ("em", "cm", "anls", "f1", "latency_sec"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["text_len_bucket"] = df["ground_truth"].apply(text_len_bucket)
    df["category"] = df["image_filename"].apply(category_label)
    df["has_error"] = df["error"].fillna("").astype(str).str.len() > 0

    model_keys = [k for k in MODEL_KEYS_ORDER if k in set(df["model_key"])]
    if not model_keys:
        model_keys = sorted(df["model_key"].unique().tolist())
    model_tags = {k: MODEL_TAG.get(k, k) for k in model_keys}

    # ---- 總覽卡片 ----
    overall = df.groupby("model_key").agg(
        em=("em", "mean"), cm=("cm", "mean"), anls=("anls", "mean"), f1=("f1", "mean"),
        latency=("latency_sec", "mean"), n=("em", "count"), errors=("has_error", "sum"),
    ).reindex(model_keys)

    cards = []
    for k in model_keys:
        row = overall.loc[k]
        cards.append(f'''
        <div class="card">
          <div class="card-model">{html.escape(model_tags[k])}</div>
          <div class="card-hero">{row["anls"]:.0%}<span class="card-hero-label">ANLS</span></div>
          <div class="card-sub-grid">
            <div><span class="muted">EM</span> {row["em"]:.0%}</div>
            <div><span class="muted">CM</span> {row["cm"]:.0%}</div>
            <div><span class="muted">F1</span> {row["f1"]:.0%}</div>
            <div><span class="muted">延遲</span> {row["latency"]:.1f}s</div>
          </div>
          <div class="card-footnote">n={int(row["n"])}，錯誤 {int(row["errors"])} 次</div>
        </div>''')
    cards_html = "".join(cards)

    # ---- 圖表 A：各模型整體指標 ----
    series_a = []
    for mkey, mlabel in METRIC_ORDER:
        vals = [overall.loc[k, mkey] if k in overall.index else 0.0 for k in model_keys]
        series_a.append((mkey, mlabel, vals))
    hue_a = {mkey: METRIC_HUES[i % len(METRIC_HUES)] for i, (mkey, _) in enumerate(METRIC_ORDER)}
    counts_a = {(mkey, ci): int(overall.loc[k, "n"]) for ci, k in enumerate(model_keys) for mkey, _ in METRIC_ORDER}
    chart_a_svg = svg_grouped_bar([model_tags[k] for k in model_keys], series_a, hue_a, counts=counts_a)
    chart_a_legend = legend_html(hue_a, METRIC_ORDER)
    table_a = table_html(
        ["模型", "EM", "CM", "ANLS", "F1", "平均延遲(s)", "樣本數", "錯誤次數"],
        [[model_tags[k],
          f'{overall.loc[k,"em"]:.2%}', f'{overall.loc[k,"cm"]:.2%}',
          f'{overall.loc[k,"anls"]:.2%}', f'{overall.loc[k,"f1"]:.2%}',
          f'{overall.loc[k,"latency"]:.2f}', int(overall.loc[k, "n"]), int(overall.loc[k, "errors"])]
         for k in model_keys]
    )

    # ---- 圖表 B：文字長度對 ANLS 的影響（x=長度分組，series=模型） ----
    len_order = [b[2] for b in TEXT_LEN_BUCKETS]
    chart_b_svg, chart_b_legend = model_series_chart(df, "text_len_bucket", len_order, model_keys, model_tags)

    # ---- 圖表 C：類別對 ANLS 的影響（x=從檔名偵測到的類別，series=模型） ----
    # 只有偵測到 2 種以上類別才顯示這張圖，只有一種類別（例如全部都是 billboard）畫出來沒意義。
    categories = sorted(df["category"].unique().tolist())
    if len(categories) > 1:
        chart_c_svg, chart_c_legend = model_series_chart(df, "category", categories, model_keys, model_tags)
        chart_c_section = CHART_C_SECTION_TEMPLATE.format(
            chart_c_svg=chart_c_svg, chart_c_legend=chart_c_legend,
        )
    else:
        chart_c_section = ""

    # ---- Gallery 資料（每張圖 + 每個模型一筆） ----
    images_b64 = {}
    rows = []
    for _, r in df.iterrows():
        fn = str(r["image_filename"])
        if fn not in images_b64:
            images_b64[fn] = thumbnail_data_uri(os.path.join(images_dir, fn))
        gt_html, pred_html = char_diff_html(str(r["ground_truth"]), str(r["prediction"]))
        rows.append({
            "image_filename": fn,
            "model_key": str(r["model_key"]),
            "model_tag": str(r["model_tag"]),
            "ground_truth": str(r["ground_truth"]),
            "prediction": str(r["prediction"]),
            "prediction_raw": str(r["prediction_raw"]) if "prediction_raw" in df.columns and pd.notna(r["prediction_raw"]) else "",
            "gt_html": gt_html,
            "pred_html": pred_html,
            "em": float(r["em"]) if pd.notna(r["em"]) else 0.0,
            "cm": float(r["cm"]) if pd.notna(r["cm"]) else 0.0,
            "anls": float(r["anls"]) if pd.notna(r["anls"]) else 0.0,
            "f1": float(r["f1"]) if pd.notna(r["f1"]) else 0.0,
            "text_len_bucket": str(r["text_len_bucket"]),
            "category": str(r["category"]),
            "latency": float(r["latency_sec"]) if pd.notna(r["latency_sec"]) else None,
            "error": str(r["error"]) if pd.notna(r["error"]) and str(r["error"]) else "",
        })

    rows_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    images_json = json.dumps(images_b64, ensure_ascii=False).replace("</", "<\\/")
    model_options = "".join(
        f'<option value="{html.escape(k)}">{html.escape(model_tags[k])}</option>' for k in model_keys
    )
    model_hue_json = json.dumps({k: hue_pair(model_hue(k)) for k in model_keys}, ensure_ascii=False)

    html_out = HTML_TEMPLATE.format(
        n_images=df["image_filename"].nunique(),
        n_models=len(model_keys),
        cards_html=cards_html,
        chart_a_svg=chart_a_svg, chart_a_legend=chart_a_legend, table_a=table_a,
        chart_b_svg=chart_b_svg, chart_b_legend=chart_b_legend,
        chart_c_section=chart_c_section,
        model_options=model_options,
        rows_json=rows_json, images_json=images_json, model_hue_json=model_hue_json,
    )

    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"[+] 報告已產生: {out_html}")


CHART_C_SECTION_TEMPLATE = """
  <div class="section">
    <h2>圖片類別對 ANLS 的影響（從檔名前綴自動偵測）</h2>
    {chart_c_svg}
    {chart_c_legend}
  </div>"""


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>TC-STR 場景文字辨識(OCR)模型評估報告</title>
<style>
:root {{
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --gridline: #e1e0d9; --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --diff-add-bg: #e3f6ea; --diff-add-fg: #0a6b2f;
  --diff-del-bg: #fbe7e7; --diff-del-fg: #a13232;
  --tag-good: #0ca30c; --tag-bad: #d03b3b;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --gridline: #2c2c2a; --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --diff-add-bg: #12301f; --diff-add-fg: #6fdb9a;
    --diff-del-bg: #3a1c1c; --diff-del-fg: #e88b8b;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.5;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 80px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 16px; margin: 0 0 12px; color: var(--text-secondary); }}
.meta {{ color: var(--text-muted); font-size: 13px; margin-bottom: 28px; }}
.section {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 20px 22px; margin-bottom: 22px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px;
  margin-bottom: 22px; }}
.card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; }}
.card-model {{ font-size: 13px; color: var(--text-secondary); margin-bottom: 6px; word-break: break-all; }}
.card-hero {{ font-size: 32px; font-weight: 600; }}
.card-hero-label {{ font-size: 12px; font-weight: 400; color: var(--text-muted); margin-left: 6px; }}
.card-sub-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; margin-top: 10px; font-size: 13px; }}
.muted {{ color: var(--text-muted); }}
.card-footnote {{ margin-top: 10px; font-size: 12px; color: var(--text-muted); }}
.chart {{ width: 100%; height: auto; overflow: visible; }}
.gridline {{ stroke: var(--gridline); stroke-width: 1; }}
.baseline {{ stroke: var(--baseline); stroke-width: 1; }}
.axis-label {{ fill: var(--text-muted); font-size: 10px; }}
.bar {{ cursor: pointer; }}
@media (prefers-color-scheme: dark) {{
  .bar {{ fill: attr(data-dark); }}
}}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 8px; font-size: 12px; color: var(--text-secondary); }}
.legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
.swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
details.table-toggle {{ margin-top: 10px; }}
details.table-toggle > summary {{ cursor: pointer; font-size: 12px; color: var(--text-secondary); }}
.data-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }}
.data-table th, .data-table td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }}
.data-table th {{ color: var(--text-muted); font-weight: 500; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 16px; }}
.controls select, .controls input[type="text"] {{
  background: var(--surface-1); color: var(--text-primary); border: 1px solid var(--border);
  border-radius: 8px; padding: 7px 10px; font-size: 13px; font-family: inherit;
}}
.controls label {{ font-size: 13px; color: var(--text-secondary); display: inline-flex; align-items: center; gap: 6px; }}
.gallery-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; }}
.gcard {{ border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--surface-1); }}
.gcard img {{ width: 100%; display: block; background: #333; }}
.gcard-body {{ padding: 12px 14px; }}
.gcard-row {{ font-size: 13px; margin-bottom: 4px; }}
.gcard-label {{ color: var(--text-muted); font-size: 11px; margin-bottom: 2px; }}
.raw-toggle {{ margin-top: 4px; }}
.raw-toggle > summary {{ cursor: pointer; font-size: 11px; color: var(--text-muted); }}
.raw-row {{ font-size: 12px; color: var(--text-secondary); word-break: break-all; margin-top: 4px; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
.badge {{ font-size: 11px; padding: 2px 7px; border-radius: 999px; background: var(--gridline); color: var(--text-secondary); }}
.badge.good {{ background: color-mix(in srgb, var(--tag-good) 18%, transparent); color: var(--tag-good); }}
.badge.bad {{ background: color-mix(in srgb, var(--tag-bad) 18%, transparent); color: var(--tag-bad); }}
.diff-add {{ background: var(--diff-add-bg); color: var(--diff-add-fg); border-radius: 3px; padding: 0 1px; }}
.diff-del {{ background: var(--diff-del-bg); color: var(--diff-del-fg); border-radius: 3px; padding: 0 1px;
  text-decoration: line-through; }}
#tooltip {{
  position: fixed; pointer-events: none; background: var(--text-primary); color: var(--surface-1);
  font-size: 12px; padding: 5px 9px; border-radius: 6px; opacity: 0; transform: translate(-50%, -120%);
  transition: opacity .08s; z-index: 50; white-space: nowrap;
}}
.empty-state {{ text-align: center; color: var(--text-muted); padding: 40px 0; font-size: 13px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>TC-STR 場景文字辨識(OCR)模型評估報告</h1>
  <div class="meta">{n_images} 張圖片 × {n_models} 個模型 · 指標定義見頁尾</div>

  <div class="cards">{cards_html}</div>

  <div class="section">
    <h2>各模型整體表現（EM / CM / ANLS / F1）</h2>
    {chart_a_svg}
    {chart_a_legend}
    <details class="table-toggle"><summary>顯示詳細數字表</summary>{table_a}</details>
  </div>

  <div class="section">
    <h2>文字長度對 ANLS 的影響</h2>
    {chart_b_svg}
    {chart_b_legend}
  </div>
{chart_c_section}

  <div class="section">
    <h2>圖片明細（可篩選 / 排序，看哪張出錯）</h2>
    <div class="controls">
      <label>模型
        <select id="f-model"><option value="">全部</option>{model_options}</select>
      </label>
      <label><input type="checkbox" id="f-error-only" /> 只顯示有錯誤 (EM=0)</label>
      <label>排序
        <select id="f-sort">
          <option value="anls-asc">ANLS 由低到高</option>
          <option value="anls-desc">ANLS 由高到低</option>
          <option value="filename">檔名</option>
        </select>
      </label>
      <input type="text" id="f-search" placeholder="搜尋文字內容…" />
      <span class="muted" id="f-count"></span>
    </div>
    <div class="gallery-grid" id="gallery"></div>
    <div class="empty-state" id="gallery-empty" style="display:none">沒有符合條件的結果</div>
  </div>

  <div class="section">
    <h2>指標定義</h2>
    <div style="font-size:13px; color:var(--text-secondary)">
      <p><b>EM</b>（Exact Match）：預測文字同 ground truth 完全一致先計 1 分。</p>
      <p><b>CM</b>（Containment Match）：預測文字完整包含 ground truth 作為子字串就計 1 分。</p>
      <p><b>ANLS</b>：1 − 正規化編輯距離，低於 0.5 門檻直接當 0 分（DocVQA 慣例）。</p>
      <p><b>F1</b>：字元級 precision/recall 的調和平均（中文沒有分詞邊界，以字元當作 token）。</p>
    </div>
  </div>
</div>

<div id="tooltip"></div>

<script>
const ROWS = {rows_json};
const IMAGES = {images_json};
const MODEL_HUE = {model_hue_json};

function pct(v) {{ return (v * 100).toFixed(0) + '%'; }}

function escapeHtml(s) {{
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}}

function renderGallery() {{
  const modelFilter = document.getElementById('f-model').value;
  const errorOnly = document.getElementById('f-error-only').checked;
  const sortBy = document.getElementById('f-sort').value;
  const search = document.getElementById('f-search').value.trim();

  let rows = ROWS.filter(r => {{
    if (modelFilter && r.model_key !== modelFilter) return false;
    if (errorOnly && r.em >= 1) return false;
    if (search && !r.ground_truth.includes(search) && !r.prediction.includes(search)) return false;
    return true;
  }});

  if (sortBy === 'anls-asc') rows.sort((a, b) => a.anls - b.anls);
  else if (sortBy === 'anls-desc') rows.sort((a, b) => b.anls - a.anls);
  else rows.sort((a, b) => a.image_filename.localeCompare(b.image_filename));

  document.getElementById('f-count').textContent = `${{rows.length}} / ${{ROWS.length}} 筆`;
  const grid = document.getElementById('gallery');
  const empty = document.getElementById('gallery-empty');
  grid.innerHTML = '';
  empty.style.display = rows.length ? 'none' : 'block';

  const frag = document.createDocumentFragment();
  for (const r of rows) {{
    const el = document.createElement('div');
    el.className = 'gcard';
    const emBadge = r.em >= 1 ? 'good' : 'bad';
    el.innerHTML = `
      <img loading="lazy" src="${{IMAGES[r.image_filename] || ''}}" alt="${{r.image_filename}}" />
      <div class="gcard-body">
        <div class="gcard-label">Ground Truth</div>
        <div class="gcard-row">${{r.gt_html}}</div>
        <div class="gcard-label">${{r.model_tag}} 辨識結果</div>
        <div class="gcard-row">${{r.pred_html || '<span class="muted">(空白)</span>'}}</div>
        ${{r.prediction_raw && r.prediction_raw !== r.prediction ? `
        <details class="raw-toggle">
          <summary>原始輸出（清理前）</summary>
          <div class="gcard-row raw-row">${{escapeHtml(r.prediction_raw)}}</div>
        </details>` : ''}}
        <div class="badges">
          <span class="badge ${{emBadge}}">EM ${{pct(r.em)}}</span>
          <span class="badge">CM ${{pct(r.cm)}}</span>
          <span class="badge">ANLS ${{pct(r.anls)}}</span>
          <span class="badge">F1 ${{pct(r.f1)}}</span>
          <span class="badge">${{r.text_len_bucket}}</span>
          <span class="badge">${{r.category}}</span>
          ${{r.error ? `<span class="badge bad">錯誤: ${{r.error}}</span>` : ''}}
        </div>
      </div>`;
    frag.appendChild(el);
  }}
  grid.appendChild(frag);
}}

['f-model', 'f-error-only', 'f-sort'].forEach(id =>
  document.getElementById(id).addEventListener('change', renderGallery));
document.getElementById('f-search').addEventListener('input', renderGallery);
renderGallery();

// 圖表 hover tooltip（事件代理）
const tooltip = document.getElementById('tooltip');
document.addEventListener('mousemove', (e) => {{
  const bar = e.target.closest('[data-tip]');
  if (!bar) {{ tooltip.style.opacity = 0; return; }}
  tooltip.textContent = bar.getAttribute('data-tip');
  tooltip.style.left = e.clientX + 'px';
  tooltip.style.top = e.clientY + 'px';
  tooltip.style.opacity = 1;
}});
document.addEventListener('mouseleave', () => {{ tooltip.style.opacity = 0; }});

// dark mode 底下用 data-dark 色值取代 bar 顏色（CSS attr() 對 fill 支援有限，用 JS 保險）
if (window.matchMedia('(prefers-color-scheme: dark)').matches) {{
  document.querySelectorAll('.bar').forEach(b => {{
    const d = b.getAttribute('data-dark');
    if (d) b.style.fill = d;
  }});
}}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="產生一頁式 TC-STR OCR 評估報告")
    parser.add_argument("--raw", default=RAW_RESULTS_CSV)
    parser.add_argument("--out", default=REPORT_HTML)
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    args = parser.parse_args()

    if not os.path.exists(args.raw):
        print(f"[!] 找不到原始結果 CSV: {args.raw}（請先跑 run_eval.py）")
        return

    build(args.raw, args.out, args.images_dir)


if __name__ == "__main__":
    main()
