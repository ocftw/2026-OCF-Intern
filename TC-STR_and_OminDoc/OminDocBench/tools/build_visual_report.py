#!/usr/bin/env python3
"""Build a lightweight, self-contained visual report: page image + ground
truth + each model's raw prediction, side by side, for a configurable subset
of pages (best / worst / random by text-block edit distance). Never touches
any run's official evaluation output -- read-only.

Usage:
    python3 tools/build_visual_report.py --run-id RUN_ID \
        --models gemma4_e2b,gemma4_26b_a4b \
        --best 5 --worst 5 --random 5 --seed 42 \
        --out /path/to/report.html
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omnidocbench.core import Settings, canonical_visible_text  # noqa: E402

MAX_TEXT_CHARS = 2500
MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def latest_result_dir(settings: Settings, model_id: str) -> pathlib.Path:
    root = settings.output_dir / "models" / model_id / "official_evaluation_full_1651"
    candidates = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no official evaluation result found under {root}")
    return candidates[0]


def load_per_page_scores(settings: Settings, model_id: str) -> dict[str, float]:
    result_dir = latest_result_dir(settings, model_id)
    path = result_dir / "predictions_quick_match_text_block_per_page_edit.json"
    return json.loads(path.read_text(encoding="utf-8"))


def encode_image(image_path: pathlib.Path) -> tuple[str, str]:
    if not image_path.is_file():
        return "", ""
    mime = MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/png")
    return base64.b64encode(image_path.read_bytes()).decode("ascii"), mime


def truncate(text: str, limit: int = MAX_TEXT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = text[: int(limit * 0.7)]
    tail = text[-int(limit * 0.2):]
    return f"{head}\n\n... [截斷，原長 {len(text)} 字元] ...\n\n{tail}", True


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


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_page_card(
    filename: str,
    group: str,
    gt_pages_by_name: dict[str, dict],
    settings: Settings,
    models: list[str],
    model_scores: dict[str, dict[str, float]],
    model_pred_dirs: dict[str, pathlib.Path],
) -> str:
    image_path = settings.image_dir / filename
    image_b64, image_mime = encode_image(image_path)
    gt_page = gt_pages_by_name.get(filename)
    gt_text = canonical_visible_text(gt_page) if gt_page else "(找不到 GT 對應頁面)"
    gt_text, gt_trunc = truncate(gt_text)

    group_label = {"best": "最佳", "worst": "最差", "random": "隨機"}[group]
    group_class = {"best": "ok", "worst": "danger", "random": "info"}[group]

    model_panels = []
    for model_id in models:
        score = model_scores[model_id].get(filename)
        pred_path = model_pred_dirs[model_id] / pathlib.Path(filename).with_suffix(".md").name
        try:
            pred_text = pred_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pred_text = "(找不到預測檔案)"
        pred_text, pred_trunc = truncate(pred_text)
        score_str = f"{score:.3f}" if score is not None else "N/A"
        model_panels.append(f"""
        <div class="panel">
          <div class="panel-head">
            <span class="model-name">{html_escape(model_id)}</span>
            <span class="pill {'ok' if (score or 1) < 0.3 else 'warn' if (score or 1) < 0.6 else 'danger'}">
              edit_dist {score_str}
            </span>
          </div>
          <pre class="text-block">{html_escape(pred_text)}</pre>
          {'<p class="trunc-note">已截斷，僅顯示部分內容</p>' if pred_trunc else ''}
        </div>""")

    return f"""
    <section class="card">
      <header class="card-head">
        <span class="pill {group_class}">{group_label}</span>
        <span class="filename">{html_escape(filename)}</span>
      </header>
      <div class="card-body">
        <div class="panel image-panel">
          <div class="panel-head"><span class="model-name">原始頁面</span></div>
          {f'<img src="data:{image_mime};base64,{image_b64}" alt="{html_escape(filename)}" />' if image_b64 else '<p class="trunc-note">找不到圖片</p>'}
        </div>
        <div class="panel">
          <div class="panel-head"><span class="model-name">Ground Truth(依閱讀順序重建)</span></div>
          <pre class="text-block">{html_escape(gt_text)}</pre>
          {'<p class="trunc-note">已截斷，僅顯示部分內容</p>' if gt_trunc else ''}
        </div>
        {''.join(model_panels)}
      </div>
    </section>"""


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
body{background:var(--bg); color:var(--text); font-family:var(--sans); max-width:1400px; margin:0 auto; padding:2.5rem 1.5rem 4rem;}
h1{font-size:1.6rem; margin:0 0 .3rem; text-wrap:balance;}
.sub{color:var(--muted); margin:0 0 1rem; font-family:var(--mono); font-size:.85rem;}
.legend{display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:2rem; font-size:.85rem; color:var(--muted);}
.card{background:var(--surface); border:1px solid var(--border); border-radius:12px; margin-bottom:1.5rem; overflow:hidden;}
.card-head{display:flex; align-items:center; gap:.7rem; padding:.8rem 1.2rem; border-bottom:1px solid var(--border); background:var(--surface-2);}
.filename{font-family:var(--mono); font-size:.85rem; color:var(--muted); word-break:break-all;}
.card-body{display:grid; grid-template-columns:repeat(auto-fit, minmax(320px,1fr)); gap:1px; background:var(--border);}
.panel{background:var(--surface); padding:1rem; display:flex; flex-direction:column; gap:.6rem;}
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
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a lightweight image+prediction visual report")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", required=True, help="comma-separated model ids")
    parser.add_argument("--best", type=int, default=0)
    parser.add_argument("--worst", type=int, default=0)
    parser.add_argument("--random", type=int, default=0, dest="rnd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank-by-model", default=None, help="model id to rank best/worst by (default: first --models entry)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rank_model = args.rank_by_model or models[0]

    import os

    os.environ["OMNIDOCBENCH_RUN_ID"] = args.run_id
    settings = Settings.load()

    gt_pages = json.loads(settings.gt_path.read_text(encoding="utf-8"))
    gt_pages_by_name = {pathlib.Path(p["page_info"]["image_path"]).name: p for p in gt_pages}

    model_scores = {m: load_per_page_scores(settings, m) for m in models}
    model_pred_dirs = {m: settings.output_dir / "models" / m / "predictions" for m in models}

    selection = select_pages(model_scores[rank_model], args.best, args.worst, args.rnd, args.seed)
    total_selected = sum(len(v) for v in selection.values())

    cards = []
    for group in ("worst", "random", "best"):
        for filename in selection[group]:
            cards.append(
                build_page_card(filename, group, gt_pages_by_name, settings, models, model_scores, model_pred_dirs)
            )

    html = f"""<title>OmniDocBench 視覺化報告 · {args.run_id}</title>
<style>{CSS}</style>
<h1>OmniDocBench v1.6 視覺化辨識結果報告</h1>
<p class="sub">run_id={args.run_id} · models={','.join(models)} · 排序依據={rank_model} 的 text_block edit distance · 共 {total_selected} 頁</p>
<div class="legend">
  <span class="pill danger">最差</span> 依 {html_escape(rank_model)} text edit distance 由高到低
  <span class="pill info">隨機</span> 排除最佳/最差後隨機抽樣(seed={args.seed})
  <span class="pill ok">最佳</span> 依 {html_escape(rank_model)} text edit distance 由低到高
</div>
{''.join(cards)}
"""
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({len(cards)} pages, {out_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
