import os
import io
import sys
import json
import base64
import shutil
import string
import re
import time
import argparse
from pathlib import Path
from collections import Counter
import requests
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Metrics functions
def clean_text(text):
    if not text:
        return ""
    # Remove all whitespace
    text = re.sub(r'\s+', '', text)
    # Remove English punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove Chinese punctuation
    zh_punc = "，。！？；：（）［］【】「」『』〈〉《》——……——、•·「」『』"
    text = text.translate(str.maketrans('', '', zh_punc))
    return text

def calc_em(pred, gt):
    return 1.0 if clean_text(pred) == clean_text(gt) else 0.0

def calc_cm(pred, gt):
    p = clean_text(pred)
    g = clean_text(gt)
    if not p or not g:
        return 0.0
    return 1.0 if (g in p or p in g) else 0.0

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + 1,    # deletion
                               dp[i][j-1] + 1,    # insertion
                               dp[i-1][j-1] + 1)  # substitution
    return dp[m][n]

def calc_anls(pred, gt):
    p = clean_text(pred)
    g = clean_text(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    dist = edit_distance(p, g)
    max_len = max(len(p), len(g))
    score = 1.0 - dist / max_len
    return score if score >= 0.5 else 0.0

def calc_f1(pred, gt):
    p = clean_text(pred)
    g = clean_text(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    pred_counter = Counter(p)
    gt_counter = Counter(g)
    intersection = pred_counter & gt_counter
    common = sum(intersection.values())

    precision = common / len(p)
    recall = common / len(g)
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0

def get_base64_from_pil(pil_img):
    buffered = io.BytesIO()
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def clean_model_output(text):
    if not text:
        return ""
    text = text.strip()
    # Remove markdown code block wrappers
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        if len(lines) > 2:
            if lines[0].startswith("```"):
                text = "\n".join(lines[1:-1]).strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text

# Query Ollama
def query_ollama(model_name, img_base64, prompt, ollama_host):
    url = f"{ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_base64]
            }
        ],
        "options": {
            "num_predict": 25
        } if ("GLM" in model_name or "glm" in model_name) else {
            "temperature": 0.0,
            "top_p": 0.00001,
            "top_k": 1
        },
        "stream": False
    }

    # 40s timeout to allow initial model load
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=40)
            if response.status_code == 200:
                content = response.json()['message']['content']
                return clean_model_output(content)
            else:
                print(f"Ollama error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Ollama query failed: {e}, retrying...")
        time.sleep(1)
    return "[ERROR]"

def load_tc_str(base_dir, limit=-1):
    print("Loading TC-STR dataset...")
    base_dir = os.path.abspath(base_dir)
    labels_file = os.path.join(base_dir, "test_labels.txt")
    samples = []

    with open(labels_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if limit > 0:
        lines = lines[:limit]

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        rel_path, label = parts
        abs_path = os.path.join(base_dir, rel_path)
        samples.append({
            "dataset": "TC-STR-Benchmark",
            "index": i,
            "image_path": abs_path,
            "label": label
        })
    print(f"Loaded {len(samples)} samples from TC-STR.")
    return samples

def load_hf_dataset(split_name, limit=-1):
    print(f"Loading HuggingFace {split_name} split...")
    from datasets import load_dataset
    parquet_file = f"data/{split_name}-00000-of-00001.parquet"
    dataset = load_dataset("ZihCiLin/traditional-chinese-ocr-synthetic", data_files=parquet_file)
    data = dataset["train"]

    samples = []
    num_samples = len(data)
    if limit > 0:
        num_samples = min(num_samples, limit)

    for i in range(num_samples):
        item = data[i]
        samples.append({
            "dataset": f"HF-Synthetic-{split_name.replace('test_', '').capitalize()}",
            "index": i,
            "image": item["image"], # PIL image
            "label": item["text"]
        })
    print(f"Loaded {len(samples)} samples from HF {split_name}.")
    return samples

def generate_html_report(results, summary, output_html):
    print(f"Generating HTML report at {output_html}...")
    from jinja2 import Template

    html_template = """
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <title>OCR 模型評測報告 (Benchmark Report)</title>
    <style>
        :root {
            --bg-color: #0d1117;
            --panel-bg: #161b22;
            --border-color: #30363d;
            --text-color: #c9d1d9;
            --text-muted: #8b949e;
            --primary: #58a6ff;
            --success: #2ea043;
            --danger: #f85149;
            --accent: #ab7df8;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #fff;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .card {
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 25px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th, td {
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
        }
        th {
            background-color: rgba(255,255,255,0.05);
            color: #fff;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            font-size: 12px;
            font-weight: 600;
            border-radius: 12px;
            color: #fff;
        }
        .badge-success { background-color: var(--success); }
        .badge-danger { background-color: var(--danger); }

        /* Filters and detailed list */
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            align-items: center;
        }
        select, input {
            background-color: var(--bg-color);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            padding: 8px 12px;
            border-radius: 4px;
        }
        .test-case {
            border: 1px solid var(--border-color);
            border-radius: 6px;
            margin-bottom: 15px;
            background-color: var(--panel-bg);
            overflow: hidden;
        }
        .test-case-header {
            background-color: rgba(255,255,255,0.02);
            padding: 12px 15px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .test-case-body {
            padding: 15px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .test-image-container {
            flex: 0 0 250px;
            max-width: 300px;
            border: 1px solid var(--border-color);
            padding: 5px;
            background: #fff;
            border-radius: 4px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .test-image {
            max-width: 100%;
            max-height: 120px;
            object-fit: contain;
        }
        .test-details {
            flex: 1;
            min-width: 300px;
        }
        .compare-row {
            display: grid;
            grid-template-columns: 130px 1fr 1fr;
            gap: 10px;
            margin-bottom: 8px;
            align-items: center;
        }
        .compare-label {
            font-weight: bold;
            color: var(--text-muted);
        }
        .match-val {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .text-green { color: #58a6ff; font-weight: bold; }
        .text-red { color: #f85149; font-weight: bold; }

        .score-pills {
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }
        .score-pill {
            font-size: 11px;
            background: rgba(255,255,255,0.08);
            padding: 2px 6px;
            border-radius: 4px;
        }
    </style>
    <script>
        function filterResults() {
            const datasetFilter = document.getElementById('dataset-filter').value;
            const statusFilter = document.getElementById('status-filter').value;
            const searchVal = document.getElementById('search-input').value.toLowerCase();

            const cases = document.querySelectorAll('.test-case');
            cases.forEach(c => {
                const ds = c.getAttribute('data-dataset');
                const isErr = c.getAttribute('data-has-error') === 'true';
                const text = c.getAttribute('data-text').toLowerCase();

                let show = true;
                if (datasetFilter !== 'all' && ds !== datasetFilter) show = false;
                if (statusFilter === 'mismatch' && !isErr) show = false;
                if (statusFilter === 'perfect' && isErr) show = false;
                if (searchVal && !text.includes(searchVal)) show = false;

                c.style.display = show ? 'block' : 'none';
            });
        }
    </script>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1>OCR 模型基準評測報告</h1>
            <p style="color: var(--text-muted); margin-top: 5px;">
                評測時間: {{ summary.timestamp }} | 總測試樣本數: {{ summary.total_samples }}
            </p>
        </div>
    </div>

    <!-- Summary Metrics Cards -->
    <h2>評測數據總覽</h2>
    <div class="metrics-grid">
        {% for dataset, models in summary.dataset_metrics.items() %}
        <div class="card">
            <h3>{{ dataset }}</h3>
            <table>
                <thead>
                    <tr>
                        <th>模型</th>
                        <th>EM</th>
                        <th>CM</th>
                        <th>ANLS</th>
                        <th>F1</th>
                    </tr>
                </thead>
                <tbody>
                    {% for model_name, metrics in models.items() %}
                    <tr>
                        <td style="font-weight: 600;">{{ model_name }}</td>
                        <td>{{ "%.2f"|format(metrics.em * 100) }}%</td>
                        <td>{{ "%.2f"|format(metrics.cm * 100) }}%</td>
                        <td>{{ "%.2f"|format(metrics.anls * 100) }}%</td>
                        <td>{{ "%.2f"|format(metrics.f1 * 100) }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endfor %}
    </div>

    <!-- Detailed Analysis section -->
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 40px;">
        <h2>題庫詳細比對</h2>
        <div class="controls">
            <select id="dataset-filter" onchange="filterResults()">
                <option value="all">所有數據集</option>
                {% for dataset in summary.dataset_names %}
                <option value="{{ dataset }}">{{ dataset }}</option>
                {% endfor %}
            </select>
            <select id="status-filter" onchange="filterResults()">
                <option value="all">所有狀態</option>
                <option value="mismatch">僅顯示辨識出錯</option>
                <option value="perfect">僅顯示完全正確</option>
            </select>
            <input type="text" id="search-input" onkeyup="filterResults()" placeholder="搜尋正確答案...">
        </div>
    </div>

    <div class="test-list">
        {% for res in results %}
        {% set has_err = res.qwen.em < 1.0 or res.glm.em < 1.0 %}
        <div class="test-case"
             data-dataset="{{ res.dataset }}"
             data-has-error="{{ 'true' if has_err else 'false' }}"
             data-text="{{ res.label }}">
            <div class="test-case-header">
                <span style="font-weight: bold; color: var(--accent);">ID: {{ loop.index }} [{{ res.dataset }}]</span>
                {% if has_err %}
                <span class="badge badge-danger">有模型出錯</span>
                {% else %}
                <span class="badge badge-success">完全一致</span>
                {% endif %}
            </div>
            <div class="test-case-body">
                <div class="test-image-container">
                    <img class="test-image" src="{{ res.rel_image_path }}" alt="OCR Test Case">
                </div>
                <div class="test-details">
                    <div class="compare-row">
                        <div class="compare-label">正確答案:</div>
                        <div style="font-weight: bold; color: #fff; font-size: 1.1em;" colspan="2">{{ res.label }}</div>
                    </div>

                    <div class="compare-row" style="margin-top: 15px;">
                        <div class="compare-label">Qwen2.5-VL-3B:</div>
                        <div class="{{ 'text-green' if res.qwen.em >= 1.0 else 'text-red' }}">
                            {{ res.qwen.pred or '[無輸出]' }}
                        </div>
                        <div class="score-pills">
                            <span class="score-pill">EM: {{ res.qwen.em }}</span>
                            <span class="score-pill">ANLS: {{ "%.2f"|format(res.qwen.anls) }}</span>
                            <span class="score-pill">F1: {{ "%.2f"|format(res.qwen.f1) }}</span>
                        </div>
                    </div>

                    <div class="compare-row">
                        <div class="compare-label">GLM-OCR:</div>
                        <div class="{{ 'text-green' if res.glm.em >= 1.0 else 'text-red' }}">
                            {{ res.glm.pred or '[無輸出]' }}
                        </div>
                        <div class="score-pills">
                            <span class="score-pill">EM: {{ res.glm.em }}</span>
                            <span class="score-pill">ANLS: {{ "%.2f"|format(res.glm.anls) }}</span>
                            <span class="score-pill">F1: {{ "%.2f"|format(res.glm.f1) }}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</div>
</body>
</html>
    """

    template = Template(html_template)
    html_output = template.render(results=results, summary=summary)

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_output)
    print("HTML report generated successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to run per dataset (default: 5)")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_DIR / "results"), help="Output directory")
    parser.add_argument("--tc-str-dir", type=str, default=str(PROJECT_DIR / "data" / "TC-STR"), help="TC-STR dataset directory")
    parser.add_argument("--prompt-file", type=str, default=str(PROJECT_DIR / "prompts" / "ocr_zh.txt"), help="UTF-8 prompt file")
    parser.add_argument("--ollama-host", type=str, default=DEFAULT_OLLAMA_HOST, help="Ollama API base URL")
    parser.add_argument("--qwen-model", type=str, default="qwen2.5vl:3b", help="Ollama Qwen model tag")
    parser.add_argument("--glm-model", type=str, default="ggml-org/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf", help="Ollama GLM model tag")
    args = parser.parse_args()

    limit = args.limit
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    image_dir = os.path.join(output_dir, "report_images")
    os.makedirs(image_dir, exist_ok=True)

    # Models list
    models = {
        "qwen": args.qwen_model,
        "glm": args.glm_model
    }

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {args.prompt_file}")

    # Load all samples first
    all_samples = []

    # 1. TC-STR
    tc_samples = load_tc_str(args.tc_str_dir, limit)
    all_samples.extend(tc_samples)

    # 2. HF test_random
    hf_random = load_hf_dataset("test_random", limit)
    all_samples.extend(hf_random)

    # 3. HF test_semantic
    hf_semantic = load_hf_dataset("test_semantic", limit)
    all_samples.extend(hf_semantic)

    # Prepare image base64s and save local copies of images
    print("Preparing images and saving to report folder...")
    for sample in all_samples:
        ds_name = sample["dataset"]
        idx = sample["index"]
        rel_img_path = f"report_images/{ds_name.lower().replace('-', '_')}_{idx}.jpg"
        dest_img_path = os.path.join(output_dir, rel_img_path)
        sample["rel_img_path"] = rel_img_path

        if "image_path" in sample:
            if not os.path.exists(dest_img_path):
                shutil.copy(sample["image_path"], dest_img_path)
            with open(sample["image_path"], "rb") as f:
                sample["img_base64"] = base64.b64encode(f.read()).decode('utf-8')
        else:
            if not os.path.exists(dest_img_path):
                img = sample["image"]
                img.save(dest_img_path)
            # Load from saved image to save memory
            with open(dest_img_path, "rb") as f:
                sample["img_base64"] = base64.b64encode(f.read()).decode('utf-8')
            # Delete reference to PIL image to release memory
            sample["image"] = None

    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    qwen_predictions = []
    glm_predictions = []

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                cp = json.load(f)
                qwen_predictions = cp.get("qwen_predictions", [])
                glm_predictions = cp.get("glm_predictions", [])
                print(f"Found checkpoint! Loaded {len(qwen_predictions)} Qwen predictions and {len(glm_predictions)} GLM predictions.")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}. Starting fresh.")

    # Run batch inference for qwen2.5vl:3b
    print(f"\nRunning inference for model: {models['qwen']}...")
    start_qwen = len(qwen_predictions)
    if start_qwen < len(all_samples):
        for i in range(start_qwen, len(all_samples)):
            sample = all_samples[i]
            pred = query_ollama(models["qwen"], sample["img_base64"], prompt, args.ollama_host)
            qwen_predictions.append(pred)
            if (i+1) % 50 == 0 or (i+1) == len(all_samples):
                print(f"Progress ({models['qwen']}): {i+1}/{len(all_samples)}")
                # Save checkpoint
                try:
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump({"qwen_predictions": qwen_predictions, "glm_predictions": glm_predictions}, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Failed to save checkpoint: {e}")
    else:
        print("All Qwen predictions loaded from checkpoint. Skipping inference.")

    # Run batch inference for GLM-OCR
    print(f"\nRunning inference for model: {models['glm']}...")
    start_glm = len(glm_predictions)
    if start_glm < len(all_samples):
        for i in range(start_glm, len(all_samples)):
            sample = all_samples[i]
            pred = query_ollama(models["glm"], sample["img_base64"], prompt, args.ollama_host)
            glm_predictions.append(pred)
            if (i+1) % 50 == 0 or (i+1) == len(all_samples):
                print(f"Progress ({models['glm']}): {i+1}/{len(all_samples)}")
                # Save checkpoint
                try:
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump({"qwen_predictions": qwen_predictions, "glm_predictions": glm_predictions}, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Failed to save checkpoint: {e}")
    else:
        print("All GLM predictions loaded from checkpoint. Skipping inference.")

    # Calculate metrics
    print("\nCalculating metrics...")
    results = []

    # Accumulate metrics per dataset
    metrics_acc = {}
    for sample in all_samples:
        ds_name = sample["dataset"]
        if ds_name not in metrics_acc:
            metrics_acc[ds_name] = {
                "qwen": {"em": [], "cm": [], "anls": [], "f1": []},
                "glm": {"em": [], "cm": [], "anls": [], "f1": []}
            }

    for i, sample in enumerate(all_samples):
        ds_name = sample["dataset"]
        label = sample["label"]
        q_pred = qwen_predictions[i]
        g_pred = glm_predictions[i]

        q_em = calc_em(q_pred, label)
        q_cm = calc_cm(q_pred, label)
        q_anls = calc_anls(q_pred, label)
        q_f1 = calc_f1(q_pred, label)

        g_em = calc_em(g_pred, label)
        g_cm = calc_cm(g_pred, label)
        g_anls = calc_anls(g_pred, label)
        g_f1 = calc_f1(g_pred, label)

        metrics_acc[ds_name]["qwen"]["em"].append(q_em)
        metrics_acc[ds_name]["qwen"]["cm"].append(q_cm)
        metrics_acc[ds_name]["qwen"]["anls"].append(q_anls)
        metrics_acc[ds_name]["qwen"]["f1"].append(q_f1)

        metrics_acc[ds_name]["glm"]["em"].append(g_em)
        metrics_acc[ds_name]["glm"]["cm"].append(g_cm)
        metrics_acc[ds_name]["glm"]["anls"].append(g_anls)
        metrics_acc[ds_name]["glm"]["f1"].append(g_f1)

        results.append({
            "dataset": ds_name,
            "label": label,
            "rel_image_path": sample["rel_img_path"],
            "qwen": {
                "pred": q_pred,
                "em": q_em,
                "cm": q_cm,
                "anls": q_anls,
                "f1": q_f1
            },
            "glm": {
                "pred": g_pred,
                "em": g_em,
                "cm": g_cm,
                "anls": g_anls,
                "f1": g_f1
            }
        })

    dataset_metrics = {}
    for ds_name, m_dict in metrics_acc.items():
        dataset_metrics[ds_name] = {}
        for m_name in ["qwen", "glm"]:
            dataset_metrics[ds_name][m_name] = {
                "em": sum(m_dict[m_name]["em"]) / len(m_dict[m_name]["em"]) if m_dict[m_name]["em"] else 0,
                "cm": sum(m_dict[m_name]["cm"]) / len(m_dict[m_name]["cm"]) if m_dict[m_name]["cm"] else 0,
                "anls": sum(m_dict[m_name]["anls"]) / len(m_dict[m_name]["anls"]) if m_dict[m_name]["anls"] else 0,
                "f1": sum(m_dict[m_name]["f1"]) / len(m_dict[m_name]["f1"]) if m_dict[m_name]["f1"] else 0,
            }

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "total_samples": len(all_samples),
        "dataset_names": list(metrics_acc.keys()),
        "dataset_metrics": dataset_metrics
    }

    # Save raw data
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, ensure_ascii=False, indent=2)

    # Generate HTML report
    generate_html_report(results, summary, os.path.join(output_dir, "report.html"))

    # Remove checkpoint on successful completion
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print("Checkpoint cleared.")
        except Exception as e:
            print(f"Failed to remove checkpoint: {e}")

    print("\nBenchmark finished successfully!")

if __name__ == "__main__":
    main()
