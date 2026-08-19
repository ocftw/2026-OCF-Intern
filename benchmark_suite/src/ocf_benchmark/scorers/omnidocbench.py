"""OmniDocBench 官方 Docker evaluator adapter。

輸出格式及命令依 upstream ``configs/end2end.yaml``、``pdf_validation.py``。
本模組不提供 fallback 近似分數；任何官方 evaluator 失敗都回報 blocked。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def prediction_markdown_path(prediction_dir: Path, image_name: str) -> Path:
    return prediction_dir / f"{Path(image_name).stem}.md"


def run_official_evaluator(
    *,
    evaluator_dir: Path,
    gt_json: Path,
    predictions: Path,
    output_dir: Path,
    docker_image: str = "ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "end2end_eval": {
            "metrics": {
                "text_block": {"metric": ["Edit_dist"]},
                "display_formula": {"metric": ["Edit_dist", "CDM"], "cdm_workers": 1},
                "table": {"metric": ["TEDS", "Edit_dist"], "teds_workers": 1},
                "reading_order": {"metric": ["Edit_dist"]},
            },
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": "/workspace/gt/ground_truth.json"},
                "prediction": {"data_path": "/workspace/predictions"},
                "match_method": "quick_match",
                "match_workers": 1,
                "quick_match_truncated_timeout_sec": 300,
                "match_timeout_sec": 420,
                "timeout_fallback_max_chunk_span": 10,
                "timeout_fallback_order_penalty": 0.10,
            },
        }
    }
    config_path = output_dir / "official_end2end.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--entrypoint",
        "bash",
        "-v",
        f"{gt_json.resolve()}:/workspace/gt/ground_truth.json:ro",
        "-v",
        f"{predictions.resolve()}:/workspace/predictions:ro",
        "-v",
        f"{evaluator_dir.resolve()}:/workspace/official:ro",
        "-v",
        f"{output_dir.resolve()}:/workspace/official/result",
        "-v",
        f"{config_path.resolve()}:/workspace/official/configs/ocf.yaml:ro",
        docker_image,
        "-c",
        "cd /workspace/official && python pdf_validation.py --config configs/ocf.yaml",
    ]
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    (output_dir / "official_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "official_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode:
        return {
            "status": "blocked",
            "reason": f"官方 evaluator exit={proc.returncode}",
            "command": command,
        }
    candidates = sorted(output_dir.glob("*metric_result.json"))
    if not candidates:
        return {"status": "blocked", "reason": "官方 evaluator 未產生 metric_result.json"}
    raw = json.loads(candidates[-1].read_text(encoding="utf-8"))

    def metric(section: str, name: str) -> float | None:
        value = ((raw.get(section) or {}).get("all") or {}).get(name)
        if isinstance(value, dict):
            value = value.get("ALL_page_avg", value.get("all", value.get("ALL")))
        return float(value) if isinstance(value, (int, float)) else None

    text_edit = metric("text_block", "Edit_dist")
    table_teds = metric("table", "TEDS")
    formula_cdm = metric("display_formula", "CDM")
    reading_order = metric("reading_order", "Edit_dist")
    table_points = table_teds * 100 if table_teds is not None and table_teds <= 1 else table_teds
    formula_points = (
        formula_cdm * 100 if formula_cdm is not None and formula_cdm <= 1 else formula_cdm
    )
    overall = None
    if text_edit is not None and table_points is not None and formula_points is not None:
        overall = ((1 - text_edit) * 100 + table_points + formula_points) / 3
    return {
        "status": "completed",
        "overall": overall,
        "text_edit_distance": text_edit,
        "table_teds": table_teds,
        "formula_cdm": formula_cdm,
        "reading_order_edit_distance": reading_order,
        "official_metrics": raw,
        "command": command,
    }
