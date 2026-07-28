"""TC-STR 主 scorer 與隔離的 legacy adapter。"""

from __future__ import annotations

import hashlib
import importlib.util
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def normalize_minimal(text: str | None) -> str:
    """唯一允許的主評分 normalization：CRLF、NFC、首尾 whitespace。"""
    return unicodedata.normalize("NFC", (text or "").replace("\r\n", "\n")).strip()


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def char_multiset_f1(prediction: str, ground_truth: str) -> float:
    p, g = Counter(prediction), Counter(ground_truth)
    if not prediction and not ground_truth:
        return 1.0
    if not prediction or not ground_truth:
        return 0.0
    overlap = sum((p & g).values())
    precision = overlap / len(prediction)
    recall = overlap / len(ground_truth)
    return 0.0 if not overlap else 2 * precision * recall / (precision + recall)


def score(prediction: str | None, ground_truth: str | None) -> dict[str, float]:
    raw_p, raw_g = prediction or "", ground_truth or ""
    p, g = normalize_minimal(raw_p), normalize_minimal(raw_g)
    distance = edit_distance(p, g)
    nls = 1 - distance / max(len(p), len(g), 1)
    return {
        "exact": float(p == g),
        "raw_exact": float(raw_p == raw_g),
        "edit_distance": float(distance),
        "gt_length": float(len(g)),
        "anls": nls if nls >= 0.5 else 0.0,
        "containment": float(g in p) if g else float(not p),
        "char_f1": char_multiset_f1(p, g),
        "format_compliance": float(raw_p == p and "\n" not in raw_p),
        "empty": float(not p),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows) or 1
    distances = sum(float(row["metrics"]["edit_distance"]) for row in rows)
    gt_length = sum(float(row["metrics"]["gt_length"]) for row in rows)
    return {
        "exact_accuracy": sum(row["metrics"]["exact"] for row in rows) / n,
        "raw_exact_accuracy": sum(row["metrics"]["raw_exact"] for row in rows) / n,
        "cer_micro": distances / max(gt_length, 1),
        "anls": sum(row["metrics"]["anls"] for row in rows) / n,
        "containment": sum(row["metrics"]["containment"] for row in rows) / n,
        "char_f1": sum(row["metrics"]["char_f1"] for row in rows) / n,
    }


class LegacyPostprocessor:
    """動態載入既有 ablation postprocessor；不得影響主 prediction。"""

    def __init__(self, source: Path):
        self.source = source
        payload = source.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        spec = importlib.util.spec_from_file_location("_ocf_legacy_postprocess", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"無法載入 legacy postprocessor: {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._clean = module.clean_prediction

    def clean(self, text: str) -> str:
        return str(self._clean(text))
