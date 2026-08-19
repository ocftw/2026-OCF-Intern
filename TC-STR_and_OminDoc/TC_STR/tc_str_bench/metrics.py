from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import Any


def normalize_primary(text: str | None) -> str:
    return unicodedata.normalize("NFC", (text or "").replace("\r\n", "\n").replace("\r", "\n")).strip()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def score(prediction: str | None, ground_truth: str | None) -> dict[str, float]:
    pred = normalize_primary(prediction)
    gt = normalize_primary(ground_truth)
    distance = levenshtein(pred, gt)
    nls = 1.0 - distance / max(len(pred), len(gt), 1)
    cp, cg = Counter(pred), Counter(gt)
    overlap = sum((cp & cg).values())
    if not pred and not gt:
        f1 = 1.0
    elif not pred or not gt or not overlap:
        f1 = 0.0
    else:
        precision, recall = overlap / len(pred), overlap / len(gt)
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "em": float(pred == gt),
        "cm": float(gt in pred) if gt else float(not pred),
        "anls": nls if nls >= 0.5 else 0.0,
        "f1": f1,
    }


_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")
_PREFIX = re.compile(r"^(文字|文本|辨識結果|OCR結果|answer|text|ocr)\s*[:：]\s*", re.I)
_HTML = re.compile(r"<[^>]+>")
_REPEAT = re.compile(r"(.)\1{3,}")
_ECHO = re.compile(
    r"(你是一個專業的OCR|你是一个专业的OCR|OCR文本辨識引擎|OCR文字辨識引擎|"
    r"OCR文本辨识引擎|OCR文字辨识引擎|請仔細觀察圖片|请仔细观察图片|"
    r"不要加任何解釋|不要加任何解释)"
)
_THINK = re.compile(r"^.*</think>", re.S)
_WRAP_START = "\"'「『“‘"
_WRAP_END = "\"'」』”’"


def aligned_supplementary(text: str | None) -> tuple[str, list[str]]:
    value = (text or "").strip()
    changes: list[str] = []
    if "</think>" in value:
        value = _THINK.sub("", value).strip()
        changes.append("remove_thinking_prefix")
    echo = _ECHO.search(value)
    if echo:
        value = value[: echo.start()].strip()
        changes.append("truncate_prompt_echo")
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, list) and parsed:
        extracted: list[str] = []
        for item in parsed:
            if isinstance(item, str):
                extracted.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "value", "caption"):
                    if isinstance(item.get(key), str) and item[key]:
                        extracted.append(item[key])
        if extracted:
            value = "".join(extracted)
            changes.append("extract_json_array")
    cleaned = _CODE_FENCE.sub("", value).strip()
    if cleaned != value:
        value = cleaned
        changes.append("remove_code_fence")
    if "<" in value and ">" in value:
        cleaned = _HTML.sub("", value).strip()
        if cleaned != value:
            value = cleaned
            changes.append("remove_html_tags")
    cleaned = _PREFIX.sub("", value).strip()
    if cleaned != value:
        value = cleaned
        changes.append("remove_label_prefix")
    if len(value) >= 2 and value[0] in _WRAP_START and value[-1] in _WRAP_END:
        value = value[1:-1].strip()
        changes.append("unwrap_quotes")
    cleaned = re.sub(r"[\r\n\t]+", "", value)
    if cleaned != value:
        value = cleaned
        changes.append("remove_linebreaks_tabs")
    cleaned = _REPEAT.sub(r"\1", value)
    if cleaned != value:
        value = cleaned
        changes.append("compress_4plus_repeated_character")
    if len(value) > 200:
        value = value[:200]
        changes.append("truncate_200_characters")
    return value, changes


def anomaly_flags(raw: str | None, prompt: str, done_reason: str | None, eval_count: int | None, limit: int) -> dict[str, bool]:
    value = raw or ""
    stripped = value.strip()
    lower = stripped.lower()
    control_removed = re.sub(r"</?(think|analysis|final|tool)[^>]*>", "", stripped, flags=re.I).strip()
    repetitive = bool(re.search(r"(.)\1{9,}", stripped)) or bool(re.search(r"(.{2,20})\1{5,}", stripped))
    formatting = bool(re.match(r"^\s*(```|[{[]|<(html|div|p)\b)", stripped, re.I))
    refusal = any(token in lower for token in ("無法辨識", "無法協助", "抱歉", "cannot", "can't help", "sorry"))
    echo = prompt[:24] in value or "你是一個專業的OCR文字辨識引擎" in value
    thinking = bool(re.search(r"</?think>|analysis:", value, re.I))
    truncated = done_reason == "length" or (eval_count is not None and eval_count >= limit)
    return {
        "empty": not stripped,
        "control_only": bool(stripped) and not control_removed,
        "prompt_echo": echo,
        "refusal": refusal,
        "formatting": formatting,
        "repetition": repetitive,
        "thinking": thinking,
        "unrelated": False,
        "truncated": truncated,
    }


def aggregate(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {"em": 0.0, "cm": 0.0, "anls": 0.0, "f1": 0.0}
    return {key: sum(row[key] for row in metric_rows) / len(metric_rows) for key in ("em", "cm", "anls", "f1")}
