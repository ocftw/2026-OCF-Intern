"""VisTW 離線 deterministic parser，絕不把資料送到外部 API。"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any

import requests

_MARKER = re.compile(r"答案\s*[:：]\s*([A-DＡ-Ｄ])", re.IGNORECASE)
_SINGLE = re.compile(r"^\s*([A-DＡ-Ｄ])\s*$", re.IGNORECASE)


def parse_answer(response: str | None) -> str | None:
    text = response or ""
    matches = list(_MARKER.finditer(text))
    if matches:
        return unicodedata.normalize("NFKC", matches[-1].group(1)).upper()
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        match = _SINGLE.match(lines[-1])
        if match:
            return unicodedata.normalize("NFKC", match.group(1)).upper()
    return None


def score(response: str | None, ground_truth: str) -> dict[str, Any]:
    answer = parse_answer(response)
    return {"parsed_answer": answer, "valid": answer is not None, "correct": answer == ground_truth}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subjects: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        subjects[row["subject"]].append(bool(row["metrics"]["correct"]))
    per_subject = {
        subject: {"n": len(values), "accuracy": sum(values) / len(values)}
        for subject, values in sorted(subjects.items())
    }
    n = len(rows) or 1
    return {
        "macro_accuracy": sum(v["accuracy"] for v in per_subject.values())
        / max(len(per_subject), 1),
        "micro_accuracy": sum(r["metrics"]["correct"] for r in rows) / n,
        "invalid_rate": sum(not r["metrics"]["valid"] for r in rows) / n,
        "per_subject": per_subject,
    }


def official_parser_adapter(
    response: str,
    formatted_question: str,
    *,
    enabled: bool,
    api_key: str | None,
    model: str = "gpt-4o-mini",
) -> str | None:
    """選配、獨立分數的 upstream-style LLM fallback；預設永遠不會呼叫。"""
    if not enabled:
        raise RuntimeError("official parser adapter 未明確啟用")
    if not api_key:
        raise RuntimeError("啟用 official parser adapter 必須提供 OPENAI_API_KEY")
    prompt = (
        "Extract the final multiple-choice answer from the response. "
        "Return exactly one letter A, B, C, or D.\n\n"
        f"Question and choices:\n{formatted_question}\n\nResponse:\n{response}"
    )
    result = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    result.raise_for_status()
    text = result.json()["choices"][0]["message"]["content"]
    match = re.fullmatch(r"\s*([A-D])\s*", text, re.IGNORECASE)
    return match.group(1).upper() if match else None
