# -*- coding: utf-8 -*-
"""清理模型輸出：去除多餘的解釋文字、code fence、引號、換行等雜訊"""

import json
import re

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")
_LABEL_PREFIX_RE = re.compile(
    r"^(文字|文本|辨識結果|OCR結果|answer|text|ocr)\s*[:：]\s*",
    re.IGNORECASE,
)
_WRAP_QUOTES = "\"'「『“‘"
_WRAP_QUOTES_END = "\"'」』”’"

# 部分模型（例如 chandra-ocr-2）偶爾會用 JSON 陣列包住答案，可能是單純字串陣列
# ["答案文字"]，也可能是文件版面模型常見的 [{"text": "字"}, ...] 或
# [{"label": "Text", "bbox": "..."}]（純版面框，沒有文字內容）。
# 一律先嘗試用 json.loads 真正解析，不要用正規表達式硬抓引號中間的內容——
# 舊版正規表達式遇到多元素陣列或 \uXXXX unicode 逃逸字元會直接抓壞
# （例如 ["始丘", "始丘"] 會被抓成沒解碼、夾雜逗號引號的亂碼）。
_JSON_ARRAY_SHAPE_RE = re.compile(r"^\s*\[.*\]\s*$", re.DOTALL)
_TEXT_KEYS = ("text", "content", "value", "caption")

# chandra-ocr-2 也常用 HTML 標籤包住文字（例如 <div data-bbox="..."><p>字</p></div>、
# 南<br/>南<br/>南），標籤本身不是辨識結果的一部分，只有偵測到角括號才觸發，避免誤傷
# 正常文字裡本來就有的「<」「>」符號。
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# 部分模型（尤其量化 GGUF，例如 glm_ocr）在正確答案後面會接著把系統 prompt 整段複誦
# 回來（有時還夾雜繁簡體切換的改寫版本）。這些片語只會出現在 prompt 裡，只要在輸出
# 中間偵測到，代表從這裡開始都是複誦雜訊，只保留它前面的部分。
_PROMPT_ECHO_RE = re.compile(
    "(" + "|".join([
        "你是一個專業的OCR", "你是一个专业的OCR",
        "OCR文本辨識引擎", "OCR文字辨識引擎", "OCR文本辨识引擎", "OCR文字辨识引擎",
        "請仔細觀察圖片", "请仔细观察图片",
        "不要加任何解釋", "不要加任何解释",
    ]) + ")"
)

# 推理模型（例如 chandra-ocr-2）習慣先輸出一大段 <think>...</think> 推理過程，
# 正式答案接在 </think> 之後。用貪婪比對抓到「最後一個」</think>，只保留它後面的內容。
_THINK_TAG_RE = re.compile(r"^.*</think>", re.DOTALL)

# 部分模型（尤其量化 GGUF）偶爾會卡入重複輸出迴圈（例如吐幾百個重複字元），
# 把連續 4 次以上的同一字元收窄成 1 次，避免拖爆 ANLS/F1。
_REPEAT_CHAR_RE = re.compile(r"(.)\1{3,}")

# 安全上限：就算前面的規則都沒攔到，也不讓失控輸出把長度撐爆
MAX_PREDICTION_LENGTH = 200


def _extract_json_array_text(t):
    """t 本身就是 JSON 陣列時嘗試解析：
    - 字串陣列（例如 ["始丘", "始丘"]）：真正用 json.loads 解碼（含 unicode 逃逸字元）後接起來。
    - dict 陣列且有 text/content/value/caption 其中一個欄位：取出來接起來。
    - 解析失敗，或是 dict 陣列但完全沒有上述任何欄位（例如純 bbox/label 版面框，
      沒有實際文字內容）：回傳 None，保留原始字串不動，讓使用者在報告裡還看得到
      原始輸出方便除錯，而不是直接清空成看不出所以然的空字串。
    """
    if not _JSON_ARRAY_SHAPE_RE.match(t):
        return None
    try:
        data = json.loads(t)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, list) or not data:
        return None
    if all(isinstance(d, str) for d in data):
        return "".join(data)
    if not all(isinstance(d, dict) for d in data):
        return None
    texts = [
        str(d[k]) for d in data for k in _TEXT_KEYS
        if isinstance(d.get(k), str) and d[k]
    ]
    return "".join(texts) if texts else None


def clean_prediction(text):
    t = (text or "").strip()
    if "</think>" in t:
        t = _THINK_TAG_RE.sub("", t).strip()

    m_echo = _PROMPT_ECHO_RE.search(t)
    if m_echo:
        t = t[:m_echo.start()].strip()

    extracted = _extract_json_array_text(t)
    if extracted is not None:
        t = extracted

    t = _CODE_FENCE_RE.sub("", t).strip()
    if "<" in t and ">" in t:
        t = _HTML_TAG_RE.sub("", t).strip()
    t = _LABEL_PREFIX_RE.sub("", t).strip()
    if len(t) >= 2 and t[0] in _WRAP_QUOTES and t[-1] in _WRAP_QUOTES_END:
        t = t[1:-1].strip()
    # 只去除模型可能加的換行/tab（這些在單行句子裡一定不是有意義的內容），
    # 但保留空格——部分句子的正確答案本身就用空格分隔多個詞組。
    t = re.sub(r"[\r\n\t]+", "", t)
    t = _REPEAT_CHAR_RE.sub(r"\1", t)
    if len(t) > MAX_PREDICTION_LENGTH:
        t = t[:MAX_PREDICTION_LENGTH]
    return t
