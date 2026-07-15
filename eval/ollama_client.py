# -*- coding: utf-8 -*-
"""包裝 Ollama /api/generate 呼叫（多模態：文字 prompt + 圖片）"""

import base64
import time

import requests


def image_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_ollama(host, model, prompt, image_path, timeout=180, extra_options=None):
    """回傳 (回應文字, 耗時秒數, 錯誤訊息或None)

    extra_options: 選擇性覆蓋/補充 Ollama 的 options 參數
    （例如 {"repeat_penalty": 1.3, "num_predict": 120}）
    """
    try:
        images = [image_to_b64(image_path)]
    except OSError as e:
        return "", 0.0, f"讀取圖片失敗: {e}"

    options = {"temperature": 0}
    if extra_options:
        options.update(extra_options)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": images,
        "stream": False,
        "options": options,
    }

    start = time.time()
    try:
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        return text, time.time() - start, None
    except requests.exceptions.RequestException as e:
        return "", time.time() - start, str(e)
