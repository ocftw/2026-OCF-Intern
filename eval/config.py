# -*- coding: utf-8 -*-
"""評估腳本共用設定：模型清單、Ollama連線位置、資料集路徑"""

import os

# Ollama API 位置。在跑著 ollama 的機器本地執行就用 localhost，
# 如果想連到其他機器，改環境變數 OLLAMA_HOST 即可。
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# 三個要評估的模型。"tag" 要跟 `ollama list` 顯示的名稱一致，
# 如果實際 tag 名稱有出入，改這裡就好，不用改其他檔案。
# 每個模型可以選擇性加上 "options"，會直接合併進 Ollama /api/generate 的
# options 參數（例如針對容易跑進重複輸出迴圈的模型加 repeat_penalty / num_predict）。
MODELS = [
    {"key": "qwen", "tag": "qwen2.5vl:3b"},
    {"key": "chandra", "tag": "chandra-ocr-2:latest"},
    {
        "key": "glm_ocr",
        "tag": "ggml-org/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf:latest",
        # 這個模型系列先前實測會陷入整句重複輸出的迴圈，還會在重複時繁簡體切換、
        # 字元持續漂移。repeat_penalty 加重到 1.6、num_predict 收緊到 80 可以大幅
        # 壓縮重複/漂移的空間。TC-STR 的招牌文字通常很短，如果實測發現正確答案常常
        # 被這個上限截斷，可以調高 num_predict。
        "options": {"repeat_penalty": 1.6, "num_predict": 80},
    },
]

OCR_PROMPT = (
    "你是一個專業的OCR文字辨識引擎。請仔細觀察圖片，"
    "只輸出圖片中看到的文字內容（繁體中文為主，若同時有英文或數字也一併輸出），"
    "只回傳純文字本身，不要輸出HTML標籤、Markdown或JSON等任何格式化標記，"
    "不要加任何解釋、前綴或後綴、標點符號以外的內容，不要翻譯，不要加引號。"
    "如果部分文字模糊、傾斜或被遮擋，也要盡量根據字形推斷最可能的字。"
)

# 用絕對路徑（以本檔案位置推算），這樣無論在哪個目錄執行 python run_eval.py
# 都找得到資料，不用規定一定要在 eval/ 或專案根目錄下執行。
# 假設目錄結構是：
#   TC-STR/
#     images/
#     test_labels.txt
#     train_labels.txt
#     eval/          <- 本檔案所在位置
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)

TEST_LABELS = os.path.join(PROJECT_ROOT, "test_labels.txt")
TRAIN_LABELS = os.path.join(PROJECT_ROOT, "train_labels.txt")
DATASET_LABELS = TEST_LABELS  # 預設評估 test split，train 可用 --labels 覆蓋

IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
RAW_RESULTS_CSV = os.path.join(RESULTS_DIR, "raw_results.csv")
REPORT_HTML = os.path.join(RESULTS_DIR, "report.html")

# ANLS 的標準門檻（DocVQA 慣例）：NLS 低於這個值就當 0 分
ANLS_THRESHOLD = 0.5

# ground truth 文字長度分組區間（TC-STR 招牌文字從單字到短句都有），
# 用於報告中「文字長度對 ANLS 的影響」那張圖。區間為左閉右開 [lo, hi)。
TEXT_LEN_BUCKETS = [(0, 2, "1字"), (2, 4, "2-3字"), (4, 999, "4字以上")]
