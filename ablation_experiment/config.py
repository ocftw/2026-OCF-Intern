# -*- coding: utf-8 -*-
"""消融實驗共用設定：baseline prompt 與 TC-STR 資料集路徑。"""

import os

OCR_PROMPT = (
    "你是一個專業的OCR文字辨識引擎。請仔細觀察圖片，"
    "只輸出圖片中看到的文字內容（繁體中文為主，若同時有英文或數字也一併輸出），"
    "只回傳純文字本身，不要輸出HTML標籤、Markdown或JSON等任何格式化標記，"
    "不要加任何解釋、前綴或後綴、標點符號以外的內容，不要翻譯，不要加引號。"
    "如果部分文字模糊、傾斜或被遮擋，也要盡量根據字形推斷最可能的字。"
)

# 預設假設 repo 與 TC-STR 是相鄰資料夾。也可以用 TC_STR_DIR
# 指向其他位置；這只調整檔案路徑，不改變評測內容或樣本順序。
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
TC_STR_DIR = os.environ.get("TC_STR_DIR", os.path.join(WORKSPACE_ROOT, "TC-STR"))

TEST_LABELS = os.path.join(TC_STR_DIR, "test_labels.txt")
IMAGES_DIR = os.path.join(TC_STR_DIR, "images")
