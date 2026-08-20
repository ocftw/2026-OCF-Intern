> **成果定位**｜第 2 階段・對照組｜作者 [@hyslchs](https://github.com/hyslchs)｜2026-07-16
>
> 這是與 [`../eval/`](../eval/) **各自獨立實作**的第二套 TC-STR 跑分程式，另外納入
> Hugging Face 繁中合成資料集。它的價值不只在於多跑了一組數字——正因為兩套流程是
> 獨立寫的，它們在三個地方出現了差異：**短 prompt**、**最小後處理**、**不同的
> scorer 定義**（忽略標點空白、CM 採雙向判定）。
>
> 這三項差異後來被直接當成 [`../ablation_experiment/`](../ablation_experiment/) 的
> 三個受測變因，量化出各自對分數的影響。所以這個目錄實際上是消融實驗的「另一個
> 端點」，兩者要對照著看。
>
> **結果數據**：未進版控（`results/` 在 `.gitignore` 中），需依下方步驟重跑產生。
> 若只是想看這套設定的分數影響，直接看消融實驗的
> [`RESULTS.md`](../ablation_experiment/RESULTS.md) 更快。
>
> 完整脈絡見 [repo 根目錄 README](../README.md)。

---

# Sixhuang OCR Benchmark

這是 Sixhuang 使用 Ollama 執行繁體中文 OCR 跑分的程式。主程式會評估兩個模型：

- `qwen2.5vl:3b`
- GLM-OCR Q8_0 GGUF

評測資料包含 TC-STR test split，以及 Hugging Face
`ZihCiLin/traditional-chinese-ocr-synthetic` 的 `test_random`、`test_semantic` split。
程式會保存逐筆預測、EM、CM、ANLS、字元級 F1，並產生 HTML 報告。

## 目錄結構

```text
Sixhuang/
├── benchmark.py                 # 主跑分、評分與 HTML 報告程式
├── Modelfile_glm                # 本地 GLM-OCR GGUF 的 Ollama 設定
├── prompts/ocr_zh.txt           # 正式跑分使用的 prompt
├── requirements.txt             # Python 依賴
└── scripts/
    ├── download_glm.py          # 下載 GLM-OCR GGUF 與 mmproj
    ├── download_tc_str.py       # 下載及解壓 TC-STR
    ├── test_inference.py        # 小量比較 prompt 與模型輸出
    ├── test_hf_dataset.py       # 測試 Hugging Face split 載入
    └── test_hf_dataset_fast.py  # 直接指定 parquet 的載入測試
```

模型、dataset、執行結果與 Python virtual environment 不納入 Git。

## 環境需求

- Python 3.10 以上
- Ollama
- 足夠的磁碟空間存放約 1.4 GB 的 GLM-OCR GGUF 與 mmproj

在 repository 根目錄執行：

```bash
cd Sixhuang
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 準備資料集

```bash
python3 scripts/download_tc_str.py
```

預設會下載並解壓到：

```text
Sixhuang/data/TC-STR/
```

Hugging Face synthetic dataset 會在第一次跑 benchmark 時由 `datasets` 自動下載。

## 準備模型

先準備 Qwen：

```bash
ollama pull qwen2.5vl:3b
```

下載 GLM-OCR GGUF：

```bash
python3 scripts/download_glm.py
```

接著在 `Sixhuang/` 目錄建立 Ollama model：

```bash
ollama create glm-ocr-sixhuang -f Modelfile_glm
```

確認模型可用：

```bash
ollama list
```

## 執行跑分

先各跑 5 筆確認環境：

```bash
python3 benchmark.py \
  --limit 5 \
  --glm-model glm-ocr-sixhuang
```

跑完整資料集時使用 `--limit -1`：

```bash
python3 benchmark.py \
  --limit -1 \
  --glm-model glm-ocr-sixhuang
```

如果 Ollama 不在預設的 `http://localhost:11434`：

```bash
python3 benchmark.py \
  --limit 5 \
  --ollama-host http://HOST:11434 \
  --glm-model glm-ocr-sixhuang
```

也可以使用以下參數覆蓋預設值：

```text
--tc-str-dir    TC-STR 資料夾
--output-dir    結果輸出資料夾
--prompt-file   UTF-8 prompt 檔
--qwen-model    Qwen 的 Ollama tag
--glm-model     GLM-OCR 的 Ollama tag
```

預設輸出到 `Sixhuang/results/`：

- `results.json`：逐筆 raw prediction 與評分
- `report.html`：HTML 報告
- `report_images/`：報告使用的圖片
- `checkpoint.json`：未完成時的續跑進度；成功完成後會刪除

## 正式 prompt

[`prompts/ocr_zh.txt`](prompts/ocr_zh.txt)：

```text
請辨識圖片中的文字。只輸出辨識出的文字，不要有任何其他贅字、標點、說明或格式。
```

## 推論設定

程式使用 Ollama `/api/chat`，每張圖建立一個獨立 request，`stream=false`。

| 模型 | options |
|---|---|
| Qwen | `temperature=0.0`, `top_p=0.00001`, `top_k=1` |
| GLM-OCR | `num_predict=25` |

GLM 的 `Modelfile_glm` 另外設定 `temperature=0`、`top_p=0.00001`、`top_k=1`
及 chat template。模型 tag 中需包含 `glm`，主程式才會套用 GLM 的 `num_predict=25`。

## 評分方式

評分前會移除空白、英文標點與程式列出的中文標點。

- EM：清理後完全一致為 1，否則為 0。
- CM：prediction 包含 ground truth，或 ground truth 包含 prediction，即為 1。
- ANLS：`1 - edit_distance / max_length`，低於 0.5 時歸零。
- F1：以字元出現次數計算 precision、recall 與 F1。

## 續跑注意事項

`checkpoint.json` 只記錄預測陣列。若更換 prompt、模型、資料順序或 `--limit`，請先刪除舊的
`checkpoint.json`，避免把不同設定的結果混在同一次實驗中。

## 資料與模型來源

- TC-STR：<https://github.com/esun-ai/traditional-chinese-text-recogn-dataset>
- HF synthetic dataset：<https://huggingface.co/datasets/ZihCiLin/traditional-chinese-ocr-synthetic>
- GLM-OCR GGUF：<https://huggingface.co/ggml-org/GLM-OCR-GGUF>

請依各上游專案的授權條款使用資料與模型。
