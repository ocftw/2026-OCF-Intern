# GLM-OCR 單一變量消融實驗

這個資料夾保存 Sixhuang 與合作者評測流程對齊後的 one-factor-at-a-time（OFAT）消融實驗。每個實驗只改一項設定，其他條件保持 baseline 不變，用來量出該變因對 TC-STR 3,706 筆 OCR 結果的影響。

已完成的結果摘要請先看 [`RESULTS.md`](RESULTS.md)，完整互動式表格與代表案例在 [`results/ablation_report.html`](results/ablation_report.html)。

## 實驗內容

| 實驗 | 相對 baseline 唯一改動 |
|---|---|
| `baseline` | `/api/generate`、長 prompt、`repeat_penalty=1.6`、`num_predict=80` |
| `endpoint_chat` | 改用 `/api/chat` |
| `short_prompt` | 改用 Sixhuang 的短 prompt |
| `no_repeat_penalty` | 不傳入 `repeat_penalty=1.6`，回到 Ollama／模型預設值 |
| `num_predict_25` | 將輸出上限由 80 改成 25 |
| `minimal_postprocess` | baseline raw response 改用 Sixhuang 最小後處理，離線重算 |
| `sixhuang_scorer` | baseline processed prediction 改用 Sixhuang scorer，離線重算 |

## 資料夾內容

| 路徑 | 用途 |
|---|---|
| `run_ablation.py` | 主程式：呼叫 Ollama、逐筆寫入 CSV、續跑、離線衍生實驗及產生 HTML 報告 |
| `start_ablation.sh` | 以 `nohup` 在背景啟動完整實驗 |
| `status_ablation.sh` | 查看 PID、狀態 JSON 與最新 log |
| `config.py` | baseline prompt 與 TC-STR 路徑設定 |
| `postprocess.py` | aligned 後處理規則 |
| `metrics.py` | EM、CM、ANLS、字元級 F1 定義 |
| `results/summary.json` | 七組結果的彙總數字 |
| `results/run_manifest.json` | 本次模型、prompt、options 與資料 fingerprint |
| `results/dataset_manifest.json` | 3,706 張圖片的順序、ground truth 與 SHA-256 |
| `results/ablation_report.html` | 完整報告與差異案例 |

逐筆 CSV 約 17 MB，包含大量模型 raw response，未放入 Git；執行腳本可在本地重新產生相同欄位與報告。

## 環境準備

需求：

- Python 3.10+
- Ollama 已啟動，預設位址為 `http://localhost:11434`
- 模型 `ggml-org/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf:latest`
- TC-STR 的 `test_labels.txt` 與 `images/`

安裝 Python 套件：

```bash
cd ablation_experiment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

預設目錄結構為 repo 與 `TC-STR` 相鄰：

```text
workspace/
├── 2026-OCF-Intern/
│   └── ablation_experiment/
└── TC-STR/
    ├── test_labels.txt
    └── images/
```

如果資料集在其他位置，執行前設定：

```bash
export TC_STR_DIR=/absolute/path/to/TC-STR
```

## 執行方式

先跑 10 筆 smoke test，並使用獨立輸出目錄：

```bash
cd ablation_experiment
.venv/bin/python run_ablation.py --limit 10 --output-dir output_smoke
```

確認無誤後，在背景跑完整 3,706 筆：

```bash
./start_ablation.sh
```

完整實驗共有 5 組需要推論的版本，合計 18,530 次 Ollama request；另外 2 組由 baseline 結果離線重算。

查看狀態或即時 log：

```bash
./status_ablation.sh
tail -f output/ablation.log
```

完成後報告位於：

```text
output/ablation_report.html
```

程序中斷後再次執行 `./start_ablation.sh`，會跳過已成功的 index，只重試未完成或失敗的樣本。如果模型、資料、參數或樣本數不同，runner 會拒絕混用舊輸出，此時請指定新的 `--output-dir`。

## 解讀限制

- 數字只代表目前 GLM-OCR Q8、TC-STR 與 Ollama 環境，不代表其他模型或資料集也會有相同結果。
- OFAT 測量的是 baseline 條件下的單一變因主效應，不能把各列 delta 直接相加。
- Prompt 與後處理、`num_predict` 與後處理、prompt 與 repeat penalty 仍可能存在交互作用。
- 未指定 `repeat_penalty` 代表使用 Ollama／模型預設值，不等同於已證明其數值為 1.0。
- CM 只檢查 prediction 是否包含 ground truth；答案藏在大量多餘文字中仍可能得分，不應單獨用 CM 判斷 OCR 品質。
