> **成果定位**｜★ 第 3 階段・**本 repo 最重要的成果**｜作者 [@hyslchs](https://github.com/hyslchs)｜2026-07-21
>
> 這裡回答了 [`../eval/`](../eval/) 提出的核心問題：同一個模型、同一份資料，為什麼
> 不同人跑出的分數差這麼多？做法是把兩套獨立流程（`../eval/` 與 `../Sixhuang/`）
> 對齊成共同 baseline，然後每次**只改一項設定**，量測該變因的主效應。
>
> **規模**：GLM-OCR Q8_0 × TC-STR 3,706 筆 × 7 組 variant，其中 5 組需要推論，
> 合計 **18,530 次 Ollama request**，全部完成、API error 數為 0。
>
> **最重要的三個數字**：
> - 不指定 `repeat_penalty=1.6` → **EM 47.06% → 64.84%（+17.78 pp）**，淨增 659 題全對
> - 換成短 prompt → **EM/ANLS 歸零，但 CM 反而升到 79.06%**（字認出來了，格式失控）
> - `/api/generate` vs `/api/chat` → **3,706/3,706 逐字相同**，指標完全一致
>
> **結果數據**：✅ 完整保存在 [`results/`](results/)。逐筆 CSV（約 17 MB）未進版控，
> 但 [`results/dataset_manifest.json`](results/dataset_manifest.json) 記錄了 3,706 張
> 圖片的順序、ground truth 與 SHA-256，重跑可得到等價輸出。
>
> 這些發現直接決定了後續 [`../benchmark_suite/`](../benchmark_suite/) 與
> [`../TC-STR_and_OminDoc/`](../TC-STR_and_OminDoc/) 的設計。完整脈絡見
> [repo 根目錄 README](../README.md)。

---

# GLM-OCR 單一變量消融實驗

## 完整實驗結果

本次使用相同的 GLM-OCR Q8 模型與 TC-STR 3,706 筆測試資料，以合作者 aligned pipeline 為 baseline，每次只修改一個變因。所有需要推論的 variant 均完成 3,706 筆，API error 數為 0。

### 主要結論

- 影響最大的三個因素是 prompt、`repeat_penalty` 與後處理。
- `/api/generate` 與 `/api/chat` 的 3,706 筆 processed prediction 完全相同，endpoint 不是這次分差來源。
- 不指定 `repeat_penalty=1.6`、回到 Ollama／模型預設值後，EM 從 47.06% 提升至 64.84%，是本次表現最好的設定。
- 短 prompt 並非完全無法辨識，而是容易在答案後輸出解釋、Markdown、英文分析與 prompt 複誦，導致 EM／ANLS 歸零。
- aligned 後處理修改了 baseline 的 3,704／3,706 筆 raw response，是目前評測流程取得合理 EM／ANLS 的關鍵。
- Sixhuang scorer 會忽略空白與標點並使用雙向 CM，因此相同 prediction 會得到稍高分；這是評分規則差異，不是模型能力提升。

### 分數總覽

| 實驗 | EM | 相較 baseline | CM | ANLS | 相較 baseline | F1 | 平均延遲 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aligned baseline | 47.06% | — | 76.52% | 54.54% | — | 59.17% | 0.602 s |
| Endpoint：`/api/chat` | 47.06% | +0.00 pp | 76.52% | 54.54% | +0.00 pp | 59.17% | 0.582 s |
| Sixhuang short prompt | 0.00% | -47.06 pp | 79.06% | 0.00% | -54.54 pp | 13.56% | 0.594 s |
| 不指定 `repeat_penalty=1.6` | **64.84%** | **+17.78 pp** | **82.46%** | **72.08%** | **+17.53 pp** | **76.62%** | 0.583 s |
| `num_predict=25` | 46.03% | -1.03 pp | 76.34% | 53.29% | -1.25 pp | 65.07% | **0.293 s** |
| Sixhuang minimal postprocess | 0.00% | -47.06 pp | 76.79% | 0.00% | -54.54 pp | 6.88% | 離線重算 |
| Sixhuang scorer | 47.87% | +0.81 pp | 77.63% | 55.02% | +0.48 pp | 59.96% | 離線重算 |

`pp` 表示百分點。完整 HTML 報告與代表案例請看 [`results/ablation_report.html`](results/ablation_report.html)，機器可讀的完整彙總在 [`results/summary.json`](results/summary.json)。

### 各變因的影響

#### 1. API endpoint

- 唯一改動：`/api/generate` 改成 `/api/chat`。
- 3,706／3,706 筆 prediction 與 baseline 完全相同。
- EM、CM、ANLS、F1 全部沒有變化。
- 0.020 秒的平均延遲差距可能受模型暖機與系統負載影響，不足以證明 `/api/chat` 一定比較快。

#### 2. Prompt

- 唯一改動：171 字 aligned prompt 改成 Sixhuang 39 字短 prompt。
- 只有 2／3,706 筆 prediction 與 baseline 相同。
- 1,744 題由 EM 正確變成錯誤，沒有任何題目由錯轉對。
- 模型經常先輸出 OCR 答案，再接著產生 Markdown、解釋、翻譯、英文分析或複誦指令。
- CM 反而增加 2.54 個百分點，是因為正確答案仍藏在長輸出中；這不代表 OCR 品質提升。
- 現有 aligned 後處理主要針對長 prompt 的複誦模式設計，因此這一列也反映短 prompt 與既有後處理不相容的交互作用。

#### 3. Repeat penalty

- 唯一改動：不傳入 `repeat_penalty=1.6`，其他設定維持不變。
- EM 增加 17.78 個百分點，ANLS 增加 17.53 個百分點，F1 增加 17.44 個百分點。
- 779 題由 EM 錯誤變成正確，120 題由正確變成錯誤，淨增加 659 題完全正確。
- 這表示 1.6 對目前 GLM-OCR Q8 可能過高，會干擾本來合理的重複 token 或改變字元選擇。
- 「不傳入」代表回到 Ollama／模型預設值，不等同於已證明 repeat penalty 為 1.0 或完全關閉。

#### 4. 輸出長度上限

- 唯一改動：`num_predict=80` 改成 25。
- 平均延遲由 0.602 秒降至 0.293 秒，約縮短 51%。
- EM 有 5 題由錯轉對、43 題由對轉錯；整體 EM 降低 1.03 個百分點。
- 部分 prompt 複誦被截成不完整片段，例如 `你是一個專業的OC`，因而躲過針對完整片語的後處理規則。
- F1 上升不代表完整答案更準確，而是輸出被截短後，多餘字元變少，字元 precision 因此提高。

#### 5. 後處理

- 不重跑模型，只把 baseline raw response 改用 Sixhuang 最小後處理。
- EM 與 ANLS 都從 baseline 降至 0%，F1 降至 6.88%。
- aligned 後處理曾修改 3,704／3,706 筆 raw response，包括清除 prompt 複誦、換行、HTML、JSON、Markdown、思考標籤及失控重複。
- 目前報告衡量的是「模型＋後處理」的系統表現，不是完全未清理的模型原始輸出能力。

#### 6. 評分規則

- 不重跑模型，也不改 prediction，只換成 Sixhuang scorer。
- 額外有 30 題被判為 EM 正確；EM 增加 0.81、CM 增加 1.11、ANLS 增加 0.48、F1 增加 0.79 個百分點。
- 分數增加來自忽略空白與標點，以及 CM 使用雙向包含，不能解讀為模型能力提升。

### 本次建議設定

在目前已測組合中，最佳結果為：

- aligned 長 prompt
- `num_predict=80`
- 不顯式指定 `repeat_penalty=1.6`
- aligned 後處理
- 團隊統一的 scorer
- `/api/generate` 或 `/api/chat` 皆可，但共同評測時仍應固定同一種 endpoint

此組合得到 EM 64.84%、CM 82.46%、ANLS 72.08%、F1 76.62%。這只代表目前模型、資料集與執行環境；下一步建議針對 repeat penalty 的預設值、1.0、1.1、1.3、1.6 做更細的 sweep，並另外測試 prompt 與後處理的交互作用。

### 解讀注意事項

- 這是 OFAT 單一變量主效應分析，各列 delta 不能直接相加。
- 正值只代表該設定在目前 baseline、模型及資料集上的結果，不代表能泛化到其他模型。
- CM 只檢查 prediction 是否包含 ground truth，答案藏在大量錯誤內容中仍可能得分，不應單獨用 CM 判斷 OCR 品質。
- `temperature=0` 仍不保證跨 Ollama 版本、推論後端或硬體逐字一致。

---

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
