> **階段定位**：第一階段，研究起點｜作者 [@trickster-2005](https://github.com/trickster-2005)｜2026-07-15
>
> 本目錄為整個實習最早的評測程式，亦為後續所有實驗的起點。程式建立了自 Ollama VLM 呼叫、
> 後處理、四項指標評分至 HTML 報告的完整流程，並於下方〈為什麼跟別人的結果可能不一樣〉
> 一節提出本次實習最關鍵的研究問題：同一模型與同一份資料，何以不同人執行所得分數差異
> 顯著。
>
> 該問題後續由 [`../ablation_experiment/`](../ablation_experiment/) 以 3,706 筆資料、
> 7 組單變因實驗正面回答，結論證實此處的排序判斷正確：後處理與 prompt 的影響大於模型
> 本身。
>
> **結果數據**：`results/report.html`，18.7 MB，已進版控，可直接以瀏覽器開啟。逐筆的
> `raw_results.csv` 未進版控，須重新執行產生。
>
> 完整脈絡見[根目錄 README](../README.md)。

---

# TC-STR OCR 模型評估腳本

用 Ollama 上的三個模型（`qwen2.5vl:3b`、`chandra-ocr-2:latest`、
`ggml-org/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf:latest`）跑 TC-STR 場景文字辨識資料集，
用 EM / CM(Containment) / ANLS / F1 四個指標評分，並產生一頁式 HTML 報告。

## 檔案說明

| 檔案 | 做什麼 |
|---|---|
| [`config.py`](config.py) | **所有設定集中在這裡**：`MODELS` 是三個模型的清單（Ollama tag、以及個別的 `options` 覆蓋參數，例如 glm_ocr 用的 `repeat_penalty`/`num_predict`）；**`OCR_PROMPT` 就是下給 VLM 的提示詞**；`ANLS_THRESHOLD`、`TEXT_LEN_BUCKETS` 是評分/報告用的參數；還有資料集路徑（`TEST_LABELS`/`TRAIN_LABELS`/`IMAGES_DIR`）跟 `OLLAMA_HOST` |
| [`metrics.py`](metrics.py) | EM / CM(Containment Match) / ANLS / F1 四個指標的實際計算公式，`score_all()` 是入口 |
| [`ollama_client.py`](ollama_client.py) | 封裝呼叫 Ollama `/api/generate` 的細節：把圖片轉 base64、組 payload、送出 prompt + 圖片、回傳模型的原始文字回應 |
| [`postprocess.py`](postprocess.py) | **清理模型原始輸出的雜訊**：去掉模型把系統 prompt 複誦回來的部分、`<think>` 推理標籤、HTML 標籤（`<div>`/`<p>`/`<br/>`）、JSON 陣列格式（`["字"]` 或 `[{"text":"字"}]`）、code fence、重複字元迴圈等，把「答案本身」跟「格式雜訊」分開。這一步直接影響評分結果，細節見下面「為什麼結果可能跟別人不同」 |
| [`run_eval.py`](run_eval.py) | **主要執行腳本**：讀 `test_labels.txt` → 依序呼叫三個模型（圖片 + `OCR_PROMPT`）→ 用 `postprocess.py` 清理輸出 → 用 `metrics.py` 算分 → 逐行寫進 `results/raw_results.csv`（同時保留清理前/後兩個版本：`prediction_raw` 跟 `prediction`） |
| [`build_report.py`](build_report.py) | 讀取 `results/raw_results.csv`，彙總統計、畫 SVG 長條圖，輸出成自包含的 `results/report.html`（圖片以 base64 內嵌） |
| [`run_until_done.sh`](run_until_done.sh) | 長時間背景執行用：依序把每個模型跑到目標筆數，斷線/Ollama掛掉會自動重試，適合用 tmux 丟著跑整個資料集 |
| `requirements.txt` | Python 套件依賴（`requests`、`pandas`、`pillow`） |
| `results/` | 輸出資料夾：`raw_results.csv`（逐筆評分結果，含原始+清理後輸出）、`report.html`（最終報告）、`RUN_STATUS.txt`/`logs/`（`run_until_done.sh` 的執行進度，不進版控） |

## 1. 複製到 server

在**已經跑著 Ollama、三個模型都 `ollama pull` 好**的機器上執行（例如你目前的
`ubuntu@ip-172-31-44-64`）。把整個 `eval/` 資料夾複製過去，放在 `TC-STR/` 底下、
跟 `images/`、`test_labels.txt` 同一層：

```
~/Test_OCR/TC-STR/
  images/
  test_labels.txt
  train_labels.txt
  eval/            <- 複製這個資料夾過來
```

在本機（Windows）可以用 scp：

```powershell
scp -r "C:\Users\user\Desktop\TC-STR-eval\eval" ubuntu@<server-ip>:~/Test_OCR/TC-STR/
```

## 2. 安裝套件、確認模型

```bash
cd ~/Test_OCR/TC-STR/eval
pip install -r requirements.txt
```

確認 Ollama 已經在跑（預設 `http://localhost:11434`），並用 `ollama list` 確認看得到：

- `qwen2.5vl:3b`
- `chandra-ocr-2:latest`
- `ggml-org/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf:latest`

如果實際 tag 名稱有出入，改 [`config.py`](config.py) 裡的 `MODELS` 就好，其他腳本不用改。

## 3. 先跑小量測試

```bash
python run_eval.py --limit 10
```

確認三個模型都能正常呼叫（沒有連線錯誤、輸出看起來合理）之後，再考慮跑大量。

```bash
# 只跑其中一個模型
python run_eval.py --limit 10 --models qwen

# 中斷了想接續跑
python run_eval.py --limit 10 --resume

# 跑完整個 test_labels.txt
python run_eval.py --limit 0

# 改跑 train split
python run_eval.py --labels ../train_labels.txt --limit 50
```

結果會逐行寫入 `results/raw_results.csv`（每張圖 x 每個模型一行，含 EM/CM/ANLS/F1 四個指標）。

### 長時間、不用顧的正式跑法

如果要跑大量資料、擔心中途斷線或 Ollama 不穩，用 `run_until_done.sh`（會自動重試、
記錄進度，可以用 tmux 丟到背景跑）：

```bash
chmod +x run_until_done.sh
# 先打開腳本改 LIMIT（想跑幾筆成功，0 = 全部）跟 MODELS_TO_RUN
tmux new -s tcstreval
./run_until_done.sh
# Ctrl+B 再按 D 離開 tmux，安全斷線，之後可以 tmux attach -t tcstreval 接回來看
```

回來看進度：

```bash
cat results/RUN_STATUS.txt
tail -f results/logs/until_done_*.log
```

## 4. 產生一頁式報告

```bash
python build_report.py
```

會在 `results/report.html` 產生一個獨立、自包含的網頁（圖片以 base64 內嵌，不用另外開
server），包含：

- 各模型總覽卡片（EM / CM / ANLS / F1 / 平均延遲）
- 文字長度、圖片類別（從檔名前綴自動偵測，例如 `billboard`）對 ANLS 的影響長條圖
- 可篩選（依模型）／排序（依 ANLS）／搜尋文字內容的圖片明細，看哪張出錯、
  預測跟 ground truth 的差異用顏色標示

把 `results/report.html` 複製回本機，直接用瀏覽器打開就好：

```powershell
scp ubuntu@<server-ip>:~/Test_OCR/TC-STR/eval/results/report.html .
```

## 指標定義

| 指標 | 定義 |
|---|---|
| EM | 預測文字跟 ground truth 完全一致才計 1 分 |
| CM | Containment Match：預測文字完整包含 ground truth 就計 1 分 |
| ANLS | 1 − 正規化編輯距離，低於 0.5 門檻當 0 分（DocVQA 慣例） |
| F1 | 字元級 precision/recall 的調和平均 |

## 為什麼跟別人的結果可能不一樣

同一個模型、同一份 TC-STR 資料集，不同人跑出來的 EM/CM/ANLS/F1 還是可能差距很大
（例如同一個 `glm_ocr`，實測過 ANLS 從 0% 到 55% 都出現過）。常見原因，按影響程度排序：

1. **後處理清理程度不同（影響最大）**：[`postprocess.py`](postprocess.py) 會把模型輸出裡的
   格式雜訊（複誦的 prompt、HTML 標籤、JSON 包裝、重複字元迴圈）清掉才拿去算分。如果對方是直接拿
   Ollama 的原始回應去比對，同一個模型的 EM/ANLS 會被這些雜訊拖得非常低，但 CM/F1 相對沒差那麼多
   （因為 CM 只看有沒有包含、F1 只看字元重疊，對雜訊比較不敏感，EM 要求完全一致、ANLS 對編輯距離
   敏感，雜訊一多分數就崩）。**如果兩邊分數呈現「EM/ANLS 差很多、CM/F1 差不多」的形狀，八九不離十
   是這個原因**。`raw_results.csv` 裡同時留了 `prediction_raw`（清理前）跟 `prediction`（清理後）
   兩欄，可以直接對照。
2. **Prompt 措辭不同**：[`config.py`](config.py) 的 `OCR_PROMPT` 字句會直接影響模型輸出的風格
   （例如要不要輸出 HTML/JSON、要不要加解釋）。用詞不同，模型行為可能整個不一樣。
3. **Ollama 呼叫參數不同**：`temperature`、`repeat_penalty`、`num_predict` 這些在 `config.py` 的
   `MODELS[*]["options"]` 裡設定。`glm_ocr` 這個量化模型特別容易陷入重複輸出迴圈，這裡已經調高
   `repeat_penalty`；如果對方用預設參數，同一個模型的行為可能差很多。
4. **指標實作細節不同**：ANLS 的門檻是不是都用 0.5？F1 是字元級還是詞級（中文沒有分詞邊界，這裡固定
   用字元級）？EM/CM 有沒有做大小寫、全半形、標點正規化？這些沒有統一標準，不同實作結果本來就會有落差。
5. **評測樣本數/split 不同**：確認兩邊是不是都跑完整的 `test_labels.txt`（3706 筆），不是子集合，也
   不是誤用了 `train_labels.txt`。
6. **模型量化版本/推論後端版本不同**：即使 Ollama tag 名稱一樣，不同時間 pull 下來的權重、不同版本的
   llama.cpp 後端，數值計算細節仍可能有些微差異（通常影響不大，但無法完全排除）。

要精確定位差異，最快的方法是拿幾筆「一邊算對、一邊算錯」的同一張圖，比對雙方的 `prediction_raw`
（原始輸出）是否相同——如果原始輸出一樣、只是清理後的 `prediction` 不同，就是第 1 點；如果連
`prediction_raw` 都不一樣，就要往第 2、3、6 點找。

## 注意事項

- `glm_ocr`（GLM-OCR GGUF）先前實測容易陷入重複輸出迴圈，`config.py` 裡已經加了
  `repeat_penalty` / `num_predict` 限制；如果實測發現正確答案常被截斷，調整
  `config.py` 裡 `MODELS` 那個模型的 `options`。
- 「圖片類別對 ANLS 的影響」那張圖只有在檔名前綴偵測到 2 種以上類別時才會出現
  （例如全部都是 `billboard_xxxxx` 就不會畫這張圖）。
