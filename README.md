# TC-STR OCR 模型評估腳本

用 Ollama 上的三個模型（`qwen2.5vl:3b`、`chandra-ocr-2:latest`、
`ggml-org/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf:latest`）跑 TC-STR 場景文字辨識資料集，
用 EM / CM(Containment) / ANLS / F1 四個指標評分，並產生一頁式 HTML 報告。

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

如果實際 tag 名稱有出入，改 [`config.py`](eval/config.py) 裡的 `MODELS` 就好，其他腳本不用改。

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

## 注意事項

- `glm_ocr`（GLM-OCR GGUF）先前實測容易陷入重複輸出迴圈，`config.py` 裡已經加了
  `repeat_penalty` / `num_predict` 限制；如果實測發現正確答案常被截斷，調整
  `config.py` 裡 `MODELS` 那個模型的 `options`。
- 「圖片類別對 ANLS 的影響」那張圖只有在檔名前綴偵測到 2 種以上類別時才會出現
  （例如全部都是 `billboard_xxxxx` 就不會畫這張圖）。
