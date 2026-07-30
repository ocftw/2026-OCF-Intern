# TC-STR × 8 Ollama models

本目錄是 TC-STR 官方 test split（3,706 筆）的可重現 runner。程式不會自動 pull
模型、猜測 tag、修改 Ollama/Docker/systemd/KV cache 設定，也不會由 smoke 自動
啟動 full run。

## 固定路徑

- Dataset：`/mnt/nvme/datasets/TC-STR/`（stop/start 後消失，可重抓）
- Scratch：`/mnt/nvme/scratch/tc_str/`
- Durable run：`/opt/ocf-ai/outputs/tc_str/<run_id>/`

SQLite WAL (`results.sqlite`) 是逐題 checkpoint 的唯一真實來源。每次 request attempt
立即寫入；重試耗盡後，每一題都會形成不可變的 terminal result。成功答案、輸出異常、
空輸出與 request failure 都保留並計分，且 signature 相符時在 resume 跳過；attempt
history 仍完整保留。

## 共同使用條件下的實際失敗

依使用者於 2026-07-29 的核准，空輸出、純控制標籤、prompt echo、拒答、Markdown／JSON
包裝、長篇解釋、重複迴圈、thinking 污染、內容無關及碰到 `num_predict` 等現象，視為
模型在相同 prompt 與 generation options 下的實際表現。它們會：

- 永久保存 raw response、錯誤、HTTP 與 token metadata；
- 使用完全相同的 primary normalization 與 scorer（request failure／空輸出以空 prediction
  計零分）；
- 在 smoke 與正式報告中標示 anomaly，但不因異常率阻擋整套 Benchmark。

硬阻擋只保留資料或 signature 不一致、exact model／vision 失敗、模型完全無 HTTP 200
回應、或不是精確 `100% GPU` 等工程性條件。共同 prompt、原始圖片與生成參數不會針對
任何模型調整。

## GLM OCR BF16 completion metadata 例外

依使用者於 2026-07-29 的決定，原先的官方 `glm-ocr:bf16` 保留在評測中。這個模型在
Ollama `/api/generate` 使用共同的 `stream=false`、`think=false` 設定時，會回傳
HTTP 200 與有效 OCR `response`，但可能同時回傳 `done=false`，且不提供
`done_reason`、`prompt_eval_count`、`eval_count`。

因此只有此 exact model id 採用已記錄的 completion policy 例外：HTTP 200 且正式
`response` 非空即可保存與評分。共同 prompt、原始圖片、generation options、
normalization 與 scorer 完全不變。缺少的欄位保留 `null`；報告以
`truncation_unknown` 標示無法自動判斷截斷，token 使用與自動截斷診斷不可和其他模型
等量比較。完整決策與影響見 `PROTOCOL_EXCEPTIONS.md`。

## 模型確認

先編輯 `models.yaml`，只填主管確認的 exact local Ollama tag 與量化。空 tag、未安裝、
非 vision、量化不符、warm-up 失敗或 `ollama ps` 不是精確 `100% GPU` 都會產生
`BLOCKED` report，且不會進入 smoke。工具絕不執行 `ollama pull`。

## 階段 A：preflight 與 smoke

```bash
cd /home/hyslchs/20260729_TCSTR_and_omindoc/2026-OCF-Intern/TC-STR_and_OminDoc/TC_STR
./run_smoke.sh
```

僅執行 preflight：

```bash
./run_smoke.sh --preflight-only
```

全模型 smoke 成功後狀態先停在 `AWAITING_AI_MANUAL_REVIEW`。逐筆人工檢視 160 筆
raw response，建立帶相同 signature 的 review JSON，再明確執行：

```bash
python3 -m tc_str_bench.cli review-smoke \
  --run-id <run_id> --review-file <人工逐筆檢視.json>
```

Review JSON 必須有 160 個唯一 `(model_id, sample_index)`，verdict 僅可為
`NORMAL`、`MODEL_ABILITY_ERROR` 或 `PIPELINE_ANOMALY`。人工確認到的
`PIPELINE_ANOMALY` 會列入警告與逐題結果，而不再成為整套評測 blocker。這個檔案不會
自動產生。

## 階段 B：由使用者在 tmux 親自啟動

只有最新 run 的 smoke report 是 `READY_FOR_FULL_RUN`，且重新 preflight 得到完全相同
的 model digest、dataset fingerprint、prompt/options、Ollama/source signature 時，
以下指令才會執行；否則以非 0 exit code 停止：

```bash
./run_full.sh
```

同一指令重跑會從 EBS SQLite 自動續跑。不要把 `nohup` 寫進 runner；請在 tmux 執行。

## 狀態與重建報告

```bash
./status.sh
./build_report.sh --run-id <run_id>
./run_tests.sh
```

每個 run 會輸出 manifest、preflight/smoke/main HTML、SQLite、long/wide CSV、JSONL、
summary、metadata、config、status、完整 log 與 EBS 內可持久讀取的圖片 asset。
HTML 對 ground truth 與模型輸出做 escape，raw response 只摺疊、不刪除。
