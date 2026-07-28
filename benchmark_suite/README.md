# OCF VLM Benchmark Suite

這是一套可長時間無人值守、append-only 斷點續跑的 VLM 評測系統。它以固定順序讓
5 個 Ollama 模型逐一執行 OmniDocBench、TC-STR、VisTW-MCQ，共 15 組正式結果。
新系統只新增在 `benchmark_suite/`；既有 `Sixhuang/`、`eval/` 與
`ablation_experiment/` 不會被取代。

## 唯一正式啟動指令

從 repository 根目錄執行：

```bash
./benchmark_suite/launch_all.sh
```

launcher 先做快速 preflight，再以 `setsid + nohup` 啟動背景程序並用 `flock`
防止重複啟動。SSH 中斷不會送達背景 process。啟動後會印出 run ID、PID、log、
results 路徑，並預設每 30 秒顯示一次進度。按 `Ctrl-C` 只離開進度監看，不會停止
背景 benchmark。相同 effective config 的未完成 run 會自動續跑；config、prompt、
dataset revision 或 model digest 改變時 resume key 不相符，不會混用。

如果只想啟動後立即返回 shell：

```bash
./benchmark_suite/launch_all.sh --detach
```

也可以調整顯示間隔，例如每 10 秒：

```bash
./benchmark_suite/launch_all.sh --interval 10
```

查看狀態：

```bash
./benchmark_suite/status.sh
```

狀態畫面會以繁體中文顯示目前階段、模型、Benchmark、warm-up／推論步驟、單組樣本
進度、15 組完成數、最近更新與最近錯誤；不再直接輸出難讀的 Ollama pull log。
`launch_all.sh` 預設每 30 秒自動呼叫同一個狀態畫面，可用 `--interval 10` 改為每
10 秒。執行中的舊版 runner 若尚未寫入 phase，狀態會依固定模型／Benchmark 順序推定
下一組並明確標註；下次續跑後會在 warm-up 開始前精確更新。

中止時先從 `status.sh` 確認 PID，再執行 `kill -TERM <PID>`。這不會破壞已 fsync 的
JSONL；之後再次執行 `launch_all.sh` 即可續跑。不要刪除或手動編輯 prediction JSONL。

## 系統需求與事前準備

- Ubuntu/Linux、Python 3.10+、`git`、`curl`、`flock`、`setsid`
- 已安裝並啟動 Ollama，預設 `http://localhost:11434`
- Docker daemon 可由目前使用者使用
- 主 profile 的可重建 scratch 預留至少 80 GiB，持久 results 預留至少 20 GiB；
  實際模型與 Hugging Face cache 需求可能更高
- GPU/VRAM、RAM 必須足以承載所選模型與 context。所有請求 concurrency 固定為 1
- 使用者必須自行確認各模型及資料集授權／使用限制

OmniDocBench 的 CDM 需要 TeX Live、ImageMagick 與 Ghostscript；系統使用官方建議的
`ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204` Docker image 固定環境。
若 Docker/evaluator 失敗，prediction 會保留，但 scoring 明確標成 `blocked`，不會
改用自製近似分數。

### 使用 instance-local `/srv` 暫存空間

若主機的 `/srv` 是容量較大、但 stop 後會清除的 instance-local volume，可將 Ollama
models、Docker data-root、benchmark datasets 與 cache 放在 `/srv`，但不可把正式
prediction/result 放在那裡。由 repository 根目錄執行一次：

```bash
sudo ./benchmark_suite/scripts/configure_srv_scratch.sh
```

腳本會確認 `/srv` 是獨立且可寫的 mount，建立 `/srv/ocf-benchmark/`，以專用的
`ocf-srv-prepare.service` 確保開機後重建空目錄，並為 Docker/Ollama 加上 `/srv`
mount dependency。
既有 `/opt/ollama/models` 會先複製及驗證，再保留成
`/opt/ollama/models.pre-srv-backup`；不會在首次配置時刪除。確認 `ollama list`、
model digest 與 service 都正常後，才可清除備份：

```bash
sudo ./benchmark_suite/scripts/configure_srv_scratch.sh --cleanup-backup
```

`launch_all.sh` 發現 `/srv/ocf-benchmark/work` 可寫時，會自動將 data/cache 指向其中；
`benchmark_suite/runs/` 仍留在持久根磁碟。也可用
`BENCHMARK_SCRATCH_ROOT`、`BENCHMARK_DATA_DIR`、`BENCHMARK_CACHE_DIR` 明確覆寫，
後兩者優先。instance stop 後模型與資料會消失，下一次正式執行會重新下載；run manifest、
checkpoint 與結果不受影響。

Hugging Face 公開資料預設可匿名下載。為避免大量小檔觸發 Xet token endpoint 的 429，
資料準備預設設定 `HF_HUB_DISABLE_XET=1`、`HF_DOWNLOAD_MAX_WORKERS=2`，並對
429、HTTP 5xx 與 transport failure 做長退避後沿用 cache 重試。若需要較高 rate limit，
建議先以 `benchmark_suite/.venv/bin/hf auth login` 登入；也可在啟動時提供 `HF_TOKEN`。
若環境已驗證 Xet 穩定，可明確設定 `HF_HUB_DISABLE_XET=0`。

VisTW parquet 在固定 revision snapshot 下載完成後直接以 `pyarrow` 逐 batch 讀取，
不透過 Hugging Face `datasets` 的 pickle/fingerprint cache。這避免 Python 3.14 與舊版
`dill` 內部介面不相容，也保證 smoke／正式推論只使用已固定的本地資料。
dependency lock SHA256 也是 effective config identity 的一部分；lock 變更會建立新 run，
不會把不同 Python dependency environment 寫入同一份正式結果。

主 profile 資源不足時，在昂貴正式推論前會失敗。低資源替代方案是另一個完整實驗：

```bash
./benchmark_suite/run_all.sh --config benchmark_suite/configs/constrained.yaml
```

它不會在同一 run 途中自動降 `num_ctx`、`num_predict` 或換量化，結果不得和主 profile
混合。

## 模型

所有 tag 只在 `configs/experiment.yaml` 定義，亦可用環境變數覆寫。

| logical ID | 預設 Ollama tag | 備註 |
|---|---|---|
| qwen3_vl_4b | `hf.co/Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M` | Qwen3-VL |
| gemma4_e2b | `gemma4:e2b` | Gemma 4 |
| gemma4_e4b | `gemma4:e4b` | Gemma 4 |
| sea_lion_4b | `hf.co/mradermacher/Qwen-SEA-LION-v4-4B-VL-GGUF:Q4_K_M` | **第三方 mradermacher GGUF 量化** |
| smolvlm2_2_2b | `hf.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M` | Omni context 受限 |

例如 `MODEL_QWEN3_VL_4B=my-local-tag ./benchmark_suite/launch_all.sh`。若實際要測
Gemma 3n，將 YAML tag 改為 `gemma3n:e2b`、`gemma3n:e4b`，或設定
`MODEL_GEMMA4_E2B=gemma3n:e2b` 與 `MODEL_GEMMA4_E4B=gemma3n:e4b`。這會形成不同
effective config/run。

準備階段會先 pull **全部五個模型**，再透過 `/api/tags` 與 `/api/show` 保存實際
digest、quantization、template、Modelfile、parameters 與 model metadata。每個模型完成
三項 benchmark 後傳 `keep_alive=0` 卸載，才進入下一模型。

## Benchmark、revision 與授權

| Benchmark | 正式資料 | 固定 revision | evaluator commit |
|---|---|---|---|
| OmniDocBench | HF full v1.7，1,651 頁 | `aa1ee96d...` | `193627ae...` |
| TC-STR | test，3,706 張 | upstream Git `3519b657...` | 本地 scorer |
| VisTW-MCQ | 21 subjects 全部 test | `ea498590...` | `66bf925...` |

OmniDocBench 的「資料 revision」與 GitHub「評估程式 commit」分開記錄。OmniDocBench
官方聲明資料僅供研究、不可商用；TC-STR repository 是 Apache-2.0，但下載資料仍應依
來源條款使用；VisTW dataset card／evaluation repository 若未給出足夠明確的資料授權，
使用者需在正式下載與發表前向資料提供者確認。模型授權也必須逐一確認，特別是第三方
SEA-LION GGUF 的再散布與使用條款。

TC-STR 直接送原始 crop，不做銳化、放大、灰階或裁切。這和原論文固定 resize
preprocessing 不完全相同，因此不得宣稱數字與原論文完全可比。

## 固定推論設定

所有正式請求使用 `/api/chat`、`stream=false`、`temperature=0`、`seed=42`、
`repeat_penalty=1.0`、concurrency=1；每題獨立 request，不帶 conversation history，
不主動設定 `top_p` 或 `top_k`。

| Benchmark | num_ctx | num_predict | timeout |
|---|---:|---:|---:|
| TC-STR | 4096 | 128 | 180 s |
| VisTW-MCQ | 8192 | 2048 | 600 s |
| OmniDocBench（Qwen/Gemma/SEA-LION） | 65536 | 32000 | 1800 s |
| OmniDocBench（SmolVLM2） | 8192 | 6144 | 1800 s |

SmolVLM2 的長上下文能力不和其他模型等同，報告會保留它的 effective options。每一
model × benchmark 正式開始前另跑一筆 warm-up；不納入 accuracy 或 steady-state
latency。正式樣本仍保存 Ollama `load_duration`。

設計承接 `ablation_experiment/` 的實證：chat/generate 分數相同而 chat 略快；短 OCR
prompt 曾嚴重格式失敗；`repeat_penalty=1.6` 顯著傷害 TC-STR；`num_predict=25` 曾截斷。
`repeat_penalty=1.0` 是公平、可重現的控制值，不表示已證明它是所有模型的數學最佳值。

非正式 calibration 僅可使用 train split：

```bash
PYTHONPATH=benchmark_suite/src python3 -m ocf_benchmark.cli \
  --config benchmark_suite/configs/experiment.yaml calibrate --split train
```

其設計比較 `[1.0, 1.1, 1.3]`，不屬於 15 組正式結果，不會在 test 選參數或在正式跑分
途中改參數。

## 評分

### OmniDocBench

使用官方 `end2end` evaluator，不是 crop component evaluation。每頁輸出同名 `.md`，
公式為 LaTeX、表格為 HTML，交給官方 JSON annotation matching。主要指標：

`Overall = ((1 - Text Edit Distance) × 100 + Table TEDS + Formula CDM) / 3`

另保存 Reading Order Edit Distance、官方其他欄位、頁數、failure 與 truncated。
若官方輸出不足以做 page-level paired bootstrap，CI 顯示 N/A 與原因，絕不虛構。

### TC-STR

主分數只做 Unicode NFC、CRLF normalization、首尾 whitespace 移除；不移除內部空格、
標點，不轉繁簡／大小寫、不修字、不折疊重複字元、不截短。報告 Exact Accuracy、
micro CER、ANLS(0.5)、one-way ground-truth-in-prediction containment、character
multiset F1、raw exact、格式遵循、空輸出、failure、truncated。

另外動態載入既有 `ablation_experiment/postprocess.py`，保存 raw、minimally normalized
與 legacy cleaned response 以及 postprocessor SHA256。legacy EM/CM/ANLS/F1 只供前次
實驗比較，永不取代主 Exact Accuracy。該檔的邏輯來源與版本在 manifest/record 中可追蹤。

### VisTW-MCQ

主要名稱固定為 **VisTW Macro Accuracy (deterministic parser)**，即 21 subjects accuracy
的平均；另報 micro、每 subject N/accuracy、invalid、failure、truncated。

預設 parser：

1. 取最後一個 `答案: A-D` 或 `答案：A-D`；
2. 全形 Ａ－Ｄ先轉半形；
3. 否則只接受最後非空行是單一 A-D；
4. 不從解釋段落猜答案；無法解析即 invalid 且計錯。

官方 evaluation code 的 regex fallback 會用 `gpt-4o-mini`。本專案預設完全離線，不會
自動傳送題目、圖片或輸出到外部 API。選配 official-parser adapter 預設關閉；只有使用者
明確啟用並提供 API key 時才可另產生分數，且不得與 deterministic parser 混用。

## 續跑、retry 與紀錄

resume key 包含 config hash、logical model、實際 digest、benchmark、dataset revision、
sample ID、prompt hash。每筆完成立即 append JSONL、flush、fsync。最多三次 attempt，
僅 transport、timeout、HTTP 5xx 以 2/5/15 秒退避；content error、空輸出、invalid answer
及 `done_reason=length` 不重跑。terminal failure 的 prediction 是空字串，仍留在分母。
連續 transport failure 會打開 circuit breaker，輪詢 health，不會自行 `sudo` 或重啟服務。

明確重試 terminal failures（會建立新的 append record，不是偷偷改舊列）：

```bash
./benchmark_suite/run_all.sh --resume --retry-failed
```

## 其他操作

```bash
./benchmark_suite/run_all.sh --dry-run
./benchmark_suite/run_all.sh --smoke
./benchmark_suite/run_all.sh --resume
./benchmark_suite/run_all.sh --config benchmark_suite/configs/constrained.yaml
```

dry-run 不下載大型資料／模型、不推論。smoke 每組只跑極小量樣本。正式流程一定先完成
所有資料、evaluator、五模型準備，再跑 15 組 smoke，避免跑到一半才發現第五個模型缺失。

## 輸出

每次 run 位於 `benchmark_suite/runs/<run_id>/`：

```text
manifest.json
effective_config.yaml
progress.json
logs/
predictions/
scores/
failures.csv
summary.csv
summary.json
RESULTS.md
report.html
```

`summary` 永遠剛好 15 列，即使某組 failed/blocked。報告有三張獨立排行榜，不平均三個
raw score；若做跨任務總覽，只能用 benchmark rank/mean rank 並標為 secondary。
`report.html` 是自包含檔案，可直接開啟。

確認完成：`status.sh` 應顯示 process stopped，manifest `status` 是 `completed`，
`summary.csv` 除 header 外正好 15 行。重現或引用結果時至少保存／引用
`manifest.json`：它記錄 Git/diff、主機硬體、Ollama、model digest/show、dataset/evaluator
revision、prompt hash、effective YAML、failure policy 與 lock hash。

## 常見問題

- Ollama health 失敗：確認 `ollama serve` 與 `OLLAMA_HOST`。
- Docker permission denied：若帳號剛加入 `docker` 群組，請重新登入 SSH，或先執行
  `newgrp docker` 讓目前 shell 繼承群組；正式 runner 不會自行 sudo。
- scratch 磁碟不足：先配置上述 `/srv`，或清理非本 run cache／使用 constrained
  profile；不要在 run 中途改 YAML。
- `/srv` stop 後被清空：這是預期行為；啟動服務時會重建目錄，模型與 dataset 需重抓，
  持久的 `benchmark_suite/runs/` 仍可依相同 resume identity 續跑。
- Hugging Face 429：系統會自動退避並沿用已下載 cache；登入 Hugging Face 可取得較高
  rate limit。不要刪除 `/srv/ocf-benchmark/work/cache` 後重試。
- 模型 tag 不存在：先用 `ollama pull <tag>` 驗證，或用 YAML／環境變數替換。
- Omni scoring blocked：查看 `scores/*artifacts/official_stderr.log`，prediction 不會遺失。
- OOM：正式推論不會自動降參數。停止後改用 constrained profile，形成另一個 run。
- 資料／授權有疑義：不要執行正式下載或發表，先向 upstream/model owner 確認。

實作重用聲明：Ollama request 包裝與 legacy 清理概念分別參考 repository 既有
`eval/ollama_client.py`、`eval/run_until_done.sh` 與
`ablation_experiment/postprocess.py`；新 runner 已改為規格要求的 `/api/chat`、
不遞補失敗、強 resume key 與完整 telemetry。官方 adapter 依各 upstream 實際格式實作。
