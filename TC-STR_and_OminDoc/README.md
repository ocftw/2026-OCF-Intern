> **成果定位**｜第 5 階段・規模最大的一套｜作者 [@hyslchs](https://github.com/hyslchs)｜2026-07-30 ～ 08-14
>
> ⚠️ **結果數據不在這個 repo 裡。** 正式輸出（checkpoint、raw response、prediction、
> metadata、評分與報告）保存在執行主機 AWS EC2 的 EBS 持久磁碟上，未隨 repo 散布。
> 本目錄交付的是完整的**可重現執行工具**——所有版本、digest 與設定都已釘死，重跑
> 可得到等價輸出。
>
> **兩套評測**：
> - [`TC_STR/`](./TC_STR/)：繁中短文字辨識，TC-STR test split 3,706 張，**8 個模型**
>   （Gemma 4 五種尺寸、GLM-OCR BF16、GLM-4.6V-Flash 9B、Kimi-VL-A3B）
> - [`OminDocBench/`](./OminDocBench/)：整頁文件解析，OmniDocBench v1.6 **全量 1,651
>   頁**，用釘死版本的**官方 Docker evaluator** 計分（不自製近似分數）。主設定 2 個
>   模型，`config_variants/` 另有 7 個（Gemma 4 E4B/12B/31B、Qwen3-VL 4B/32B、
>   InternVL3.5 4B/38B）
>
> **值得注意的方法論細節**：官方指標與自製診斷指標嚴格分區（EM/CM/ANLS/F1 明確標為
> 「非 OmniDocBench 官方 leaderboard 指標」）；模型自己造成的失敗照樣計分、只標
> anomaly；protocol 例外（如 GLM OCR BF16 的 `done=false` 問題）明文記錄在
> [`TC_STR/PROTOCOL_EXCEPTIONS.md`](./TC_STR/PROTOCOL_EXCEPTIONS.md)；第三方 GGUF
> 量化的來源與風險記錄在
> [`TC_STR/MODEL_CANDIDATES.md`](./TC_STR/MODEL_CANDIDATES.md)。
>
> 完整脈絡見 [repo 根目錄 README](../README.md)。

---

# TC-STR 與 OmniDocBench 文件辨識評測

本目錄包含兩套以本機 Ollama 執行視覺語言模型（VLM）的可重現評測工具：

- [`TC_STR/`](./TC_STR/)：繁體中文短文字辨識，使用 TC-STR test split。
- [`OminDocBench/`](./OminDocBench/)：整頁文件解析，固定使用 OmniDocBench v1.6
  full set。（目錄名稱 `OminDocBench` 保留專案既有拼字。）

本文記錄足以撰寫研究方法與重現實驗的設定，但**不包含任何尚在執行中的測試結果**。
實際執行時的 model digest、資料 fingerprint、時間、GPU 狀態、逐題輸出與分數，另存於
每個 run 的 durable output directory；應以該 run 的 `metadata.json`、manifest 與報告為準。

## 實驗設計與可比性

兩套評測都遵守以下原則：

1. 同一 Benchmark 內，各模型取得完全相同的圖片、prompt、generation options 與評分器。
2. 每次 request 只輸入一張圖片（batch size 1），透過 Ollama
   `http://127.0.0.1:11434/api/generate`、`stream=false` 執行。
3. 模型依序載入；切換模型時卸載前一模型，避免多模型同時占用 VRAM。
4. warm-up 不納入分數與 latency；正式 latency 只在相同輸入、參數及
   `ollama ps` 顯示 `100% GPU` 時比較。
5. 模型本身造成的空輸出、拒答、重複、格式錯誤、無關內容或長度上限等現象會保留並計分，
   不會事後針對個別模型調整 prompt 或參數。資料／signature 不符、模型無法載入、非
   `100% GPU`、API 完全無有效回應等非模型工程錯誤才會阻擋評測。
6. raw response 永久保存；主評分不使用依模型客製的文字清理。
7. SQLite checkpoint、prediction、raw response、metadata 與報告直接寫入 EBS。
   `/mnt/nvme` 僅存放可重新下載或建立的資料。

## 系統環境

以下是建立及驗證本套工具時的主機環境（2026-07-29 至 2026-07-30）。每個 run 仍會重新
記錄當下環境，避免軟體更新後把不同條件的結果混在一起。

| 項目 | 版本／規格 |
|---|---|
| Cloud host | AWS EC2 |
| OS | Ubuntu 26.04 LTS（Resolute Raccoon） |
| Kernel | Linux `7.0.0-1009-aws` |
| CPU | AMD EPYC 7R13；8 logical CPUs、4 cores、2 threads/core |
| RAM | 61 GiB；無 swap |
| GPU | NVIDIA L40S；46,068 MiB VRAM（約 45 GiB） |
| NVIDIA driver | 610.43.02 |
| Host Python | 3.14.4 |
| Ollama | 0.31.1 |
| Docker Engine | 29.6.2（build `dfc4efb`） |
| Ollama KV cache | fp16 baseline；runner 只記錄，不修改服務設定 |
| Time records | ISO 8601 UTC；HTML 另顯示 Asia/Taipei |

OmniDocBench 的官方評分在獨立 Docker image 中執行；已驗證的 runtime 為 Python
3.10.16、TeX Live 2025（pdfTeX 3.141592653-2.6-1.40.28）、ImageMagick 7.1.1-47
及 Ghostscript 9.55.0。

儲存空間的角色如下：

| 路徑 | 用途 | EC2 stop/start 後 |
|---|---|---|
| `/mnt/nvme/datasets/` | dataset | 會消失，可重抓 |
| `/mnt/nvme/scratch/`、`/mnt/nvme/outputs/` | 可重建的暫存／cache | 會消失 |
| `/opt/ocf-ai/outputs/` | checkpoint、raw、prediction、metadata、評分與報告 | EBS 持久保存 |

## TC-STR：繁體中文短文字辨識

### Dataset 與評測範圍

- 來源：[`esun-ai/traditional-chinese-text-recogn-dataset`](https://github.com/esun-ai/traditional-chinese-text-recogn-dataset)
- Dataset 名稱：TC-STR 7k-word
- Split：官方 test split
- 樣本數：3,706
- Label file：`test_labels.txt`
- 本次固定 label SHA-256：
  `398c04e674ec9ab5409fc228d12bd3b92e97029203a463f832831a6de1e1e9dc`
- 包含所有圖片內容與標註的 dataset fingerprint：
  `3043510ba884a50615be3dc2770a8e03557426c1793af899c846983ede91664c`
- 輸入方式：直接傳送原始圖片 bytes；不做 resize、OCR crop、旋轉、銳化、超解析或其他增強。
- Smoke：固定分層選取 20 題；8 個模型使用相同題目，共 160 個輸出。

### 模型

下表是本機 preflight 實際確認的模型映射。正式 run 會再次比對 exact tag、digest、vision
capability、量化及 `100% GPU`，不會自動下載或以其他模型替代。

| 邏輯模型 | Exact Ollama tag | 參數量 | 量化 | Digest（SHA-256） |
|---|---|---:|---|---|
| Gemma 4 E2B | `gemma4:e2b-it-qat` | 4.6B | Q4_0 | `07ea59a474013479c8b6b802bef095c40e964a1d776ba02f264c0e30e1aede0c` |
| Gemma 4 E4B | `gemma4:e4b-it-qat` | 7.5B | Q4_0 | `ee665637121887cf3befff38abbb1be4ee117c7db867d97a67e29049ecd7e15f` |
| Gemma 4 12B | `gemma4:12b-it-qat` | 11.9B | Q4_0 | `38044be4f923e5a55264ed7df4eaac2676651a905f735197c504045140c02bd3` |
| Gemma 4 26B A4B | `gemma4:26b-a4b-it-qat` | 25.2B total / 約 3.8B active | Q4_0 | `2dd70431afed94dd3688d790443768c1487ed086b57147ff083851116ae4c4e4` |
| Gemma 4 31B | `gemma4:31b-it-qat` | 30.7B | Q4_0 | `e0812a55773bfeac846b2d605b4d93638b8dfa7119d9587f3d91475afc78185e` |
| GLM OCR BF16 | `glm-ocr:bf16` | 1.1B | F16（模型 tensor 為 BF16） | `6effedd0dc8aec66ad86c94b8099395376e212f22c87a9d7147c422f7e447be4` |
| GLM-4.6V-Flash 9B | `haervwe/GLM-4.6V-Flash-9B:latest` | 9.4B | Q4_K_M | `ad2e2e374c6bcda3739e6563a5d0edf231d3deb0e5beeae27e89d6e5c16f8715` |
| Kimi-VL-A3B-Instruct | `hf.co/mradermacher/Kimi-VL-A3B-Instruct-GGUF:Q4_K_M` | 16B total / A3B active | Q4_K_M | `252db10777f74ff9a13f5fe24efdea51aedc546eebac8ea14becec99c9b70698` |

第三方 Ollama／GGUF build 的來源與核准狀態記錄在
[`TC_STR/models.yaml`](./TC_STR/models.yaml) 及
[`TC_STR/MODEL_CANDIDATES.md`](./TC_STR/MODEL_CANDIDATES.md)。

### Prompt 與生成參數

所有模型逐字使用以下 prompt：

> 你是一個專業的OCR文字辨識引擎。請仔細觀察圖片，只輸出圖片中看到的文字內容（繁體中文為主，若同時有英文或數字也一併輸出），只回傳純文字本身，不要輸出HTML標籤、Markdown或JSON等任何格式化標記，不要加任何解釋、前綴或後綴、標點符號以外的內容，不要翻譯，不要加引號。如果部分文字模糊、傾斜或被遮擋，也要盡量根據字形推斷最可能的字。

| 參數 | 設定 |
|---|---:|
| `num_ctx` | 65,536 |
| `num_predict` | 80 |
| `temperature` | 0 |
| `repeat_penalty` | 1.0 |
| `top_k` | 64 |
| `top_p` | 0.95 |
| `think` | `false` |
| Timeout | 180 秒 |
| Max attempts | 3 |
| Retry backoff | 2、5 秒 |
| `keep_alive` | 0（每個模型階段後卸載） |

完整唯一來源為 [`TC_STR/config.yaml`](./TC_STR/config.yaml)。

### Normalization 與評分

主評分使用 `nfc_outer_trim_v1`：將 null 視為空字串、統一換行、套用 Unicode NFC，
並只移除最外層 whitespace。標點、內部空白、Markdown 字元、重複字元均保留。

這四項是本研究 runner 的共同 OCR 診斷指標，不應誤稱為 TC-STR 官方 leaderboard 指標：

| 指標 | 定義 | 方向 |
|---|---|---:|
| Exact Match（EM） | normalization 後 prediction 與 GT 完全相同為 1，否則 0 | ↑ |
| Containment Match（CM） | 單向判定 `GT in prediction` | ↑ |
| ANLS | `1 - LevenshteinDistance / max(len(GT), len(pred))`；小於 0.5 計 0 | ↑ |
| Character F1 | 以 Unicode 字元 multiset overlap 計 precision、recall 與 F1 | ↑ |

另保留 `reference_aligned_v1` 作為與舊實驗對照的 supplementary score；它可能移除 wrapper
或格式、壓縮過度重複字元並截至 200 字，因此不作為主結果，也不得與上述 raw-preserving
分數混用。完整定義見 [`TC_STR/METRICS.md`](./TC_STR/METRICS.md)。

GLM OCR BF16 在 Ollama 可能以 HTTP 200 回傳有效 OCR 文字，但同時為 `done=false` 且缺少
token metadata。經人工核准後，只有此 exact model 接受「HTTP 200 且 response 非空」為
可評分結果；缺少欄位保持 null，標為 `truncation_unknown`，不可用於等量比較 token
efficiency。詳見
[`TC_STR/PROTOCOL_EXCEPTIONS.md`](./TC_STR/PROTOCOL_EXCEPTIONS.md)。

## OmniDocBench v1.6：整頁文件解析

### Dataset 與版本固定

- Dataset：Hugging Face `opendatalab/OmniDocBench`
- Revision：`d386947f7fc3bafdcd756c8485845a2f43a19875`（commit title：`add v1.6`）
- 範圍：v1.6 full set，共 1,651 pages
- `OmniDocBench.json` SHA-256：
  `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`
- 輸入方式：直接使用與 GT 一一對應的官方 page image，保存原始 dimensions 與 image hash；
  不重新 rasterize，也不做 OCR crop、旋轉、增強或模型專屬 resize。
- Smoke：固定選取相同 20 pages，涵蓋語言、欄數、表格、公式、閱讀順序、模糊掃描與複雜
  版面；兩模型共 40 個輸出。

Docker image 內建的 1,355-page GT 不是本研究的 v1.6 full set，runner 會拒絕混用。
版本判定與差異詳見
[`OminDocBench/VERSION_PINNING.md`](./OminDocBench/VERSION_PINNING.md)。

### 模型

| 邏輯模型 | Exact Ollama tag | 參數量 | 量化 | Digest（SHA-256） |
|---|---|---:|---|---|
| Gemma 4 E2B | `gemma4:e2b-it-qat` | 4.6B | Q4_0 | `07ea59a474013479c8b6b802bef095c40e964a1d776ba02f264c0e30e1aede0c` |
| Gemma 4 26B A4B | `gemma4:26b-a4b-it-qat` | 25.2B total / 約 3.8B active | Q4_0 | `2dd70431afed94dd3688d790443768c1487ed086b57147ff083851116ae4c4e4` |

兩模型皆先以正式 65K context warm-up，且 `ollama ps` 必須精確顯示 `100% GPU`。

### Prompt 與生成參數

OmniDocBench v1.6 repository 沒有 Gemma 4 專用或通用 end-to-end VLM prompt，因此兩模型
逐字使用下列共同 prompt：

> 請將這一頁文件完整解析為 Markdown。依照原始閱讀順序保留所有可見文字；標題、段落、清單與程式碼請使用適當的 Markdown；表格請以能保留列、欄與合併儲存格結構的 HTML table 輸出；行內公式與獨立公式請使用 LaTeX；不要描述圖片內容，不要加入解釋、前言、結語或 code fence，只輸出解析後的文件內容。

| 參數 | 設定 |
|---|---:|
| `num_ctx` | 65,536 |
| `num_predict` | 16,384 |
| `temperature` | 0 |
| `repeat_penalty` | 1.0 |
| `top_k` | 64 |
| `top_p` | 0.95 |
| `think` | `false` |
| Timeout | 1,800 秒 |
| Max attempts | 3 |
| Retry backoff | 5、20 秒 |
| `keep_alive` | 10 分鐘 |

完整唯一來源為
[`OminDocBench/config/benchmark.json`](./OminDocBench/config/benchmark.json)。
主 prediction 只移除最外層 whitespace，以及兩模型共用、精確匹配的空 thought wrapper；
不刪除 HTML table、Markdown、公式或 JSON-looking 文件內容，不壓縮重複字元，也不截斷。

### 官方 evaluator 與主指標

正式主結果使用 OmniDocBench v1.6 官方 `end2end` evaluator：

| 項目 | Pin／設定 |
|---|---|
| Official code commit | `147cd5ac9472002f5751221d390bf00abdbc0d2f` |
| Docker image | `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204` |
| Docker digest | `sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617` |
| Supervisor config | `configs/reproduce_chutao_v16_repro_conda_20260408.yaml` |
| Config SHA-256 | `e15568889c314beca1fe511103dcdf6612af61958ccb13a9e2ae922bd7c31880` |
| Matching | `quick_match` |
| Workers | matching 4、CDM 3、TEDS 3 |
| Timeouts | quick-match truncated 300 秒、match 420 秒 |

GT 與 predictions 以 read-only mount 傳入 container；每個模型使用獨立可寫 result mount。
官方主指標及方向如下：

| 官方指標 | 內容 | 方向 |
|---|---|---:|
| Text normalized Edit Distance | 文字正規化編輯距離 | ↓ |
| Formula CDM | 公式結構／內容匹配 | ↑ |
| Table TEDS | 表格樹結構相似度 | ↑ |
| Table TEDS-S | v1.6 config 有輸出時保留 | ↑ |
| Reading Order Edit Distance | 閱讀順序編輯距離 | ↓ |
| Overall | 完全沿用 v1.6 evaluator 定義，不自行重算 | 依官方定義 |

page-level、category-level、language、layout 與 attribute 細分只要 evaluator 有輸出皆原樣
保存。官方設定、matching／ignore／attribute 邏輯不以自製整頁字串比對取代。

### 補充 OCR 診斷指標

EM、單向 CM、ANLS 與 Character F1 使用與 TC-STR 相同定義，但在報告中獨立標為
「非 OmniDocBench 官方 leaderboard 指標」。優先使用官方 matcher 產生的 text-block
pairs；若 evaluator 沒有足以可靠重建 pairs 的 artifact，才計算明確標示為 diagnostic
的 page-level flattened visible-text metrics，不自行發明 bbox 或段落 matching。

Normalization 為 Unicode NFC、統一換行及只移除最外層 whitespace；不忽略標點、內部
空白、Markdown、表格或公式。詳細定義與 aggregate denominator 見
[`OminDocBench/METRICS.md`](./OminDocBench/METRICS.md)。

## Checkpoint、續跑與可追溯性

兩套 runner 都使用 EBS 上的 SQLite transaction checkpoint。每題／每頁完成後，以 atomic
write 保存 raw response、正式 prediction、API metadata、attempt history、latency、token
counts 與 hashes，再 commit checkpoint。

Resume 只會跳過 signature 相同、terminal status 符合政策、檔案存在且 hash 相符的項目。
資料版本／manifest、model tag／digest／量化、prompt、options、image hash、Ollama
version、evaluator pin、官方 config、normalization／scorer 或 runner source fingerprint
任一不同，都不得混入舊結果。SIGINT／SIGTERM 會更新狀態，之後以同一 `./run_full.sh`
安全接續。

## 操作指令

TC-STR：

```bash
cd /home/hyslchs/20260729_TCSTR_and_omindoc/2026-OCF-Intern/TC-STR_and_OminDoc/TC_STR
./run_smoke.sh
./run_full.sh
./status.sh
./build_report.sh --run-id <run_id>
```

OmniDocBench v1.6：

```bash
cd /home/hyslchs/20260729_TCSTR_and_omindoc/2026-OCF-Intern/TC-STR_and_OminDoc/OminDocBench
./run_smoke.sh
./run_full.sh
./status.sh
./build_report.sh
./evaluate_only.sh
```

`run_smoke.sh` 不會自動進入 full run；通過 preflight、smoke 及人工檢視後，才由操作者在
tmux 中親自執行 `./run_full.sh`。重跑相同指令會從 durable checkpoint 接續，不需要指定
題號。腳本不會建立 nohup、背景排程或自動核准檔。

## 結果報告原則

本 README 刻意不寫入執行中的結果。完成後應從各 run directory 的 immutable artifacts
引用數值，並至少同時報告：

- exact tag、digest、量化與實際 `100% GPU`；
- 樣本數、成功／API failure／retry／空輸出／截斷或其他 anomaly 數；
- latency 的 mean、median、p95 與（metadata 可得時）tokens/second；
- 各 aggregate 的 denominator、unmatched count 與 empty count；
- OmniDocBench 官方主指標和非官方 supplementary metrics 的清楚分區；
- 任何 protocol exception 或無法等量比較的欄位。

如此可避免將模型能力造成的低分誤判為 pipeline error，也避免把工程失敗或不同評分規則
包裝成模型間的有效差異。
