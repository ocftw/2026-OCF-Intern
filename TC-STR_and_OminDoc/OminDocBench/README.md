# 生成多模型比較報告

任何有這台機器帳號的人都可以直接執行，**不需要這個 repo 擁有者的登入權限**——資料與程式
碼都已部署在共用路徑，只有讀取權限，不會動到任何正在進行中的評測：

```bash
cd /opt/ocf-ai/outputs/omnidocbench_reports/code

# 先看看目前有哪些模型已經有完整的官方評測結果可以選
python3 tools/build_comparison_report.py --list-models

# 以 gemma4_e2b 的 text_block edit distance 為基準，各抽 5 筆最好/最差/隨機，
# 目前所有已跑完的模型全部並列比較
python3 tools/build_comparison_report.py \
  --baseline gemma4_e2b --best 5 --worst 5 --random 5 \
  --out /opt/ocf-ai/outputs/omnidocbench_reports/generated/<你的名字>_report.html
```

- `generated/` 底下大家都能互相讀寫，產生的報告留在那裡讓其他人也能打開看。
- 打開產生的 HTML 後，畫面最上面可以直接**點欄位標題排序**整體分數表（預設 Overall 由高到
  低），也有一個工具列可以即時**切換排序依據的模型、方向、篩選最佳/最差/隨機、依檔名搜
  尋**——這些都是純前端互動，不需要重新執行指令。往下展開「📖 指標說明」摺疊區塊可以看
  六個子指標(`text_block edit`/`table edit`/`table TEDS`/`formula edit`/`formula CDM`/
  `reading_order edit`)的意義、演算法跟方向。
- 完整參數說明：`python3 tools/build_comparison_report.py --help`。
- 想要舊版「兩個模型並排、依單一模型排序、產生時就固定順序」的簡化版報告，可以用
  `tools/build_visual_report.py`（用法見檔案內 docstring）。

## 共用資料維護

上面那條路徑(`/opt/ocf-ai/outputs/omnidocbench_reports/`)是這個目錄的**唯讀鏡像**，不含任
何一頁圖片或預測資料本身——它讀的其實是 `/mnt/nvme/datasets/OmniDocBench_v1.6/`(GT、圖片)
跟 `/opt/ocf-ai/outputs/omnidocbench_v1_6/`(各模型的預測與官方評測結果)，這兩處資料夾原本
的檔案權限只有擁有者能讀，已經用 `tools/fix_shared_report_permissions.py` 開了 other-read。

**新模型跑完評測後**，要讓其他人也看得到，依序做兩件事：

```bash
# 1. 幫新模型的預測檔案(以及萬一有新圖片)開 other-read 權限(原地生效，不搬移/複製資料，
#    可重複執行，已經開過的檔案會被跳過)
python3 tools/fix_shared_report_permissions.py

# 2. 把程式碼變更同步到共用鏡像(只鏡像程式碼/設定，不含 *.html、__pycache__)
rsync -a --delete \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.html' --exclude='.git/' \
  ./ /opt/ocf-ai/outputs/omnidocbench_reports/code/
chmod -R o+rX /opt/ocf-ai/outputs/omnidocbench_reports/code
chgrp -R ocfai /opt/ocf-ai/outputs/omnidocbench_reports/code
```

（這台機器系統碟空間吃緊，資料本身刻意不複製一份到 `/opt/ocf-ai/outputs`，只鏡像程式碼——
程式碼加設定總共幾百 KB，資料在原地用權限開放讀取即可。）

---

# OmniDocBench v1.6 可重現執行工具

這個目錄提供一套端對端、可斷點續傳的執行腳本，用來跑完整的 **1,651 頁 OmniDocBench v1.6
全量測試集**。用主機上的 Ollama 做 batch size 為 1 的 VLM 推論，搭配釘死版本的官方 Docker
評測器做 `end2end` 計分。

目前的流程有一道人工確認的硬性關卡：

1. `./run_smoke.sh` 會做 preflight、每個模型的暖機/GPU 檢查、對每個設定好的模型做同樣固定
   的 20 頁推論、官方 smoke 評測，以及 smoke 報告。
2. 它不會呼叫全量執行腳本。一次有效的 smoke 執行會停留在 `SMOKE_REQUIRES_AI_REVIEW` 狀態，
   直到所有輸出都被檢視過，並且產生了內容簽章的 `ai_review.json` 以及 `READY_FOR_FULL_RUN`
   manifest 為止。
3. 只有在檢視過那份報告之後，操作者才應該手動在 tmux 裡執行 `./run_full.sh`。重新執行同一
   條指令會從 EBS 上的 checkpoint 接續執行。

沒有任何腳本會自動建立 tmux session、背景工作、排程，或審核通過的檔案。

**只針對 `config_variants/*.yaml` 的執行方式**（絕不適用於主要的 `config/models.yaml` 路
徑），`tools/run_model_variant.py full-inference --skip-review` 是一個明確的、操作者自己
選擇跳過這道關卡的選項：不加這個參數時行為完全不變，而且這個參數整個實作都放在
`tools/run_model_variant.py` 裡，不在 `code_hash` 涵蓋的範圍內——用了它不會改變任何一次執
行的 `run_signature`/`run_id`，Gemma 那組也不例外。用這種方式跳過 smoke，也代表
`run_official_evaluator()`（原本會在 smoke 評測時順便寫出每個模型的
`immutable_config/{model_id}_official.yaml`）不會被呼叫到。`full-inference` 每次都會呼叫
`ensure_official_configs()`——如果 smoke 已經產生過那個檔案就什麼都不做；但如果沒有（也就
是用了 `--skip-review`），它會自己把同樣內容、雜湊值也一致的檔案針對每個模型補上，這樣
`tools/evaluate_model.py` 之後就不會因為找不到「immutable official config」而失敗。

在「固定設定比較」的政策下，模型自己造成的結果——像是複讀、空白內容、拒答、格式錯誤的
Markdown，或是 `done_reason=length`——都會被保留下來並且照樣計分。它們會以 anomaly flag 的
形式顯示出來，但本身不會擋下這次實驗。API/執行期失敗、輸入或 signature 錯誤、GPU
fallback、predictions 缺失，以及 evaluator 失敗，這些仍然會擋下實驗。因為某一頁模型輸出退
化而導致官方 evaluator 卡死的情況，會用明確、可稽核的方式另外處理——請見下方的
`tools/mark_eval_timeout_page.py`。

## 模型

好幾組各自獨立的模型集合，每一組都有自己的 logical-to-local 對應檔案，這樣新增一組不會動
到另一組的 `run_id`（詳見 `tools/run_model_variant.py`）：

| 模型 id | 顯示名稱 | 對應檔案 |
|---|---|---|
| `gemma4_e2b` | Gemma 4 E2B | `config/models.yaml` |
| `gemma4_26b_a4b` | Gemma 4 26B A4B | `config/models.yaml` |
| `gemma4_e4b` | Gemma 4 E4B | `config_variants/models_e4b_12b_31b.yaml` |
| `gemma4_12b` | Gemma 4 12B | `config_variants/models_e4b_12b_31b.yaml` |
| `gemma4_31b` | Gemma 4 31B | `config_variants/models_e4b_12b_31b.yaml` |
| `qwen3_vl_4b` | Qwen3-VL 4B | `config_variants/models_qwen_internvl.yaml` |
| `qwen3_vl_32b` | Qwen3-VL 32B | `config_variants/models_qwen_internvl.yaml` |
| `internvl3_5_4b` | InternVL3.5 4B | `config_variants/models_qwen_internvl.yaml` |
| `internvl3_5_38b` | InternVL3.5 38B | `config_variants/models_qwen_internvl.yaml` |

九個模型目前都已經有完整的 `official_evaluation_full_1651` 結果；
`python3 tools/build_comparison_report.py --list-models` 隨時反映即時狀態（如果有模型還沒
有結果，也會說明是還在跑還是真的失敗了）。比較報告工具本身完全不用改，就能看到新的模型組
合——它會自動掃描所有 run 目錄底下、任何已經有完整結果的模型，Gemma 跟非 Gemma 都一樣；確
切指令請見本檔案最上方的「生成多模型比較報告」。

那四份 `config_variants/model_<id>.yaml` 單一模型的設定檔（各自有自己的 `run_id`），是在合
併執行之前，用來分別對每個模型做 smoke 測試用的；`config_variants/models_qwen_internvl.yaml`
才是實際用來完成上面那次全量執行的設定檔，之後要重跑或擴充都應該用這一份。

## 釘死的輸入版本

- 資料集：`opendatalab/OmniDocBench` 版本 `d386947f7fc3bafdcd756c8485845a2f43a19875`
  （`add v1.6`）
- 完整 GT 的 SHA-256：
  `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`
- 預期頁數/圖片數：1,651
- 官方評測程式碼：v1.6 main 分支最後一次 commit
  `147cd5ac9472002f5751221d390bf00abdbc0d2f`
- 執行用的映像檔：
  `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204@sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617`

1,355 頁版本的內嵌 GT 有一個重要的差異，請見 `VERSION_PINNING.md`。

## 路徑

- 程式碼/設定：這個 repo 目錄本身
- 資料集：`/mnt/nvme/datasets/OmniDocBench_v1.6/`
- 可重建的暫存空間：`/mnt/nvme/scratch/omnidocbench_v1_6/`
- 可重建的推論快取：`/mnt/nvme/outputs/omnidocbench_v1_6/`
- Checkpoint、原始回應、predictions、evaluator 結果、報告：
  `/opt/ocf-ai/outputs/omnidocbench_v1_6/<run_id>/`
- 給其他使用者也能讀取、專門用來產生報告的共用鏡像（見上方說明）：
  `/opt/ocf-ai/outputs/omnidocbench_reports/`

這個執行腳本會先把成功的原始回應跟 predictions 直接寫進 EBS，再去 commit SQLite
transaction。`/mnt/nvme` 永遠不會是某次推論結果唯一的一份拷貝。

## 資料集準備

下載腳本刻意跟 preflight 分開。它只會下載釘死的那個版本，檢查 GT 的雜湊值跟每一個 Hugging
Face LFS object 的雜湊值，並且會從已經驗證過的檔案繼續下載：

```bash
python3 scripts/fetch_dataset.py
```

想在不下載圖片的情況下檢視遠端的 immutable image manifest：

```bash
python3 scripts/fetch_dataset.py --manifest-only
```

## 模型核准

`config/models.yaml`（以及任何透過 `tools/run_model_variant.py` 使用的
`config_variants/*.yaml`）是唯一有效的 logical-to-local 對應檔案。一筆對應必須包含一個確
切的本機 Ollama tag，而且 `supervisor_approved` 要是 `true`。Preflight 絕對不會自己去 pull
模型，也絕對不會偷偷換成別的尺寸、別的量化版本、第三方轉的版本，或是雲端 API。模型的
digest、architecture、quantization、capabilities 跟 context 相關的中繼資料，都是直接從本
機的 Ollama API 取得的。

不是每個模型都有官方的 Ollama library 條目。`internvl3_5_4b` 和 `internvl3_5_38b` 是從
OpenGVLab 的 InternVL3.5 的社群 GGUF 量化版本匯入的（官方 Ollama library 裡沒有任何
InternVL 的視覺模型）；`tools/fetch_vlm_models.py` 對兩個 Qwen3-VL 的官方 tag 做
`ollama pull`，對兩個社群 GGUF 的 InternVL3.5 則是下載、驗證 sha256、用雙 `FROM` 的
Modelfile 去 `ollama create`。一筆對應也可以額外宣告 `expected_min_context`，用來覆寫預設
的 65536 這個門檻——給那些原生 context 真的比較低的模型用（InternVL3.5 的原生 context 是
40960）。這是一個明確記錄下來、承認這個限制存在的決定，不是在掩蓋一個 bug；完整的來龍去脈
（社群 GGUF 的來源、匯入時碰到的硬碟空間問題、context 的但書）都寫在這兩筆對應各自的
`mapping_evidence` 欄位裡。

## 指令

```bash
./run_tests.sh
./run_smoke.sh
./run_smoke.sh --preflight-only
./status.sh
./build_report.sh --smoke
./evaluate_only.sh
```

smoke 明確回報成 `READY_FOR_FULL_RUN` 之後，唯一的全量執行指令是：

```bash
./run_full.sh
```

`evaluate_only.sh` 完全不會呼叫 Ollama。`build_report.sh` 只讀 EBS 上的 checkpoint 跟
evaluator 產出的檔案，不會做任何推論。

## `tools/` 參考

這裡的每一個腳本，對任何一次執行的官方評測輸出來說都是唯讀的，除非它自己的 docstring 另外
說明。

| 腳本 | 作用 |
|---|---|
| `evaluate_model.py` | 對一個已經存在的全量推論結果，執行釘死版本的官方 evaluator；支援可斷點續傳的批次處理（`--batch-size`），每個批次都 `done` 之後會自動合併成整個語料庫的結果，`--eval-timeout-as-empty` 可以把已知會卡死的那一頁換成空白 prediction（見 `mark_eval_timeout_page.py`）。 |
| `auto_resume_evaluation.py` | 在已確認的「輸出退化導致複讀」的 evaluator 卡死問題上，負責監督 `evaluate_model.py`：重試卡住的批次，只有在卡死的訊號完全符合已知的特徵時，才會自動透過 `mark_eval_timeout_page.py` 登記——否則會停下來讓人工檢視。 |
| `mark_eval_timeout_page.py` | 把某一頁 ground truth 記錄成（或移除）已知的官方 evaluator 卡死頁面，寫進 `evaluation_timeout_pages.json`，一定要附上理由跟證據。只有在某次執行帶了 `--eval-timeout-as-empty` 時才會真的被套用；真正的 prediction 內容永遠不會被修改。 |
| `fix_shared_report_permissions.py` | 每次有新模型時要跑一次的權限修正（見上方「共用資料維護」）：對 GT JSON、資料集圖片，以及每個模型的 `predictions/` 檔案開 other-read 權限，讓非擁有者的使用者也能產生報告。可重複執行，不會動到其他任何檔案。 |
| `run_model_variant.py` | 對一組定義在 `config_variants/*.yaml`（放在 `config/` 之外）裡的替代模型組合，執行 preflight/smoke/inference-only，讓它擁有自己的 `run_id`，不會干擾到其他任何一次執行的 signature。`full-inference` 支援 `--skip-review`（見上方硬性審核關卡那段說明）。 |
| `fetch_vlm_models.py` | 抓取/匯入 `config_variants/models_qwen_internvl.yaml` 跟四份單一模型 `model_<id>.yaml` 背後的模型：對兩個 Qwen3-VL 的 tag 做官方的 `ollama pull`，對兩個社群 GGUF 的 InternVL3.5 匯入則是下載+驗證 sha256+雙 `FROM` Modelfile 的 `ollama create`。`--only <id>` 可以只抓其中幾個；`--smoke-test` 會在抓完之後真的送一張圖片去問每個模型，並印出回應內容。 |
| `full_inference_only.py` | 執行完整的 1,651 頁推論，但不含最後內建呼叫 evaluator 那一步（評測是之後另外用 `evaluate_model.py`/`auto_resume_evaluation.py` 做，這兩個工具有內建呼叫沒有的卡死保護）。 |
| `evaluation_status.py` / `inference_status.py` | 針對正在進行中的官方評測/推論執行，提供唯讀的進度快照（單次或 `--watch`）。 |
| `stop_evaluation.py` | 停掉某次執行/某個模型還留著在跑的 evaluator container。 |
| `build_comparison_report.py` | 自成一體、可互動排序的 HTML 報告：把每個有完整結果的模型並排顯示，包含頁面圖片、ground truth，以及依某個基準模型分數抽出的最佳/最差/隨機頁面子集，瀏覽器裡可以依任何一個模型或任何一項總表指標重新排序。這就是本檔案最上方說明的那個工具。 |
| `build_visual_report.py` | 比較舊版、一次只看兩個模型並排的視覺化報告，產生當下就固定好排序方式。 |
| `_container_ops.py`、`_merge_driver.py` | 給上面那些腳本用的內部輔助工具（docker container 查詢；在釘死版本的 evaluator image 裡執行的批次合併邏輯），不是設計來直接被呼叫的。 |

## 斷點續傳與中斷

`results.sqlite` 用 `(run_signature, model_id, page_id)` 當 key。一頁只有在狀態是成功、圖
片的雜湊值一致、raw 跟 Markdown 檔案都存在、而且兩邊的雜湊值都還對得上的情況下，才會被視為
可以重複使用。SIGINT/SIGTERM 會先設一個「正在中斷」的狀態，等目前這一頁完整處理完才真正停
下來。任何 config、模型對應、prompt/options、資料集版本、evaluator 版本，或計分器版本的變
動，都會產生不同的 signature，原本的 checkpoint 會拒絕沿用。

Predictions 目錄底下只放官方要的 `.md` 檔案。Raw API 回傳的物件跟每一頁的中繼資料，放在旁
邊的兄弟目錄裡。

## 共通的推論規格

每個模型都用一模一樣的 prompt 文字，以及來自 `config/benchmark.json` 的選項：
`num_ctx=65536`、`num_predict=16384`、`temperature=1.0`、`repeat_penalty=1.0`、
`top_k=64`、`top_p=0.95`、`seed=20260731`、最上層的 `think=false`、`stream=false`、batch
size 為 1，以及 fp16 的 KV cache。官方的頁面圖片是直接傳進去的，不做任何 OCR 裁切、影像增
強，或重新 rasterize。

某次回應如果碰到共同設定的 16,384 token 上限，會被保留下來，並且回報成「在共同設定下，模
型本身產生的一個可計分結果」。這個上限不會為了單一模型而調高。
