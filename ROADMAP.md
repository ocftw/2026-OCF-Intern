# 計畫全貌、未竟事項與參與方式

本文記錄 OCF 2026 AI 暑期實習計畫原先的規劃、實際完成的部分，以及尚未執行的部分，並就
未完成者盡可能整理至可直接接手的狀態。

本計畫的立場在於將整次嘗試完整交代，包含未完成的部分。僅展示成功項目的成果頁對後來者
助益有限；真正有用的資訊，在於得知何處已有既成路徑，何處尚無。

---

## 一、原先的規劃：兩條研究路線

本實習計畫原先設計兩條並行的研究路線：

| | 路線 A：評測 | 路線 B：微調 |
|---|---|---|
| 題目 | 開源 VLM 的繁中 OCR 與文件解析評測 | Gemma 4 台灣領域微調 |
| 待答問題 | 開源模型於繁中 OCR 上的表現程度為何？如何方屬公平的量測？ | 針對台灣領域微調，能否於台灣 benchmark 上量測到進步？ |
| 採用的基準 | TC-STR、OmniDocBench、VisTW-MCQ | TMMLU+、TMLU、TC-Eval |
| 技術路線 | Ollama、自建 runner 與官方 evaluator | lm-evaluation-harness、vLLM、QLoRA 與 SFT |
| 硬體 | AWS `g6e.2xlarge`（L40S 48GB） | 原規劃本地 2× RTX PRO 6000 96GB，**最終未採購** |
| 實際狀態 | **已完成**，成果即本專案 | **僅完成規劃，未執行** |

兩條路線互為前提：唯有先能公平地量測，才有資格判斷微調是否有效。路線 A 的核心發現
（分數主要取決於評測設定而非模型能力）恰可說明路線 B 的困難所在：baseline 若無法測準，
「微調後進步 3%」一語便失去意義。

---

## 二、路線 A：已完成的部分

完整成果見 [README.md](README.md)，摘要如下：

- 五個階段、五個目錄，自最早的評測流程至兩套可重現的評測基礎建設
- 核心發現為同一模型與資料之下，Exact Match 可在 0% 至 64.84% 之間變動，取決於後處理、
  prompt 與生成參數
- 以 18,530 次推論完成的單變因消融實驗，完整結果已進版控

### 路線 A 尚未完成的部分

依價值排序如下，前兩項僅需執行，程式均已完成。

| 序 | 待辦事項 | 難度 | 價值 |
|---|---|---|---|
| 1 | **執行 [`benchmark_suite/`](benchmark_suite/) 的 15 組正式結果** | 低（純執行） | 5 模型 × 3 benchmark 的系統已完成並可執行，`runs/` 目前為空，投入產出比最高 |
| 2 | **執行 [`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/)** | 低至中 | 涵蓋 8 模型 TC-STR 與 OmniDocBench 全量 1,651 頁，另備 9 個模型設定，需要 GPU 時間與 Docker 環境 |
| 3 | **`repeat_penalty` 的細部 sweep** | 低 | 消融實驗僅證明 1.6 明顯有害，尚未定位最佳值，建議測試預設值、1.0、1.1、1.3 與 1.6 |
| 4 | **檢驗變因之間的交互作用** | 中 | OFAT 無法捕捉交互作用，惟 prompt 與後處理、`num_predict` 與後處理之間均有理由懷疑存在 |
| 5 | **於其他模型上驗證消融結論** | 中 | 目前結論僅在 GLM-OCR Q8 上成立，Gemma 4 與 Qwen3-VL 是否具同樣的參數敏感性仍屬開放問題 |
| 6 | **將散存於 EC2 的結果收回版控** | 低 | 兩套系統的正式輸出留在執行主機的 EBS 上，未隨專案散布 |

---

## 三、路線 B：規劃完整，未曾執行

本路線最終並無實習生執行。規劃文件相當完整，包含 8 週路線圖、可逐行照跑的 Phase 0
runbook，以及雲端自開機至關機的完整指令。與其留存於內部硬碟，不如一併整理公開。該路線
目前仍為空白，歡迎接手。

> 以下內容整理自 2026 年 6 月的內部規劃文件。模型 repo 名稱、套件 API 與 benchmark task
> 名稱當時均經查證，惟已歷時一段時間，執行之前請重新核對。

### 目標

以兩個月產出一個經台灣領域微調的 Gemma 4 模型，並達成兩項要求：

1. 於台灣 benchmark（TMMLU+、TMLU、TC-Eval）上相對 base model 有可量測的進步
2. 附一套可重現、可開源的完整管線，包含程式碼、資料與 model card，以 Apache 2.0 或 OCF
   名義釋出

驗收僅檢視上述兩項。全參數微調或 QLoRA、31B Dense 或較小尺寸、SFT 或 DPO，技術路線完全
自由，惟須 benchmark 有所進步且可重現。

### 8 週路線圖

核心策略為先在最小模型上建立端到端流程，再將同一套管線移至目標大模型。

| Phase | 週次 | 完成的定義 | 產出 |
|---|---|---|---|
| **0** | Week 0 | 以 base Gemma 4（建議先採 E4B）執行 TMMLU+ 與 TMLU，產出一張 baseline 分數表，並能說明 harness 的計分方式 | baseline 分數表 |
| **1** | Week 1–2 | 建立資料、訓練、評測至前後對比的完整迴圈；品質普通亦可，重點在於迴圈能夠運作並產出 before/after 數字 | 訓練 script、小型台灣領域資料集、前後對照 |
| **2** | Week 3–4 | 整理一份實質的台灣領域 SFT 資料集，並依硬體決定目標路線 | 附文件的資料集與訓練設定檔 |
| **3** | Week 5–6 | 於目標模型執行實際微調，迭代超參數，使 eval 相較 baseline 有所進步 | checkpoint、訓練 log、進步數據 |
| **4** | Week 7–8 | 跨 TMMLU+、TMLU 與 TC-Eval 完成完整評測與 ablation，留意資料污染並誠實回報，完成開源釋出 | 模型、報告與可重現管線 |

Phase 1 為最關鍵的階段。迴圈建立完成之後，後續工作即為更換更大的模型並改善資料。

### 領域候選

Phase 2 擇一或數項進行，選擇標準為資料可取得、執行者具備興趣，且 benchmark 量得出進步：

- **法律**：台灣法規、判決與法律常識
- **醫療**：台灣醫療衛教、用藥與健保情境
- **一般台灣知識與文化**：在地常識與繁中通用能力
- **台語**：台灣閩南語的理解與生成評測

### 背景資訊

- **Gemma 4**（Google，Apache 2.0）包含 E2B、E4B、26B MoE（每 token 啟用約 3.8B）與
  31B Dense（30.7B 全啟用），原生多模態並支援大 context。
- **台灣 benchmark** 方面，TMMLU+ 為 66 科繁中選擇題，TMLU 在地化程度深且抗資料污染，
  TC-Eval 涵蓋閱讀理解、摘要與情感。
- **顯存需求**方面，31B 的 QLoRA 4-bit 約 22GB，LoRA 16-bit 約 65GB，全參數微調約
  496GB（約 8 張卡，本案不採），完整表格見 [HARDWARE.md](HARDWARE.md)。
- **硬體現況**方面，原先規劃的本地雙卡工作站最終未採購，OCF 改為委請廠商規劃。接手者
  請以自雲端單卡起步為前提，Phase 0 與 Phase 1 租用即可，成本估算見
  [HARDWARE.md](HARDWARE.md#二雲端資源租用先於採購)。

---

## 四、Phase 0 Runbook

以下為路線 B 唯一已寫至可直接執行的部分，目標為產出 base Gemma 4 於台灣 benchmark 上的
baseline 分數表。

### 0. 取得 Gemma 4 存取權

Gemma 4 於 Hugging Face 屬 gated 模型，須先同意授權。請先至模型頁面按下同意，再建立一組
具 read 權限的 Access Token。

```bash
pip install -U "huggingface_hub[cli]"
hf auth login   # 輸入 token
```

### 1. 環境

```bash
conda create -n gemma4 python=3.11 -y && conda activate gemma4
# 或： python3 -m venv ~/venv/gemma4 && source ~/venv/gemma4/bin/activate

# PyTorch：請至 https://pytorch.org 取得對應 CUDA 版本的指令
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -U transformers accelerate datasets sentencepiece
pip install -U bitsandbytes peft trl          # Phase 1 的 QLoRA 將會使用
pip install -U "lm-eval[vllm]"                 # 評測；vllm 為選用，可加速大模型

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
```

### 2. 確認 benchmark task 名稱

```bash
python -m lm_eval --tasks list 2>/dev/null | grep -i -E "tmmlu|tmlu"
```

- **TMMLU+** 的 task 為 `tmmluplus`，資料集為 `ikala/tmmluplus`，共 66 科。
- **TMLU** 為 `miulab/tmlu`，共 37 科。若 `--tasks list` 未列出，請改裝最新 main
  （`pip install -U git+https://github.com/EleutherAI/lm-evaluation-harness`），或改用
  [MiuLab/TMLU](https://github.com/MiuLab/TMLU) 官方 repo 評測。

### 3. 執行 baseline

```bash
mkdir -p results

python -m lm_eval --model hf \
  --model_args pretrained=google/gemma-4-E4B-it,dtype=bfloat16 \
  --tasks tmmluplus \
  --device cuda:0 --batch_size auto \
  --apply_chat_template --fewshot_as_multiturn --num_fewshot 5 \
  --output_path results/baseline_e4b_tmmluplus.json
```

大模型可改用 vLLM 後端加速，將 `--model hf` 改為 `--model vllm`，並使用
`--model_args pretrained=...,dtype=bfloat16,gpu_memory_utilization=0.85`。多卡環境下，
HF 後端可加上 `,parallelize=True` 以分片大模型，小模型則可用 `accelerate launch` 進行
data parallel。

### 4. 一次執行完整模型家族

以下於 `g6e.12xlarge`（4× L40S，合計 192GB）上透過 vLLM 張量並行執行，約需 6 至 8 小時，
建議開啟 tmux。

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?請先 export HF_TOKEN=hf_xxx（read 權限）}"
export HF_HUB_ENABLE_HF_TRANSFER=1   # 加速大模型下載（31B 約 62GB）

python3 -m venv ~/venv/gemma4
source ~/venv/gemma4/bin/activate
pip install -U pip
pip install -U "lm-eval[vllm]" vllm transformers accelerate datasets \
  sentencepiece hf_transfer "huggingface_hub[cli]"
hf auth login --token "$HF_TOKEN" 2>/dev/null || true

mkdir -p ~/results
TASKS="tmmluplus,tmlu"     # TC-Eval 留待 Phase 4
NFEWSHOT=5

# 模型與 tensor_parallel_size 的對應（4× L40S 48GB = 192GB）
#   E4B 與 12B 單卡即可；26B MoE 使用 2 卡，31B 使用 4 卡
MODELS=(
  "google/gemma-4-E4B-it:1"
  "google/gemma-4-12B-it:1"
  "google/gemma-4-26B-A4B-it:2"
  "google/gemma-4-31B-it:4"
)

# 由小至大依序執行，先跑小模型可及早發現流程問題
for entry in "${MODELS[@]}"; do
  MODEL="${entry%%:*}"; TP="${entry##*:}"
  SAFE="$(echo "$MODEL" | tr '/:' '__')"
  echo "==================== $MODEL  (TP=$TP) ===================="
  lm_eval --model vllm \
    --model_args "pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=${TP},gpu_memory_utilization=0.9,max_model_len=4096" \
    --tasks "$TASKS" \
    --batch_size auto \
    --apply_chat_template --fewshot_as_multiturn --num_fewshot "$NFEWSHOT" \
    --output_path ~/results/"${SAFE}".json
done
```

執行完畢務必 terminate 或 stop instance，忘記關機為最常見的浪費。雲端機型選擇與成本
估算見 [HARDWARE.md](HARDWARE.md#二雲端資源租用先於採購)。

### 5. Phase 0 的交付物

| 模型 | benchmark | 總平均 | 備註（few-shot 與日期） |
|---|---|---|---|
| gemma-4-E4B-it (base) | TMMLU+ | \_\_.\_% | 5-shot / 2026-\_\_-\_\_ |
| gemma-4-E4B-it (base) | TMLU | \_\_.\_% | 5-shot / 2026-\_\_-\_\_ |

完成的定義為分數表確實產出，且執行者能說明 harness 的計分方式。

### 常見問題

| 症狀 | 處理方式 |
|---|---|
| 下載出現 403 或 gated | 回到第 0 步，確認已於模型頁同意授權且登入成功 |
| CUDA OOM | 調降 `--batch_size`、加上 `max_length=2048`，或先以更小的模型確認流程 |
| chat template 警告 | `-it` 版本方需 `--apply_chat_template`，base 版本應移除 |
| 找不到 TMLU task | 安裝 harness 最新 main，或改用 MiuLab/TMLU 官方 repo |
| `VcpuLimitExceeded` | G 系列 GPU 配額尚未核准，須另行申請 |
| `InsufficientInstanceCapacity` | 更換 region，或改用 Spot |

---

## 五、如何參與

本專案為 OCF 的公開研究成果，歡迎各種形式的接手與延伸。

### 依投入程度區分

| 可投入的資源 | 建議的參與方式 |
|---|---|
| **十分鐘** | 閱讀 [README 的核心發現](README.md#核心發現)，並以瀏覽器開啟 [`ablation_report.html`](ablation_experiment/results/ablation_report.html) 檢視逐題證據。如發現錯誤，請開立 issue。 |
| **一個下午與一張消費級顯示卡** | 重現[消融實驗](ablation_experiment/)，僅需 Ollama、單一 1B 級 OCR 模型與 TC-STR 資料集。可用以驗證本研究的數字，或更換模型檢驗結論是否成立。 |
| **數日與一張 48GB 卡** | 執行 [`benchmark_suite/`](benchmark_suite/) 的 15 組正式結果，或 [`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/) 的 8 模型評測。程式均已完成，僅待執行。 |
| **兩個月** | 接手[路線 B](#三路線-b規劃完整未曾執行)，自 Phase 0 起步，路線圖與 runbook 均如前述。 |
| **不撰寫程式** | 協助檢視方法論、指出評測設計的盲點、提供台灣領域的資料集線索。路線 B 的 Phase 2 所需的領域知識更甚於工程能力。 |

### 建議遵循的原則

延伸本研究時，建議沿用以下作法，均為本次實習所得的教訓：

1. **引用數字時一併引用執行條件**，至少包含 exact model tag、digest、資料版本、prompt、
   生成參數與後處理版本。此為本研究最主要的結論。
2. **不以自製近似分數頂替官方 evaluator**，失敗即標示為失敗。
3. **區分模型造成的失敗與工程造成的失敗**，前者照常計分，後者應擋下實驗。
4. **例外須留下紀錄**，載明由何人於何時基於何種理由核准何項豁免，以及哪些欄位因而不可
   等量比較。
5. **保留 raw response**，事後一切爭議均須據以釐清。

### 聯絡方式

請透過 [OCF](https://ocf.tw/) 聯繫，或直接於本專案開立 issue 與 PR。

---

## 相關文件

- [README.md](README.md)：研究成果與核心發現
- [CONTRIBUTORS.md](CONTRIBUTORS.md)：研究團隊
- [INFRASTRUCTURE.md](INFRASTRUCTURE.md)：執行環境與主機配置
- [HARDWARE.md](HARDWARE.md)：硬體規劃與成本參考
