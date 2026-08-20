# 計畫全貌、未竟之事，以及如何參與

這份文件記錄 OCF 2026 AI 實習計畫**原本規劃了什麼、實際完成了什麼、什麼還沒做**，
並且盡可能把未完成的部分整理成可以直接接手的狀態。

我們的立場是：**把這次的嘗試完整交代出來，包括沒做完的部分。** 一個只展示成功的
成果頁對後來者沒有幫助——真正有用的是知道哪裡已經有路、哪裡還沒有。

---

## 一、原本的規劃：兩條研究路線

這個實習計畫原本設計了兩條並行的研究路線：

| | **路線 A：評測** | **路線 B：微調** |
|---|---|---|
| 題目 | 開源 VLM 的繁中 OCR 與文件解析評測 | Gemma 4 台灣領域微調 |
| 要回答的問題 | 開源模型在繁中 OCR 上能做到什麼程度？怎麼才算公平地測？ | 針對台灣領域微調，能否在台灣 benchmark 上量測到進步？ |
| 使用的尺 | TC-STR、OmniDocBench、VisTW-MCQ | TMMLU+、TMLU、TC-Eval |
| 技術路線 | Ollama + 自建 runner + 官方 evaluator | lm-evaluation-harness + vLLM + QLoRA／SFT |
| 硬體 | AWS `g6e.2xlarge`（L40S 48GB） | 原規劃本地 2× RTX PRO 6000 96GB（**最後未採購**） |
| **實際狀態** | ✅ **完成**，成果即本 repo | 📋 **僅完成規劃，未執行** |

兩條路線是互補的：**你得先能公平地量，才有資格說微調有沒有效。** 路線 A 的核心發現
（分數主要由評測設定決定，而非模型能力）其實正好說明了路線 B 為什麼困難——如果連
baseline 都測不準，「微調後進步了 3%」這句話就沒有意義。

---

## 二、路線 A：完成了什麼

完整成果見 [README.md](README.md)。摘要：

- 五個階段、五個目錄，從最早的評測 pipeline 到兩套可重現的評測基礎建設
- 核心發現：同一個模型與資料，Exact Match 可以在 **0% 到 64.84%** 之間移動，取決於
  後處理、prompt 與生成參數
- 18,530 次推論的單變因消融實驗，完整結果已進版控

### 路線 A 還沒做完的部分

依價值排序。**前兩項只需要跑，程式都已經寫完了。**

| # | 待辦 | 難度 | 為什麼值得做 |
|---|---|---|---|
| 1 | **跑完 [`benchmark_suite/`](benchmark_suite/) 的 15 組正式結果** | 低（純執行） | 5 模型 × 3 benchmark 的系統已完成並可執行，`runs/` 目前是空的。投入產出比最高。 |
| 2 | **跑完 [`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/)** | 低～中 | 8 模型 TC-STR、OmniDocBench 全量 1,651 頁，含 9 個模型設定。需要 GPU 時間與 Docker 環境。 |
| 3 | **`repeat_penalty` 細部 sweep** | 低 | 消融只證明「1.6 明顯有害」，尚未定位最佳值。建議測預設值、1.0、1.1、1.3、1.6。 |
| 4 | **測試變因之間的交互作用** | 中 | OFAT 無法捕捉交互作用，但 prompt × 後處理、`num_predict` × 後處理都有理由懷疑存在。 |
| 5 | **把消融結論在其他模型上驗證** | 中 | 目前結論只在 GLM-OCR Q8 上成立。Gemma 4 與 Qwen3-VL 是否有同樣的參數敏感性是開放問題。 |
| 6 | **把散落在 EC2 上的結果收回版控** | 低 | 兩套系統的正式輸出留在執行主機的 EBS 上，未隨 repo 散布。 |

---

## 三、路線 B：規劃完整，但沒有執行

這條路線最後**沒有實習生執行**。規劃文件寫得相當完整，包含 8 週路線圖、逐行可照跑的
Phase 0 runbook、以及雲端開機到關機的完整指令。與其讓它留在內部硬碟裡，我們把它整理
出來——**這條路線現在是空的，歡迎接手。**

> 以下內容整理自 2026-06 的內部規劃文件。**模型 repo 名稱、套件 API 與 benchmark task
> 名稱當時查證過，但已過了一段時間，執行前請重新核對。**

### 北極星目標

兩個月產出一個**台灣領域微調過的 Gemma 4 模型**，達成兩件事：

1. 在台灣 benchmark（TMMLU+／TMLU／TC-Eval）上，**相對 base model 有可量測的進步**；
2. 附一套**可重現、可開源**的完整管線（程式碼＋資料＋model card，以 Apache 2.0／OCF 釋出）。

**驗收只看這兩件事。** 全參數 vs QLoRA、31B Dense vs 較小尺寸、SFT vs DPO——技術路線
全部自由，只要 benchmark 有進步且可重現。

### 8 週路線圖

核心策略：**先在最小模型上把端到端流程跑通**，再把同一套管線換到目標大模型。

| Phase | 週次 | Done 的定義 | 產出 |
|---|---|---|---|
| **0** | Week 0 | 用 base Gemma 4（建議先 E4B）跑過 TMMLU+／TMLU，產出一張 baseline 分數表，並能講清楚 harness 怎麼算分 | baseline 分數表 |
| **1** ★ | Week 1–2 | 跑通 `資料 → 訓練 → 評測 → 前後對比` 整個迴圈。**品質普通沒關係**，重點是迴圈會動、有 before/after 數字 | 訓練 script、小型台灣領域資料集、前後對照 |
| **2** | Week 3–4 | 整理一份真正的台灣領域 SFT 資料集，並依硬體決定目標路線 | 有文件的資料集＋訓練設定檔 |
| **3** | Week 5–6 | 在目標模型上跑實際微調，迭代超參數，eval 較 baseline 有進步 | checkpoint、訓練 log、進步數據 |
| **4** | Week 7–8 | 跨 TMMLU+／TMLU／TC-Eval 完整評測＋ablation；注意**資料污染**、誠實回報；開源釋出 | 模型＋報告＋可重現管線 |

Phase 1 是最關鍵的一步。跑通之後，後面就只是「換更大的模型＋更好的資料」。

### 領域候選

Phase 2 挑一個或多個，標準是「資料拿得到、你有興趣、benchmark 量得出進步」：

- **法律**（台灣法規／判決／法律常識）
- **醫療**（台灣醫療衛教／用藥／健保情境）
- **一般台灣知識／文化**（在地常識、繁中通用能力）
- **台語**（台灣閩南語理解／生成評測）

### 背景速覽

- **Gemma 4**（Google，Apache 2.0）：E2B、E4B、**26B MoE**（每 token 啟用約 3.8B）、
  **31B Dense**（30.7B 全啟用）。原生多模態、大 context。
- **台灣 benchmark**：TMMLU+（66 科繁中選擇題）、TMLU（在地化深、抗資料污染）、
  TC-Eval（含閱讀理解／摘要／情感）。
- **顯存需求**：31B QLoRA 4-bit 約 22GB、LoRA 16-bit 約 65GB、**全參數約 496GB**
  （約 8 張卡，本案不走）。完整表格見 [HARDWARE.md](HARDWARE.md)。
- **硬體現況**：原本規劃的本地雙卡工作站**最後沒有採購**，OCF 改為委請廠商規劃。
  接手這條路線的人請假設**從雲端單卡開始**——Phase 0 與 Phase 1 用租的就夠，
  成本估算見 [HARDWARE.md](HARDWARE.md#二雲端先租不要先買)。

---

## 四、Phase 0 Runbook：直接照跑

這是路線 B 唯一寫到可以直接執行的部分。目標是產出 base Gemma 4 在台灣 benchmark 上的
baseline 分數表。

### 0. 取得 Gemma 4 存取權

Gemma 4 在 Hugging Face 是 **gated（需同意授權）**：先到模型頁按同意，再建一組 read
權限的 Access Token。

```bash
pip install -U "huggingface_hub[cli]"
hf auth login   # 貼上你的 token
```

### 1. 環境

```bash
conda create -n gemma4 python=3.11 -y && conda activate gemma4
# 或： python3 -m venv ~/venv/gemma4 && source ~/venv/gemma4/bin/activate

# PyTorch：到 https://pytorch.org 取對應你 CUDA 版本的指令
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -U transformers accelerate datasets sentencepiece
pip install -U bitsandbytes peft trl          # Phase 1 QLoRA 會用
pip install -U "lm-eval[vllm]"                 # 評測；vllm 可選、加速大模型

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
```

### 2. 確認 benchmark task 名稱

```bash
python -m lm_eval --tasks list 2>/dev/null | grep -i -E "tmmlu|tmlu"
```

- **TMMLU+**：task `tmmluplus`（資料集 `ikala/tmmluplus`，66 科）
- **TMLU**：`miulab/tmlu`（37 科）。若 `--tasks list` 沒看到，改裝最新 main：
  `pip install -U git+https://github.com/EleutherAI/lm-evaluation-harness`，
  或改用 [MiuLab/TMLU](https://github.com/MiuLab/TMLU) 官方 repo 評測。

### 3. 跑 baseline

```bash
mkdir -p results

python -m lm_eval --model hf \
  --model_args pretrained=google/gemma-4-E4B-it,dtype=bfloat16 \
  --tasks tmmluplus \
  --device cuda:0 --batch_size auto \
  --apply_chat_template --fewshot_as_multiturn --num_fewshot 5 \
  --output_path results/baseline_e4b_tmmluplus.json
```

> **加速版**（大模型）：`--model hf` 換成 `--model vllm`、`--model_args
> pretrained=...,dtype=bfloat16,gpu_memory_utilization=0.85`。
> **多卡**：HF 後端加 `,parallelize=True` 分片大模型；小模型用 `accelerate launch`
> 做 data parallel。

### 4. 一次跑完整個模型家族（雲端，約 6–8 小時）

在 `g6e.12xlarge`（4× L40S＝192GB）上用 vLLM 張量並行。**開 tmux**，跑很久。

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?請先 export HF_TOKEN=hf_xxx（read 權限）}"
export HF_HUB_ENABLE_HF_TRANSFER=1   # 加速大模型下載（31B ~62GB）

python3 -m venv ~/venv/gemma4
source ~/venv/gemma4/bin/activate
pip install -U pip
pip install -U "lm-eval[vllm]" vllm transformers accelerate datasets \
  sentencepiece hf_transfer "huggingface_hub[cli]"
hf auth login --token "$HF_TOKEN" 2>/dev/null || true

mkdir -p ~/results
TASKS="tmmluplus,tmlu"     # TC-Eval 留待 Phase 4
NFEWSHOT=5

# 模型 → tensor_parallel_size（4× L40S 48GB = 192GB）
#   E4B/12B 單卡即可；26B MoE 用 2 卡、31B 用 4 卡
MODELS=(
  "google/gemma-4-E4B-it:1"
  "google/gemma-4-12B-it:1"
  "google/gemma-4-26B-A4B-it:2"
  "google/gemma-4-31B-it:4"
)

# 由小到大逐模型跑，小模型先跑可以早點發現流程問題
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

**跑完務必 terminate 或 stop instance**，忘記關機是最常見的浪費。雲端機型選擇與成本
估算見 [HARDWARE.md](HARDWARE.md#二雲端先租不要先買)。

### 5. Phase 0 的交付物

| 模型 | benchmark | 總平均 | 備註（few-shot／日期） |
|---|---|---|---|
| gemma-4-E4B-it (base) | TMMLU+ | \_\_.\_% | 5-shot / 2026-\_\_-\_\_ |
| gemma-4-E4B-it (base) | TMLU | \_\_.\_% | 5-shot / 2026-\_\_-\_\_ |

**Done ＝ 這張表跑得出來、而且你能講清楚 harness 怎麼算分。**

### 常見卡點

| 症狀 | 處理 |
|---|---|
| 下載 403／gated | 回第 0 步確認已在模型頁同意授權、登入成功 |
| CUDA OOM | 降 `--batch_size`、加 `max_length=2048`、或先用更小模型確認流程 |
| chat template 警告 | `-it` 版才需 `--apply_chat_template`；base 版拿掉 |
| TMLU task 找不到 | 裝 harness 最新 main，或走 MiuLab/TMLU 官方 repo |
| `VcpuLimitExceeded` | G 系列 GPU 配額沒過，要另外申請 |
| `InsufficientInstanceCapacity` | 換 region，或改用 Spot |

---

## 五、如何參與

這個 repo 是 OCF 的公開研究成果，歡迎任何形式的接手與延伸。

### 依投入程度

| 你有 | 可以做 |
|---|---|
| **十分鐘** | 讀 [README 的核心發現](README.md#核心發現)，用瀏覽器開 [`ablation_experiment/results/ablation_report.html`](ablation_experiment/results/ablation_report.html) 看逐題證據。發現錯誤請開 issue。 |
| **一個下午 + 一張消費級顯卡** | 重現[消融實驗](ablation_experiment/)。只需要 Ollama、一個 1B 級 OCR 模型與 TC-STR 資料集。驗證我們的數字，或換一個模型看結論是否成立。 |
| **幾天 + 一張 48GB 卡** | 跑完 [`benchmark_suite/`](benchmark_suite/) 的 15 組正式結果，或 [`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/) 的 8 模型評測。**程式都寫完了，只差執行。** |
| **兩個月** | 接手[路線 B](#三路線-b規劃完整但沒有執行)。從 Phase 0 開始，路線圖與 runbook 都在上面。 |
| **不寫程式** | 幫忙檢視方法論、指出評測設計的盲點、提供台灣領域的資料集線索。路線 B 的 Phase 2 需要領域知識更甚於工程能力。 |

### 幾個原則

如果你要延伸這份研究，我們建議沿用以下作法——它們是這次實習得到的教訓：

1. **引用數字時連同執行條件一起引用。** 這是本研究最主要的結論。至少要有 exact model
   tag、digest、資料版本、prompt、生成參數與後處理版本。
2. **不要用自製近似分數頂替官方 evaluator。** 失敗就標成失敗。
3. **把模型造成的失敗與工程造成的失敗分開。** 前者照樣計分，後者要擋下實驗。
4. **例外要留紀錄。** 誰在什麼時候基於什麼理由核准了什麼豁免，以及哪些欄位因此不可
   等量比較。
5. **保留 raw response。** 事後所有的爭議都要靠它來釐清。

### 聯絡

透過 [OCF](https://ocf.tw/) 或直接在本 repo 開 issue／PR。

---

## 相關文件

- [README.md](README.md) — 研究成果與核心發現
- [CONTRIBUTORS.md](CONTRIBUTORS.md) — 研究團隊
- [INFRASTRUCTURE.md](INFRASTRUCTURE.md) — 執行環境與主機配置
- [HARDWARE.md](HARDWARE.md) — 硬體規劃與成本參考
