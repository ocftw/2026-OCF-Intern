# OCF 2026 AI 暑期實習研究成果
## 開源視覺語言模型（VLM）的繁體中文 OCR 與文件解析評測

本 repo 是[開放文化基金會（OCF）](https://ocf.tw/)2026 年 AI 暑期實習計畫的完整研究
成果。兩位實習生用**本機 Ollama** 執行開源 VLM，評測它們在繁體中文場景文字辨識與整頁
文件解析上的表現，並在過程中建立了一套可重現、可稽核的評測基礎建設。

實習期間：2026-07-15 ～ 2026-08-14。所有工作已合併至 `main`。

---

## TL;DR — 如果你只有三分鐘

這次實習真正的成果不是「哪個模型比較強」的排行榜，而是一個更根本的發現：

> **開源 VLM 的 OCR 評測分數，主要由評測設定決定，而不是模型能力。**
> 同一個模型、同一份 3,706 筆資料，光是調整後處理與生成參數，Exact Match 可以在
> **0% 到 64.84%** 之間移動。

這個結論是用一組嚴謹的單變因消融實驗（OFAT）證實的，證據在
[`ablation_experiment/`](ablation_experiment/)。它直接改變了後半段實習的方向：與其
繼續累積不可比的分數，不如先把「什麼設定被固定了」變成評測系統的一等公民。後半段的
[`benchmark_suite/`](benchmark_suite/) 與 [`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/)
就是這個想法的實作。

---

## 核心發現

以下全部來自同一個模型（GLM-OCR Q8_0 GGUF）、同一份資料（TC-STR test split，
3,706 筆）、同一台機器。每組實驗**只改一項設定**，其餘維持 baseline 不變。
所有需要推論的 variant 都完整跑完 3,706 筆，API error 數為 0。

| 實驗（相對 baseline 的唯一改動） | EM | CM | ANLS | F1 | 平均延遲 |
|---|---:|---:|---:|---:|---:|
| Aligned baseline | 47.06% | 76.52% | 54.54% | 59.17% | 0.602 s |
| 改用 `/api/chat` | 47.06% | 76.52% | 54.54% | 59.17% | 0.582 s |
| 改用短 prompt | **0.00%** | 79.06% | **0.00%** | 13.56% | 0.594 s |
| 不指定 `repeat_penalty=1.6` | **64.84%** | **82.46%** | **72.08%** | **76.62%** | 0.583 s |
| `num_predict` 80 → 25 | 46.03% | 76.34% | 53.29% | 65.07% | **0.293 s** |
| 改用最小後處理 | **0.00%** | 76.79% | **0.00%** | 6.88% | 離線重算 |
| 改用另一套 scorer | 47.87% | 77.63% | 55.02% | 59.96% | 離線重算 |

### 1. `repeat_penalty=1.6` 是最大的單一傷害來源

只是不顯式傳入這個參數、回到 Ollama／模型預設值，**EM 就從 47.06% 提升到 64.84%
（+17.78 個百分點）**，ANLS +17.53 pp。逐題比對是 **779 題由錯轉對、120 題由對轉錯，
淨增加 659 題完全正確**。這是本次測到最有效的單一改動，而且它完全免費——不換模型、
不加資料、不增加延遲。

### 2. EM 掉到 0% 不代表模型認不出字

短 prompt 那組 EM 與 ANLS 都是 0.00%，但 **CM（是否包含正確答案）反而是全場最高的
79.06%**。這個組合說明模型其實把字認出來了，只是在答案後面附加了解釋、Markdown、
英文分析與 prompt 複誦，導致嚴格比對全滅。

這也解釋了為什麼**看指標的「形狀」比看單一數字重要**：如果兩份報告呈現「EM/ANLS
差很多、CM/F1 差不多」，幾乎可以斷定差異來自後處理或輸出格式，而不是辨識能力。

### 3. 後處理不是細節，是評分能否成立的前提

aligned 後處理修改了 baseline **3,704 / 3,706 筆** raw response。換成最小後處理後，
同一批 raw response 的 EM 直接歸零。也就是說，現行流程能拿到合理分數，有很大一部分
要歸功於後處理規則，而不是模型輸出本身就乾淨。

### 4. `/api/generate` 與 `/api/chat` 完全等價

3,706 / 3,706 筆 processed prediction **逐字完全相同**，四個指標一模一樣。0.020 秒的
延遲差距不足以證明何者較快。這條路徑可以從「分數差異的可能原因」清單裡劃掉。

### 5. 評分器本身也會製造差異

同一批 prediction（**沒有任何一筆改變**），只換 scorer，就多判定 30 題 EM 正確
（+0.81 pp）。原因是兩套 scorer 對空白／標點的正規化與 CM 的單向／雙向定義不同。
這是報告差異，不是模型能力提升——**跨團隊比較分數前必須先對齊 scorer**。

### 6. 速度與準確度的取捨可以量化

`num_predict` 從 80 降到 25，平均延遲縮短約 **51%**（0.602 s → 0.293 s），代價是
EM -1.03 pp、ANLS -1.25 pp。但要注意：被截斷的 prompt 複誦可能反而躲過後處理規則，
所以這組數字不能單純解讀成「便宜的加速」。

> ⚠️ **解讀限制**：這是 OFAT 單變因主效應分析，各列 delta **不能直接相加**。數字只
> 代表 GLM-OCR Q8 + TC-STR + 該 Ollama 環境，不保證泛化到其他模型。CM 只檢查是否
> 包含 ground truth，答案藏在大量雜訊中仍會得分，不應單獨用來判斷 OCR 品質。

---

## 成果地圖

五個目錄對應實習的五個階段，**依時間順序閱讀最容易理解脈絡**：

```
2026-07-15  eval/                 提出問題：為什麼大家分數差那麼多？
2026-07-16  Sixhuang/             獨立第二套實作，成為對照組
2026-07-21  ablation_experiment/  ★ 回答問題：3,706 筆 × 7 組單變因實驗
2026-07-28  benchmark_suite/      把結論制度化：5 模型 × 3 benchmark 無人值守系統
2026-07-30  TC-STR_and_OminDoc/   規模化：8 模型 OCR + 官方 evaluator 文件解析
   ～08-14
```

| 目錄 | 作者 | 這是什麼 | 結果數據 |
|---|---|---|---|
| [`eval/`](eval/) | [@trickster-2005](https://github.com/trickster-2005) | TC-STR 三模型評測 pipeline，含後處理、四指標與互動式 HTML 報告。**提出了本次實習的核心問題** | ✅ `eval/results/report.html`（18.7 MB） |
| [`Sixhuang/`](Sixhuang/) | [@hyslchs](https://github.com/hyslchs) | 獨立實作的第二套跑分程式，另外納入 HF 合成資料集。它與 `eval/` 的差異（短 prompt、最小後處理、不同 scorer）後來成為消融實驗的三個變因 | ❌ 未進版控，需重跑 |
| [`ablation_experiment/`](ablation_experiment/) | [@hyslchs](https://github.com/hyslchs) | **★ 本 repo 最重要的成果。** 對齊兩套流程後，做 7 組 OFAT 單變因消融，共 18,530 次推論 | ✅ 完整（見下方說明） |
| [`benchmark_suite/`](benchmark_suite/) | [@hyslchs](https://github.com/hyslchs) | 可長時間無人值守、append-only 斷點續跑的評測系統。5 模型 × 3 benchmark（OmniDocBench／TC-STR／VisTW-MCQ）= 15 組正式結果 | ❌ 系統完成，正式結果未產出 |
| [`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/) | [@hyslchs](https://github.com/hyslchs) | 規模最大的一套。TC-STR 8 個模型；OmniDocBench v1.6 全量 1,651 頁，用**釘死版本的官方 Docker evaluator** 計分，另有 9 個模型設定與多模型比較報告工具 | ❌ 結果保存在 EC2 的 EBS 上，未進版控 |

### 關於「結果數據」欄

這點對後續接手的人很重要：**大部分實驗的原始結果不在這個 repo 裡。**

repo 內實際保存的結果只有兩份：

- `eval/results/report.html` — trickster 的三模型評測報告，自包含 HTML，直接開啟即可。
- `ablation_experiment/results/` — 消融實驗的完整成果，包含
  [`summary.json`](ablation_experiment/results/summary.json)（機器可讀的七組彙總）、
  [`run_manifest.json`](ablation_experiment/results/run_manifest.json)（模型、prompt、
  options 與資料 fingerprint）、
  [`dataset_manifest.json`](ablation_experiment/results/dataset_manifest.json)（3,706 張
  圖片的順序、ground truth 與 SHA-256）以及
  [`ablation_report.html`](ablation_experiment/results/ablation_report.html)（完整互動
  報告與逐題差異案例）。

`benchmark_suite/` 與 `TC-STR_and_OminDoc/` 交付的是**執行系統本身**，不是結果。它們
的正式輸出寫在執行主機（AWS EC2）的持久磁碟上，未隨 repo 散布。這是刻意的設計：兩者
都要求正式結果必須連同 model digest、dataset fingerprint、evaluator commit 與
effective config 一起保存，否則不具可信度。逐筆 CSV 與 raw response 體積也很大
（消融實驗光是逐筆 CSV 就約 17 MB）。

如果你要接手，**這些系統都可以重跑並產生等價輸出**——這正是它們把所有設定釘死的目的。

---

## 這套評測基礎建設解決了什麼

後半段的兩套系統是消融實驗結論的直接產物。既然分數如此容易被設定影響，評測系統就
必須讓「設定」無法被偷偷改動。它們共同的設計原則：

- **設定進入身分識別。** resume key／run signature 包含 config hash、實際 model
  digest、dataset revision、prompt hash 與 dependency lock SHA256。任何一項不同，
  就是另一次實驗，不會混入舊結果。
- **失敗要看得見，不要被補起來。** 模型自己造成的空輸出、拒答、重複、格式錯誤與
  `done_reason=length` 全部保留並照樣計分，只標成 anomaly。只有非模型因素的工程錯誤
  （資料 signature 不符、模型載不起來、非 100% GPU、evaluator 失敗）才會擋下評測。
- **不用自製分數頂替官方分數。** OmniDocBench 若官方 evaluator 失敗，prediction 保留、
  scoring 明確標成 `blocked`，絕不改用近似值。
- **官方指標與診斷指標分區。** EM／CM／ANLS／F1 在 OmniDocBench 報告中明確標為
  「非官方 leaderboard 指標」，不與官方 Edit Distance／TEDS／CDM 混用。
- **raw response 永久保存**，主評分不使用依模型客製的清理規則。
- **例外要留紀錄。** 例如 GLM OCR BF16 在 Ollama 會回傳有效文字但 `done=false` 且缺
  token metadata，這個豁免被明文記錄在
  [`PROTOCOL_EXCEPTIONS.md`](TC-STR_and_OminDoc/TC_STR/PROTOCOL_EXCEPTIONS.md)，並註明
  其 token 效率不可與其他模型等量比較。

`benchmark_suite/` 的固定推論設定（`temperature=0`、`seed=42`、**`repeat_penalty=1.0`**、
concurrency=1）正是承接消融實驗的實證而來——但文件也誠實註明：`repeat_penalty=1.0`
是為了公平與可重現而選的控制值，**不代表已證明它是所有模型的最佳值**。

---

## 評測對象

### 資料集

| 資料集 | 內容 | 規模 | 用於 |
|---|---|---|---|
| [TC-STR 7k-word](https://github.com/esun-ai/traditional-chinese-text-recogn-dataset) | 繁體中文場景文字辨識（招牌、看板等 crop） | test split 3,706 張 | 全部四個 OCR 實驗 |
| [OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench) | 整頁文件解析（文字／表格／公式／閱讀順序） | full set 1,651 頁 | `TC-STR_and_OminDoc/`（v1.6，revision `d386947f`）、`benchmark_suite/`（v1.7，revision `aa1ee96d`） |
| [VisTW-MCQ](https://huggingface.co/datasets/miulab/vistw-mcq) | 繁體中文視覺選擇題 | 21 subjects 全部 test | `benchmark_suite/` |
| [ZihCiLin/traditional-chinese-ocr-synthetic](https://huggingface.co/datasets/ZihCiLin/traditional-chinese-ocr-synthetic) | 繁中 OCR 合成資料 | `test_random`、`test_semantic` | `Sixhuang/` |

### 模型

橫跨實習的五個階段，累計評測或設定過的開源 VLM：

- **Gemma 4 系列**（QAT Q4_0）：E2B、E4B、12B、26B-A4B、31B
- **Qwen 系列**：Qwen2.5-VL 3B、Qwen3-VL 4B／32B
- **GLM 系列**：GLM-OCR Q8_0 GGUF、GLM-OCR BF16、GLM-4.6V-Flash 9B
- **InternVL3.5**：4B、38B（Q4_K_M）
- **其他**：Kimi-VL-A3B-Instruct、SmolVLM2 2.2B、Qwen-SEA-LION v4 4B-VL、chandra-ocr-2

第三方 GGUF 量化版本（非模型原廠發布）的來源、風險與核准紀錄，逐一記錄在
[`MODEL_CANDIDATES.md`](TC-STR_and_OminDoc/TC_STR/MODEL_CANDIDATES.md)。這點在評測
報告中經常被忽略，但它直接影響結果是否可歸因到原始模型。

### 執行環境

後期實驗的主機（`TC-STR_and_OminDoc/` 建立與驗證期間，2026-07-29 ～ 07-30）：

| 項目 | 規格 |
|---|---|
| Host | AWS EC2 / Ubuntu 26.04 LTS |
| CPU / RAM | AMD EPYC 7R13（8 vCPU）／61 GiB |
| GPU | NVIDIA L40S，46 GB VRAM（driver 610.43.02） |
| Ollama | 0.31.1 |
| Docker | 29.6.2（OmniDocBench 官方 evaluator 用） |

所有推論 batch size 固定為 1、concurrency 固定為 1、模型依序載入並在階段結束後卸載，
確保各模型不會互相爭用 VRAM。

---

## 從哪裡開始讀

- **只想知道結論** → 本頁〈核心發現〉，接著看
  [`ablation_experiment/RESULTS.md`](ablation_experiment/RESULTS.md)。
- **想看逐題證據** → 用瀏覽器開
  [`ablation_experiment/results/ablation_report.html`](ablation_experiment/results/ablation_report.html)
  與 [`eval/results/report.html`](eval/results/report.html)（都是自包含檔案，不需
  啟動 server）。
- **想理解方法論怎麼演進** → 依 `eval/` → `Sixhuang/` → `ablation_experiment/` →
  `benchmark_suite/` → `TC-STR_and_OminDoc/` 的順序讀各目錄 README。
- **想自己重跑** → 每個目錄的 README 都有完整的環境需求與指令。最容易重現的是
  [`ablation_experiment/`](ablation_experiment/)（只需要 Ollama + 一個模型 + TC-STR）。
- **想接續研究** → 見下方〈後續方向〉。

---

## 後續方向

實習結束時留下的明確待辦，依價值排序：

1. **`repeat_penalty` 細部 sweep。** 消融實驗只證明「1.6 明顯有害」，尚未定位最佳值。
   建議對預設值、1.0、1.1、1.3、1.6 做完整 sweep。
2. **測試變因之間的交互作用。** OFAT 無法捕捉交互作用，而 prompt × 後處理、
   `num_predict` × 後處理、prompt × repeat penalty 都有理由懷疑存在交互作用。
3. **跑完既有的評測系統。** `benchmark_suite/` 的 15 組正式結果與
   `TC-STR_and_OminDoc/` 的 8 模型 TC-STR、OmniDocBench 全量評測，系統都已就緒但
   正式數據尚未產出。這是投入產出比最高的一項——程式已經寫完了。
4. **把消融結論在其他模型上驗證。** 目前結論只在 GLM-OCR Q8 上成立，Gemma 4 與
   Qwen3-VL 是否有同樣的參數敏感性仍是開放問題。

---

## 貢獻者

| | 負責範圍 |
|---|---|
| [@trickster-2005](https://github.com/trickster-2005) | [`eval/`](eval/) — TC-STR 評測 pipeline、後處理設計、四指標實作、HTML 報告，並提出「跨團隊分數不可比」的問題 |
| [@hyslchs](https://github.com/hyslchs) | [`Sixhuang/`](Sixhuang/)、[`ablation_experiment/`](ablation_experiment/)、[`benchmark_suite/`](benchmark_suite/)、[`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/) — 對照實作、消融實驗設計與執行、兩套可重現評測基礎建設 |

兩人的工作原本分別在 `main` 與 `Sixhuang` 分支上進行，已於 2026-08-19 透過
[PR #1](https://github.com/ocftw/2026-OCF-Intern/pull/1) 合併。**`main` 目前包含全部
研究成果**；`Sixhuang` 分支僅作為歷史紀錄保留，內容與 `main` 完全一致。

---

## 授權與使用限制

本 repo 的**程式碼**由 OCF 2026 AI 暑期實習計畫產出。**資料集與模型不隨 repo 散布**，
使用者必須自行向上游取得並遵守各自的授權條款。特別注意：

- **OmniDocBench** 官方聲明資料僅供研究使用、不可商用。
- **TC-STR** 的 repository 是 Apache-2.0，但下載的資料仍應依來源條款使用。
- **VisTW** 的 dataset card 與 evaluation repository 未給出足夠明確的資料授權，正式
  下載與發表前應向資料提供者確認。
- **第三方 GGUF 量化版本**（`mradermacher`、`haervwe` 等社群發布）的再散布與使用條款
  必須逐一確認，不能假設等同於原始模型的授權。

評測數字的引用請務必連同執行條件一起引用——這正是本研究最主要的結論。
