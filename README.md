# OCF 2026 AI 研究實習成果
## 開源視覺語言模型於繁體中文 OCR 與文件解析之評測

本專案收錄[開放文化基金會（OCF）](https://ocf.tw/)2026 年 AI 研究實習計畫的研究成果。
研究以本機 Ollama 執行開源視覺語言模型（Vision Language Model, VLM），評測其在繁體中文
場景文字辨識與整頁文件解析上的表現，並於過程中建立一套可重現、可稽核的評測基礎建設。

實習期間為 2026 年 7 月 1 日至 8 月 30 日，計約 9 週；版控紀錄涵蓋 7 月 15 日至 8 月
19 日。全部工作均已合併至 `main` 分支。

---

## 計畫背景

### 計畫定位

開放文化基金會長期關注開源軟體、開放資料與數位人權。2026 年的 AI 研究實習計畫
（職缺公告見 <https://ocf.tw/hiring.html>）將主題設定為**臺灣在地 AI 評測、開放資料、
模型微調與算力基礎建設**，目的在於檢驗大型語言模型於繁體中文、臺灣在地知識，以及醫療、
法律等專業領域的能力與限制，並將成果以開源方式公開。

計畫的重心明確落於評測、資料與公開三者，而非訓練大型模型。招募條件所列的必要項目為對
開源、開放資料與公共利益的關注，閱讀英文技術文件的能力，以及整理資料、撰寫文件與記錄
研究的能力；模型訓練經驗僅列為加分。所預期的工作內容包含建立 benchmark 題庫與標準答案、
執行不同模型並分析比較結果、將評測資料與成果開源公開，以及建置評測所需的環境。

上述定位相當程度決定了本專案最終的樣貌：交付物之中份量最重者為評測方法與可重現的執行
系統，而非模型權重或排行榜名次。

### 待答的問題

計畫初期的工作紀錄留下一個具體的技術疑問：語言模型的評測方法，在輸入端加入視覺之後應
如何調整。

純文字模型的評測已有相對成熟的慣例，如 lm-evaluation-harness 與各語言的選擇題基準。
一旦模型改為讀取圖片並輸出文字，評分對象即自「選項是否正確」轉為「一段自由文字與
ground truth 之間的距離」，而後者的定義並無公認標準：編輯距離如何正規化、標點與空白
是否計入、模型附加的解釋文字應否清除，各項選擇均會顯著改變分數。

繁體中文另有兩重困難。其一為分詞邊界不明確，導致詞級指標難以直接套用，須改採字元級
計算。其二為可用的繁中評測資料集數量有限，且多數缺乏公認的評分實作，使得不同團隊的
數字往往不可比較。

### 為何評測繁體中文 OCR

文件辨識為開放資料的前置作業。政府公報、判決書、歷史報刊等大量資料以掃描影像的形式
存在，若缺乏可靠的辨識工具，即無法進入可檢索、可分析的狀態。商用 OCR API 則涉及使用
成本與資料外流的疑慮，對經費有限的組織以及處理敏感資料的應用均構成限制。

2025 至 2026 年間，多模態模型的成熟使情況出現轉機：若開源且可於本地執行的 VLM 確實堪用，
小型組織即得以自主完成文件數位化，無須將資料送交外部服務。惟前提在於能夠判斷其是否
堪用，而本研究處理的即為此一判斷。

### 計畫的組成

計畫原先規劃兩條並行的研究路線，即本專案所屬的評測路線，以及 Gemma 4 台灣領域微調路線；
後者最終未執行，規劃文件整理於 [ROADMAP.md](ROADMAP.md)。

三位實習生參與，分別就讀人類學、圖書資訊學與地質學，均非資訊工程背景。其中兩位執行評測
研究，第三位負責前期的硬體評估與詢價。

算力方面採先租後買的策略：評測全部執行於 AWS 的單張 NVIDIA L40S，同時評估自建本地工作站
的可行性。本地工作站最終未依評估規格採購，改為委請廠商規劃，評估過程與實際報價整理於
[HARDWARE.md](HARDWARE.md)。

---

## 研究摘要

本研究的主要貢獻並非模型能力的排行榜，而在於評測方法本身：

> 開源 VLM 的 OCR 評測分數主要取決於評測設定，而非模型能力。在模型與資料完全相同的
> 條件下，僅調整後處理與生成參數，Exact Match 即可在 **0% 至 64.84%** 之間變動。

上述結論由一組單變因消融實驗（one-factor-at-a-time, OFAT）證實，完整證據保存於
[`ablation_experiment/`](ablation_experiment/)。研究方向亦因而調整：與其累積彼此不可
比較的分數，不如將「何種設定已被固定」納入評測系統的核心設計。後半段的
[`benchmark_suite/`](benchmark_suite/) 與 [`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/)
即為該構想的實作。

---

## 核心發現

以下數據均取自同一模型（GLM-OCR Q8_0 GGUF）、同一資料集（TC-STR test split，
3,706 筆）與同一台主機。每組實驗僅更動一項設定，其餘條件維持 baseline 不變；所有需要
推論的 variant 均完整執行 3,706 筆，API error 數為 0。

| 實驗（相對 baseline 的唯一更動） | EM | CM | ANLS | F1 | 平均延遲 |
|---|---:|---:|---:|---:|---:|
| Aligned baseline | 47.06% | 76.52% | 54.54% | 59.17% | 0.602 s |
| 改用 `/api/chat` | 47.06% | 76.52% | 54.54% | 59.17% | 0.582 s |
| 改用短 prompt | **0.00%** | 79.06% | **0.00%** | 13.56% | 0.594 s |
| 不指定 `repeat_penalty=1.6` | **64.84%** | **82.46%** | **72.08%** | **76.62%** | 0.583 s |
| `num_predict` 由 80 降至 25 | 46.03% | 76.34% | 53.29% | 65.07% | **0.293 s** |
| 改用最小後處理 | **0.00%** | 76.79% | **0.00%** | 6.88% | 離線重算 |
| 改用另一套 scorer | 47.87% | 77.63% | 55.02% | 59.96% | 離線重算 |

### 一、repeat_penalty 對辨識結果的影響

在其餘條件不變的前提下，僅取消顯式傳入 `repeat_penalty=1.6`、回歸 Ollama 與模型的
預設值，Exact Match 由 47.06% 上升至 64.84%，增加 17.78 個百分點，ANLS 亦增加
17.53 個百分點。逐題比對顯示 779 題由錯轉對、120 題由對轉錯，淨增 659 題完全正確。
就本次實驗而言，取消該參數為效益最高的單一調整，且不涉及更換模型、增補資料或延長
推論時間。

### 二、EM 歸零與輸出格式失控的區別

採用短 prompt 的實驗組 EM 與 ANLS 均為 0.00%，但 CM（預測是否包含正確答案）達
79.06%，為全部實驗組的最高值。兩項指標的落差顯示模型已辨識出文字，惟於答案之後附加
解釋、Markdown 標記、英文分析與 prompt 複誦，致使嚴格比對全數失敗。

由此可知，觀察各項指標之間的相對關係，較單一數值更具診斷價值。若兩份報告呈現 EM 與
ANLS 差距顯著、CM 與 F1 相近的形態，差異來源多半在於後處理或輸出格式，而非辨識能力
本身。

### 三、後處理在評分流程中的地位

aligned 後處理修改了 baseline 3,706 筆 raw response 之中的 3,704 筆。同一批 raw
response 改以最小後處理計分後，EM 隨即歸零。現行流程之所以能取得合理分數，相當程度
取決於後處理規則，而非模型輸出本身已足夠乾淨。

### 四、API endpoint 的等價性

將 `/api/generate` 改為 `/api/chat` 之後，3,706 筆 processed prediction 逐字相同，
四項指標亦無變化。平均延遲相差 0.020 秒，不足以據此判定何者較快。endpoint 可自
「分數差異的可能原因」清單之中排除。

### 五、評分器造成的報告差異

在 prediction 完全未變動的情況下更換 scorer，EM 多判定 30 題正確，增加 0.81 個百分點。
差異來自兩套 scorer 對空白與標點的正規化方式不同，以及 CM 採單向或雙向判定的定義不同。
此類差異屬於報告層次，與模型能力無關；跨團隊比較分數之前，應先對齊 scorer。

### 六、輸出長度與延遲的取捨

`num_predict` 由 80 降至 25，平均延遲縮短約 51%（0.602 秒降至 0.293 秒），代價為 EM
下降 1.03 個百分點、ANLS 下降 1.25 個百分點。惟遭截斷的 prompt 複誦有可能因而避開
後處理規則，故不宜逕自解讀為低成本的加速手段。

> **解讀限制**：本實驗為 OFAT 單變因主效應分析，各列的 delta 不可直接相加。所有數字
> 僅代表 GLM-OCR Q8、TC-STR 與當時的 Ollama 環境，未必可推廣至其他模型。CM 僅檢查
> 預測是否包含 ground truth，答案夾雜於大量雜訊之中仍可得分，不宜單獨用以判斷 OCR
> 品質。

---

## 成果地圖

五個目錄對應實習的五個階段，依時間順序閱讀最易掌握脈絡。

```
2026-07-15  eval/                 建立評測流程，並提出分數分歧的問題
2026-07-16  Sixhuang/             獨立實作第二套流程，形成對照組
2026-07-21  ablation_experiment/  以 3,706 筆 × 7 組單變因實驗回答上述問題
2026-07-28  benchmark_suite/      將結論制度化：5 模型 × 3 benchmark 無人值守系統
2026-07-30  TC-STR_and_OminDoc/   擴大規模：8 模型 OCR 與官方 evaluator 文件解析
   ～08-14
```

| 目錄 | 作者 | 內容 | 結果數據 |
|---|---|---|---|
| [`eval/`](eval/) | [@trickster-2005](https://github.com/trickster-2005) | TC-STR 三模型評測流程，含後處理、四項指標與互動式 HTML 報告；本次實習的核心問題即由此提出 | 已進版控 |
| [`Sixhuang/`](Sixhuang/) | [@hyslchs](https://github.com/hyslchs) | 獨立實作的第二套跑分程式，另納入 Hugging Face 合成資料集。與 `eval/` 之間的三項差異（短 prompt、最小後處理、不同 scorer）後續成為消融實驗的受測變因 | 未進版控 |
| [`ablation_experiment/`](ablation_experiment/) | [@hyslchs](https://github.com/hyslchs) | **本專案最重要的成果。** 對齊兩套流程後執行 7 組 OFAT 單變因消融，合計 18,530 次推論 | 已進版控，完整 |
| [`benchmark_suite/`](benchmark_suite/) | [@hyslchs](https://github.com/hyslchs) | 可長時間無人值守、append-only 斷點續跑的評測系統。5 模型 × 3 benchmark（OmniDocBench／TC-STR／VisTW-MCQ），共 15 組正式結果 | 系統已完成，正式結果未產出 |
| [`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/) | [@hyslchs](https://github.com/hyslchs) | 規模最大的一套。TC-STR 涵蓋 8 個模型；OmniDocBench v1.6 全量 1,651 頁，以釘死版本的官方 Docker evaluator 計分，另備 9 個模型設定與多模型比較報告工具 | 保存於 EC2 的 EBS，未進版控 |

### 結果數據的保存狀況

接手研究者須留意，多數實驗的原始結果並未收錄於本專案。

版控之內實際保存的結果僅有兩份：

- `eval/results/report.html`，三模型評測報告，為自包含 HTML，可直接開啟。
- `ablation_experiment/results/`，消融實驗的完整成果，包含
  [`summary.json`](ablation_experiment/results/summary.json)（七組結果的機器可讀彙總）、
  [`run_manifest.json`](ablation_experiment/results/run_manifest.json)（模型、prompt、
  options 與資料 fingerprint）、
  [`dataset_manifest.json`](ablation_experiment/results/dataset_manifest.json)（3,706 張
  圖片的順序、ground truth 與 SHA-256），以及
  [`ablation_report.html`](ablation_experiment/results/ablation_report.html)（完整互動
  報告與逐題差異案例）。

`benchmark_suite/` 與 `TC-STR_and_OminDoc/` 交付的是執行系統，而非結果。兩者的正式
輸出寫入執行主機（AWS EC2）的持久磁碟，未隨專案散布，屬於刻意的設計：兩套系統均要求
正式結果必須連同 model digest、dataset fingerprint、evaluator commit 與 effective
config 一併保存，否則不具可信度。逐筆 CSV 與 raw response 的體積亦相當可觀，消融實驗
的逐筆 CSV 即約 17 MB。

兩套系統均可重新執行並產生等價輸出，亦為其將所有設定釘死的目的所在。

---

## 評測基礎建設的設計原則

後半段的兩套系統承接自消融實驗的結論。既然分數易受設定影響，評測系統即須確保設定無法
被隱蔽地更動。兩者共同遵循的原則如下：

- **設定納入身分識別**：resume key 與 run signature 包含 config hash、實際 model
  digest、dataset revision、prompt hash 與 dependency lock SHA256，其中任一項不同即
  視為另一次實驗，不與舊結果混用。
- **失敗必須可見，不得補齊**：模型自身造成的空輸出、拒答、重複、格式錯誤與
  `done_reason=length` 一律保留並照常計分，僅標示為 anomaly；唯有非模型因素的工程
  錯誤（資料 signature 不符、模型無法載入、非 100% GPU、evaluator 失敗）才會擋下評測。
- **不以自製分數頂替官方分數**：OmniDocBench 的官方 evaluator 失敗時，prediction 保留、
  scoring 明確標為 `blocked`，不改用近似值。
- **官方指標與診斷指標分區**：EM、CM、ANLS、F1 於 OmniDocBench 報告中明確標示為
  「非官方 leaderboard 指標」，不與官方 Edit Distance、TEDS、CDM 混用。
- **raw response 永久保存**，主評分不採用依模型客製的清理規則。
- **例外必須留下紀錄**：GLM OCR BF16 於 Ollama 會回傳有效文字但 `done=false` 且缺少
  token metadata，該項豁免明文記載於
  [`PROTOCOL_EXCEPTIONS.md`](TC-STR_and_OminDoc/TC_STR/PROTOCOL_EXCEPTIONS.md)，並註明
  其 token 效率不可與其他模型等量比較。

`benchmark_suite/` 的固定推論設定（`temperature=0`、`seed=42`、`repeat_penalty=1.0`、
concurrency=1）承接自消融實驗的實證。文件亦一併註明，`repeat_penalty=1.0` 係為公平與
可重現而選定的控制值，並非已證明為所有模型的最佳值。

---

## 評測對象

### 資料集

| 資料集 | 內容 | 規模 | 使用於 |
|---|---|---|---|
| [TC-STR 7k-word](https://github.com/esun-ai/traditional-chinese-text-recogn-dataset) | 繁體中文場景文字辨識（招牌、看板等 crop） | test split 3,706 張 | 四項 OCR 實驗全部 |
| [OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench) | 整頁文件解析（文字、表格、公式、閱讀順序） | full set 1,651 頁 | `TC-STR_and_OminDoc/`（v1.6，revision `d386947f`）、`benchmark_suite/`（v1.7，revision `aa1ee96d`） |
| [VisTW-MCQ](https://huggingface.co/datasets/miulab/vistw-mcq) | 繁體中文視覺選擇題 | 21 subjects 全部 test | `benchmark_suite/` |
| [ZihCiLin/traditional-chinese-ocr-synthetic](https://huggingface.co/datasets/ZihCiLin/traditional-chinese-ocr-synthetic) | 繁中 OCR 合成資料 | `test_random`、`test_semantic` | `Sixhuang/` |

### 模型

橫跨五個階段，累計評測或設定過的開源 VLM 如下：

- **Gemma 4 系列**（QAT Q4_0）：E2B、E4B、12B、26B-A4B、31B
- **Qwen 系列**：Qwen2.5-VL 3B、Qwen3-VL 4B 與 32B
- **GLM 系列**：GLM-OCR Q8_0 GGUF、GLM-OCR BF16、GLM-4.6V-Flash 9B
- **InternVL3.5**：4B、38B（Q4_K_M）
- **其他**：Kimi-VL-A3B-Instruct、SmolVLM2 2.2B、Qwen-SEA-LION v4 4B-VL、chandra-ocr-2

第三方 GGUF 量化版本（非模型原廠發布）的來源、風險與核准紀錄，逐一記載於
[`MODEL_CANDIDATES.md`](TC-STR_and_OminDoc/TC_STR/MODEL_CANDIDATES.md)。相關資訊於
評測報告中經常遭到忽略，卻直接影響結果能否歸因至原始模型。

### 執行環境

後期實驗所用主機（`TC-STR_and_OminDoc/` 建立與驗證期間，2026-07-29 至 07-30）：

| 項目 | 規格 |
|---|---|
| Instance | AWS EC2 `g6e.2xlarge`／Ubuntu 26.04 LTS |
| CPU 與 RAM | AMD EPYC 7R13（8 vCPU）／61 GiB |
| GPU | NVIDIA L40S，46 GB VRAM（driver 610.43.02） |
| 儲存 | 419 GB instance store（不跨 stop/start 保留）與 EBS（持久） |
| Ollama | 0.31.1 |
| Docker | 29.6.2（供 OmniDocBench 官方 evaluator 使用） |

所有推論的 batch size 與 concurrency 均固定為 1，模型依序載入並於階段結束後卸載，以免
互相爭用 VRAM；正式推論之前必須確認 `ollama ps` 顯示 `100% GPU`，否則所得數字不可採用。

完整的主機規格、儲存架構、Ollama 與 Docker 設定、多人共用權限及 stop/start 復原步驟，
詳見 [INFRASTRUCTURE.md](INFRASTRUCTURE.md)。上述配置決策並非單純的維運細節，而是直接
寫入了兩套 runner 的設計之中。

---

## 閱讀指引

| 目的 | 建議起點 |
|---|---|
| 掌握結論 | 本頁〈核心發現〉，續讀 [`ablation_experiment/RESULTS.md`](ablation_experiment/RESULTS.md) |
| 檢視逐題證據 | 以瀏覽器開啟 [`ablation_report.html`](ablation_experiment/results/ablation_report.html) 與 [`report.html`](eval/results/report.html)，兩者均為自包含檔案，無須啟動 server |
| 理解方法論的演進 | 依 `eval/` → `Sixhuang/` → `ablation_experiment/` → `benchmark_suite/` → `TC-STR_and_OminDoc/` 的順序閱讀各目錄 README |
| 自行重現實驗 | 各目錄 README 均載有環境需求與指令；最易重現者為 [`ablation_experiment/`](ablation_experiment/)，僅需 Ollama、單一模型與 TC-STR |
| 了解執行環境 | [INFRASTRUCTURE.md](INFRASTRUCTURE.md) |
| 估算硬體與成本 | [HARDWARE.md](HARDWARE.md)，含顯存試算、AWS 機型與成本、七套工作站報價，以及機房用電與電費分攤方式 |
| 接手或參與 | [ROADMAP.md](ROADMAP.md) |
| 認識研究團隊 | [CONTRIBUTORS.md](CONTRIBUTORS.md) |

---

## 計畫的另一半：未執行的研究路線

本實習計畫原先規劃兩條並行的研究路線：

| | 路線 A（即本專案） | 路線 B |
|---|---|---|
| 題目 | 開源 VLM 的繁中 OCR 與文件解析**評測** | Gemma 4 台灣領域**微調** |
| 採用的基準 | TC-STR、OmniDocBench、VisTW-MCQ | TMMLU+、TMLU、TC-Eval |
| 狀態 | 已完成 | 僅完成規劃，未執行 |

兩條路線互為前提：唯有先能公平地量測，才有資格判斷微調是否有效。路線 A 的核心發現
恰可說明路線 B 的困難所在：baseline 若無法測準，「微調後進步 3%」一語便失去意義。

路線 B 最終無人執行，惟規劃文件相當完整，包含 8 週路線圖、可逐行照跑的 Phase 0
runbook，以及雲端自開機至關機的完整指令。與其留存於內部硬碟，不如一併整理公開。

該路線目前仍為空白，歡迎接手，詳見 [ROADMAP.md](ROADMAP.md)。

### 路線 A 尚未完成的部分

依價值排序如下。前兩項僅需執行，程式均已完成：

1. **執行 [`benchmark_suite/`](benchmark_suite/) 的 15 組正式結果。** 系統已可運作，
   `runs/` 目前為空，投入產出比最高。
2. **執行 [`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/)。** 涵蓋 8 模型 TC-STR 與
   OmniDocBench 全量 1,651 頁。
3. **`repeat_penalty` 的細部 sweep。** 消融實驗僅證明 1.6 明顯有害，尚未定位最佳值。
4. **檢驗變因之間的交互作用。** OFAT 無法捕捉交互作用，而 prompt 與後處理、
   `num_predict` 與後處理之間均有理由懷疑存在交互作用。
5. **於其他模型上驗證消融結論。** 目前結論僅在 GLM-OCR Q8 上成立，Gemma 4 與
   Qwen3-VL 是否具有同樣的參數敏感性，仍屬開放問題。

完整待辦清單、接手指引，以及依投入程度分級的參與方式（自十分鐘至兩個月），見
[ROADMAP.md](ROADMAP.md#五如何參與)。

---

## 研究團隊

三位實習生參與本計畫，均非資訊工程背景，分別就讀人類學、圖書資訊學與地質學。學科背景
的差異，相當程度說明了本計畫最終呈現的樣貌。

| | 背景 | 負責範圍 |
|---|---|---|
| **游聿堂**<br>[@trickster-2005](https://github.com/trickster-2005) | 臺大人類學系，語言學與計算語言學，中研院 Depositar Lab 開放資料實習 | [`eval/`](eval/)：最早的評測流程、後處理設計、四項指標實作與 HTML 報告 |
| **黃以信**<br>[@hyslchs](https://github.com/hyslchs) | 輔大圖書資訊學系，語意檢索與向量資料庫，NSTC 大專學生研究計畫 | [`Sixhuang/`](Sixhuang/)、[`ablation_experiment/`](ablation_experiment/)、[`benchmark_suite/`](benchmark_suite/)、[`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/) |
| **林柏儒** | 臺大地質系，GenAI 落地與本地模型部署、自動化工作流 | 計畫前期的硬體評估與詢價，成果見 [HARDWARE.md](HARDWARE.md) |

AI 評測領域長期欠缺兩項素養：對「測量如何被建構」的懷疑，以及對「來源與版本」的紀律。
兩者恰分別屬於人類學與開放資料治理，以及圖書資訊學的核心訓練。

游聿堂承擔了前者。他在 `eval/README.md` 寫下的〈為什麼跟別人的結果可能不一樣〉一節，
將六項可能原因依影響程度排序，並提出可操作的診斷方法，關切的是測量如何被建構，而非
分數的高低。三週之後，以 18,530 次推論完成的消融實驗證實了他的排序判斷正確。

黃以信承擔了後者。他所建立的兩套系統，核心在於來源追溯：每一項分數均須能回溯至確切的
model digest、dataset fingerprint、prompt hash 與 evaluator commit，例外須具名核准，
第三方量化版本的來源亦須逐一存證。相關作法在圖書資訊學之中自有名稱，即編目、權威控制
與來源證明。

林柏儒承擔的是落地的成本評估。他有實際在本地執行開源模型的經驗，清楚顯存不足時的後果，
因而由他試算顯存需求、比較七套機型並向廠商實際詢價。OCF 最終未依該份規格採購，惟評估
本身提供了議價與判斷的基準，並整理為 [HARDWARE.md](HARDWARE.md)，可供其他非營利組織
參考。

完整背景、逐項產出、工作紀錄與可驗證的工作量統計，見
[CONTRIBUTORS.md](CONTRIBUTORS.md)。

兩位實習生的工作原先分別在 `main` 與 `Sixhuang` 分支進行，已於 2026 年 8 月 19 日透過
[PR #1](https://github.com/ocftw/2026-OCF-Intern/pull/1) 合併。`main` 目前包含全部研究
成果，`Sixhuang` 分支僅作為歷史紀錄保留，內容與 `main` 完全一致。

---

## 授權與使用限制

本專案的程式碼由 OCF 2026 AI 研究實習計畫產出。資料集與模型不隨專案散布，使用者須自行
向上游取得，並遵守各自的授權條款。以下數項尤須留意：

- **OmniDocBench** 官方聲明資料僅供研究使用，不得商業利用。
- **TC-STR** 的 repository 採 Apache-2.0 授權，惟所下載的資料仍應依來源條款使用。
- **VisTW** 的 dataset card 與 evaluation repository 未給出足夠明確的資料授權，正式
  下載與發表之前應向資料提供者確認。
- **第三方 GGUF 量化版本**（`mradermacher`、`haervwe` 等社群發布）的再散布與使用條款
  須逐一確認，不得逕自假設等同於原始模型的授權。

引用評測數字時，務請一併引用其執行條件，此亦為本研究最主要的結論。
