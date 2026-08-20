# 研究團隊

[開放文化基金會（OCF）](https://ocf.tw/)2026 年 AI 研究實習計畫，實習期間為 2026 年
7 月 1 日至 8 月 30 日。

本頁記錄三位實習生的背景、負責範圍與具體產出。研究成果本身見
[根目錄 README](README.md)。

---

## 學科背景與成果的關聯

三位實習生均非資訊工程背景，分別就讀人類學、圖書資訊學與地質學。其中兩位（游聿堂、
黃以信）執行本專案的評測研究，第三位（林柏儒）負責計畫前期的硬體評估與詢價。

AI 評測領域長期欠缺兩項素養：對「測量如何被建構」的懷疑，以及對「來源與版本」的紀律。
兩者恰分別屬於人類學與開放資料治理，以及圖書資訊學的核心訓練。

- 游聿堂承擔前者。他所提出的問題並非哪一個模型分數較高，而是同一模型何以在不同人手上
  得出不同分數，並將可能原因依影響程度排序，提出可操作的診斷方法。
- 黃以信承擔後者。他所建立的兩套系統，核心在於來源追溯：每一項分數均須能回溯至確切的
  model digest、dataset fingerprint、prompt hash 與 evaluator commit。
- 林柏儒承擔落地的成本評估。他有實際在本地執行開源模型的經驗，因而由他試算顯存需求、
  比較機型並向廠商詢價，所得並非型錄數字，而是可用於議價的基準。

最終產出的成果為一項方法論的發現，而非排行榜：開源 VLM 的 OCR 分數主要取決於評測設定，
而非模型能力。對公民科技而言，另有一層意義在於，此一領域所需要的並不只是工程師。

---

## 游聿堂（YOU, Yu-Tang）

[@trickster-2005](https://github.com/trickster-2005)

### 背景

國立臺灣大學**人類學系**三年級，修讀語言學（系必修）、計算語言學，以及資訊法與資訊
政策專題。2025 年於**中央研究院資訊科學研究所 Depositar Lab**（研究資料寄存與開放資料
平台）暑期實習。曾為第五、六屆**臺灣好厲駭**培訓學員（教育部先進資通安全實務人才培育
計畫，2020 至 2022 年）。

長期參與開源社群並多次公開發表：

| 年份 | 場合 | 講題 |
|---|---|---|
| 2026 | SITCON | 〈地圖與權力：公眾地理資訊系統應用〉 |
| 2025 | SITCON | 〈因為反抗，所以存在：監控資本主義下的自由如何可能？〉 |
| 2024 | SITCON | 〈科技藝術入門 — 淺談 p5.js〉 |
| 2024 | COSCUP | 〈資料視覺化 — 新手也能學會的 D3.js 實戰〉 |

技術背景：Web（HTML、CSS、JS）、Git、Linux、Python、GIS（QGIS 與 ArcGIS）。

### 負責範圍

[`eval/`](eval/)，為整個實習最早的評測程式，亦為後續所有實驗的起點。

- 建立完整流程：Ollama VLM 呼叫、後處理、四項指標評分至互動式 HTML 報告
- 設計 [`postprocess.py`](eval/postprocess.py) 的雜訊清理規則，涵蓋 prompt 複誦、
  `<think>` 推理標籤、HTML 標籤、JSON 包裝、code fence 與重複字元迴圈
- 實作 EM、CM、ANLS、F1 四項指標（[`metrics.py`](eval/metrics.py)）
- 撰寫一頁式自包含 HTML 報告（[`build_report.py`](eval/build_report.py)），含模型總覽
  卡片、文字長度與圖片類別對 ANLS 的影響，以及可篩選、排序、搜尋的逐張圖明細
- 撰寫長時間無人值守的執行腳本（[`run_until_done.sh`](eval/run_until_done.sh)），
  支援斷線重試

產出計 1,312 行 Python 與 Shell，以及唯一一份完整保存的三模型評測報告
[`eval/results/report.html`](eval/results/report.html)。

### 主要貢獻

`eval/README.md` 之中有一節名為〈為什麼跟別人的結果可能不一樣〉，其中完成三件事：

1. 指出同一模型與同一份資料，不同人執行所得分數可自 ANLS 0% 至 55% 不等
2. 將六項可能原因依影響程度排序，並將「後處理清理程度」列為首位
3. 提出可操作的診斷方法：比對雙方的 `prediction_raw`。原始輸出相同而清理後的
   `prediction` 不同者，屬後處理問題；原始輸出即已不同者，則應往 prompt 與推論參數
   追查

上述關切的並非工程實作，而是方法論層次的懷疑，亦即測量如何被建構，而非分數的高低。
人類學與開放資料治理的訓練核心正在於此。

三週之後，以 3,706 筆資料、7 組變因、18,530 次推論完成的
[消融實驗](ablation_experiment/RESULTS.md)證實了該項排序判斷：後處理修改了 baseline
3,706 筆 raw response 之中的 3,704 筆，改用最小後處理之後 EM 隨即歸零。

一位人類學系三年級學生，以學科直覺指出了整個計畫最重要的發現，該問題後續成為整份研究
的主軸。

### 工作紀錄

| 日期 | 內容 |
|---|---|
| 2026-07-15 | 建立 TC-STR OCR 模型評估流程 |
| 2026-07-15 | 撰寫各檔案用途說明，並提出跨團隊分數分歧的成因分析 |

---

## 黃以信（HUANG, Yi-Hsin）

[@hyslchs](https://github.com/hyslchs)｜[sixhuang.com](https://sixhuang.com)｜社群暱稱：阿六

> 本專案 [`Sixhuang/`](Sixhuang/) 目錄的名稱即來自其社群 handle。

### 背景

輔仁大學**圖書資訊學系**三年級，學習方向為圖書資訊學、自然語言處理、數據分析與機器
學習。

**NSTC 大專學生研究計畫**（研究者：黃以信），題為〈建置基於語意相似性的書籍推薦系統〉：
以輔仁大學圖書館約 30 萬筆中文書目為基礎，使用 `text-embedding-3-large` 產生向量，並以
Chroma DB 進行相似度檢索，處理館藏系統以分類號為主的相關書推薦在跨主題探索上的限制。
該研究仍在延伸，預計投稿圖書資訊學相關期刊。

**AI CUP 2026 ESG 永續承諾驗證競賽**（進行中），涉及文字分類、資訊擷取與模型評估方法，
實作 TF-IDF、Transformer 多任務分類、BERT 與錯誤分析。

長期參與開源社群，以編輯與文件工作為主：

| 年份 | 角色 |
|---|---|
| 2026 | SITCON 編輯組員與行銷組員、**SITCON 2026 講者**、SITCON Camp 編輯組長、g0v summit 社群宣傳小組 |
| 2025 | SITCON 編輯組長、SITCON Camp 編輯組員 |
| 2024 | SITCON 編輯組、SITCON Camp 編輯組長 |

- SITCON 2026 講題：〈在書海中的航行指南；打開你對圖書館的全新視野！〉
  （[議程](https://sitcon.org/2026/agenda/dbd911/)）
- [SITCON 電子月刊](https://medium.com/sitcon.tw)撰稿

技術背景：Python、pandas、JSON 與 CSV 資料處理；Word Embedding、語意相似度、向量檢索、
TF-IDF、Transformer；向量資料庫、Markdown、Git 與 GitHub、研究紀錄整理；技術文件撰寫、
社群協作與跨角色溝通。

### 負責範圍

四個目錄，涵蓋實習的第二至第五階段。

**[`Sixhuang/`](Sixhuang/)（2026-07-16）**，與 `eval/` 各自獨立實作的第二套 TC-STR
跑分程式，另納入 Hugging Face 繁中合成資料集。兩套流程之間的差異（短 prompt、最小後
處理、不同 scorer 定義）後續直接成為消融實驗的受測變因。程式計 906 行。

**[`ablation_experiment/`](ablation_experiment/)（2026-07-21）**，本專案最重要的成果。
將兩套獨立流程對齊為共同 baseline，執行 one-factor-at-a-time 單變因消融：GLM-OCR Q8
搭配 TC-STR 3,706 筆資料，共 7 組 variant，其中 5 組需要推論，合計 18,530 次 Ollama
request，全數完成且 API error 數為 0。程式計 923 行。

所得結論包括：取消 `repeat_penalty=1.6` 使 EM 自 47.06% 上升至 64.84%，增加 17.78 個
百分點，淨增 659 題完全正確；短 prompt 使 EM 與 ANLS 歸零，惟 CM 反而升至 79.06%，
顯示文字已辨識而輸出格式失控；`/api/generate` 與 `/api/chat` 的 3,706 筆 prediction
逐字相同。

**[`benchmark_suite/`](benchmark_suite/)（2026-07-28）**，可長時間無人值守、append-only
斷點續跑的評測系統，涵蓋 5 模型 × 3 benchmark（OmniDocBench、TC-STR、VisTW-MCQ），
共 15 組正式結果。程式計 4,010 行，另有 12 個測試檔。

**[`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/)（2026-07-30 至 08-14）**，規模最大的一套。
TC-STR 涵蓋 8 個模型；OmniDocBench v1.6 全量 1,651 頁，以釘死版本的官方 Docker
evaluator 計分，另備 9 個模型設定與多模型互動比較報告工具。程式計 10,105 行，另有
12 個測試檔與 950 行文件。

### 主要貢獻

兩套評測系統的核心並非跑分，而在於來源追溯：

- resume key 與 run signature 包含 config hash、實際 model digest、dataset revision、
  prompt hash 與 dependency lock SHA256，其中任一項不同即視為另一次實驗，不與舊結果
  混用
- [`VERSION_PINNING.md`](TC-STR_and_OminDoc/OminDocBench/VERSION_PINNING.md) 釘死資料
  revision 與 evaluator commit，並拒絕混用 Docker image 內建的舊版 GT
- [`PROTOCOL_EXCEPTIONS.md`](TC-STR_and_OminDoc/TC_STR/PROTOCOL_EXCEPTIONS.md) 要求
  例外須具名核准、載明日期與影響範圍，並註明哪些欄位因而不可等量比較
- [`MODEL_CANDIDATES.md`](TC-STR_and_OminDoc/TC_STR/MODEL_CANDIDATES.md) 逐一存證第三方
  GGUF 量化版本的來源、發布者、宣稱上游、風險與核准狀態。由 `mradermacher` 發布的
  Kimi-VL 量化版並不等同於 Moonshot AI 的 Kimi-VL，其間的區別直接影響結果能否歸因至
  原始模型
- 官方指標與自製診斷指標嚴格分區，evaluator 失敗時 scoring 標為 `blocked`，不以近似值
  頂替

相關作法在圖書資訊學之中自有名稱，即編目、權威控制與來源證明（provenance）。將圖資學
的紀律移入 ML 評測，恰補上了該領域最欠缺的一環。

多年 SITCON 編輯組長的訓練亦直接反映於文件品質：`TC-STR_and_OminDoc/` 單一目錄即有
950 行文件，涵蓋方法論、指標定義、版本釘選與例外紀錄。實習後期他更主動回頭修正 README
的準確性（model-set 數量、InternVL 廠商歸屬、`ensure_official_configs` 的實際行為），
並將中英文混雜的說明統一為繁體中文。

### 工作紀錄

| 日期 | 內容 |
|---|---|
| 2026-07-16 | 建立獨立的 OCR benchmark 程式 |
| 2026-07-21 | GLM-OCR 消融實驗與報告；補齊完整消融結果文件 |
| 2026-07-28 | 建立可重現 VLM 評測系統；進度顯示、暫存空間配置、HF 下載重試、Python 3.14 相容、分開執行與樣本限制，共 7 次提交 |
| 2026-07-30 | TC-STR 與 OmniDocBench 執行工具；方法論文件 |
| 2026-08-06 | 修正硬編碼路徑；多模型比較報告工具與共用權限修正腳本；共用多人設定文件 |
| 2026-08-14 | 加入 Qwen3-VL 與 InternVL3.5 模型測試；修正 README 準確性；統一為繁體中文 |

---

## 林柏儒（LIN, Bo-Ru）

### 背景

國立臺灣大學（2021 至 2026 年），主修**地質學系**，輔修生物產業傳播暨發展學系及傳播
學程。2024 年於**國家災害防救科技中心**暑期實習，彙整 11 個國家的自然災害防治與救助
方案進行跨國比較。曾任臺大地質標本館導覽員。

技術背景偏向 GenAI 落地與自動化工作流：ComfyUI 開發、n8n 跨系統自動化（整合 GCP、
Google Workspace 及多個社群平台 API）、**開源多模態模型的本地端部署**（曾為教育機構
建立滿足隱私要求的會議轉錄與摘要方案）。另有 GenAI 自由接案與 AI 工具教學經驗。

### 負責範圍

計畫前期的硬體評估與詢價，成果整理於 [HARDWARE.md](HARDWARE.md)。

- 各模型與訓練方法的**顯存需求試算**（QLoRA 4-bit、LoRA 16-bit、全參數微調），以及每
  參數記憶體成本的拆解，說明全參數微調的記憶體成本何以達 QLoRA 的約 29 倍
- **七套工作站配置的比較與實際廠商詢價**：1 至 4 卡，AMD 與 Intel 兩種平台，自
  NT$65 萬的單卡家用桌機至 NT$277 萬的四卡工作站
- 連帶的規格判斷：VRAM 與 RAM 的比例、電源與 220V 供電需求、機箱尺寸與散熱形式

### 評估的價值

OCF 最終未依該份規格採購，改為委請廠商規劃整機。

評估本身仍具價值，其作用在於提供議價與判斷的基準。掌握 GPU 佔總價六至七成、2 卡與
4 卡的實際差距，以及電源必須為 220V 並將連動機房配電等資訊之後，方能判斷廠商所開整機
報價是否合理、規格有無遭到替換。

本地部署的背景與此項工作直接相關：評估應購置何種顯示卡者，最好是實際在本地執行過開源
模型的人，因其清楚顯存不足時的後果。

該工作亦連向計畫的另一半。本專案的評測全部執行於 AWS 雲端單卡，正因本地機尚未到位；
[HARDWARE.md](HARDWARE.md) 之中「先租，不要先買」一節，即為實際走過一遍之後的結論。

---

## 工作量

以下數字均可自本專案直接驗證。

| 項目 | 數量 |
|---|---|
| 實習期間 | 2026 年 7 月 1 日至 8 月 30 日，約 9 週 |
| 版控紀錄涵蓋 | 2026 年 7 月 15 日至 8 月 19 日 |
| 程式碼 | 15,909 行 Python（90 檔）與 1,347 行 Shell（34 檔） |
| 文件 | 2,196 行 Markdown（14 份） |
| 測試 | 29 個測試檔 |
| 設定檔 | 19 份 YAML 與 JSON |
| 完成推論 | 18,530 次（消融實驗），全數成功，API error 數為 0 |
| 評測資料規模 | TC-STR 3,706 張 × 7 組變因；OmniDocBench 全量 1,651 頁 |
| 涵蓋開源 VLM | 16 個（Gemma 4 五種尺寸、Qwen2.5-VL 與 Qwen3-VL、GLM 三種、InternVL3.5 兩種、Kimi-VL、SmolVLM2、SEA-LION、chandra-ocr-2） |

程式碼行數僅反映投入，不反映價值。本研究最重要的成果為一項
[方法論的發現](README.md#核心發現)，其證據為 18,530 次推論，而非 15,909 行程式。

---

## 分支與合併紀錄

兩位實習生的工作原先分別在 `main` 與 `Sixhuang` 分支進行，於 2026 年 8 月 19 日透過
[PR #1](https://github.com/ocftw/2026-OCF-Intern/pull/1) 合併。`main` 目前包含全部研究
成果，`Sixhuang` 分支僅作為歷史紀錄保留，內容與 `main` 完全一致（`git diff main
origin/Sixhuang` 為空）。

---

*本頁僅收錄三位實習生履歷之中原本即已公開的資訊，包含就讀學校科系、公開演講與議程連結、
公開研究計畫、GitHub 帳號與個人網頁。聯絡方式未列於此，如需聯繫請透過 OCF 或其 GitHub
個人頁面。*
