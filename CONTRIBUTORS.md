# 研究團隊

[開放文化基金會（OCF）](https://ocf.tw/)2026 年 AI 暑期實習計畫，2026-07-15 ～ 08-19。

本頁記錄三位實習生的背景、負責範圍與具體產出。研究成果本身見
[根目錄 README](README.md)。

---

## 這個團隊有意思的地方

三位實習生參與了這個計畫，**沒有一位是資訊工程背景**——分別讀人類學、圖書資訊學與
地質學。這件事不是註腳，它直接解釋了這個計畫為什麼長成現在的樣子。

其中兩位（游聿堂、黃以信）執行了本 repo 的評測研究，第三位（林柏儒）負責計畫前期的
硬體評估與詢價。

AI 評測領域長期缺兩樣東西：**對「測量本身如何被建構」的懷疑**，以及**對「來源與版本」
的紀律**。這兩件事恰好分別是人類學／開放資料治理，與圖書資訊學的核心訓練。

- 游聿堂帶進了前者。他問的不是「哪個模型分數高」，而是「為什麼同一個模型在不同人手上
  分數不一樣」——並且把可能原因依影響程度排序、提出可操作的診斷方法。
- 黃以信帶進了後者。他建的兩套系統核心不是跑分，是**來源追溯**：每一個分數都要能追回
  到確切的 model digest、dataset fingerprint、prompt hash 與 evaluator commit。

- 林柏儒帶進的是**落地的成本感**。他實際在本地跑過開源模型，知道顯存不夠時會發生
  什麼事，所以由他去試算顯存需求、比較機型、向廠商詢價，得到的不是型錄數字而是可以
  拿來議價的基準。

結果是一份成果：**開源 VLM 的 OCR 分數，主要由評測設定決定，而非模型能力**——一個
方法論的發現，而不是一張排行榜。這對公民科技有額外的意義：它說明這個領域需要的
不只是工程師。

---

## 游聿堂（YOU, Yu-Tang）

[@trickster-2005](https://github.com/trickster-2005)

### 背景

國立臺灣大學 **人類學系** 三年級。修讀語言學（系必修）、計算語言學、
資訊法與資訊政策專題。2025 年於**中央研究院資訊科學研究所 Depositar Lab**
（研究資料寄存與開放資料平台）暑期實習。第五、六屆**臺灣好厲駭**培訓學員
（教育部先進資通安全實務人才培育計畫，2020–2022）。

長期參與開源社群並多次公開發表：

| 年份 | 場合 | 講題 |
|---|---|---|
| 2026 | SITCON | 〈地圖與權力：公眾地理資訊系統應用〉 |
| 2025 | SITCON | 〈因為反抗，所以存在：監控資本主義下的自由如何可能？〉 |
| 2024 | SITCON | 〈科技藝術入門 — 淺談 p5.js〉 |
| 2024 | COSCUP | 〈資料視覺化 — 新手也能學會的 D3.js 實戰〉 |

技術背景：Web（HTML／CSS／JS）、Git、Linux、Python、GIS（QGIS／ArcGIS）。

### 在這個專案負責什麼

[`eval/`](eval/) — 整個實習最早的評測程式，也是後續所有實驗的起點。

- 建立完整鏈路：Ollama VLM 呼叫 → 後處理 → 四指標評分 → 互動式 HTML 報告
- 設計 [`postprocess.py`](eval/postprocess.py) 的雜訊清理規則：prompt 複誦、`<think>`
  推理標籤、HTML 標籤、JSON 包裝、code fence、重複字元迴圈
- 實作 EM／CM／ANLS／F1 四個指標（[`metrics.py`](eval/metrics.py)）
- 一頁式自包含 HTML 報告（[`build_report.py`](eval/build_report.py)）：模型總覽卡片、
  文字長度與圖片類別對 ANLS 的影響、可篩選／排序／搜尋的逐張圖明細
- 長時間無人值守執行腳本（[`run_until_done.sh`](eval/run_until_done.sh)），支援斷線重試

產出：1,312 行 Python／Shell，以及唯一一份完整保存下來的三模型評測報告
[`eval/results/report.html`](eval/results/report.html)。

### 他真正的貢獻

在 `eval/README.md` 裡有一節叫〈**為什麼跟別人的結果可能不一樣**〉。他在那裡做了三件事：

1. 指出同一個模型、同一份資料，不同人跑出的分數可以從 ANLS 0% 到 55%
2. 把六個可能原因**依影響程度排序**，並把「後處理清理程度」排在第一位
3. 提出可操作的診斷方法：比對雙方的 `prediction_raw`——如果原始輸出相同、只有清理後的
   `prediction` 不同，就是後處理問題；如果連原始輸出都不同，就往 prompt 與推論參數找

這不是工程問題，是**方法論的懷疑**：不問「分數是多少」，問「這個測量是怎麼被建構出來
的」。人類學與開放資料治理的訓練核心正是這個。

而三週後，用 3,706 筆資料 × 7 組變因、18,530 次推論做完的
[消融實驗](ablation_experiment/RESULTS.md)**證實了他的排序判斷是對的**——
後處理修改了 baseline 3,704／3,706 筆 raw response，換成最小後處理後 EM 直接歸零。

一個大三人類學系學生，用領域直覺押中了整個專案最重要的發現。這個問題後來成為整份研究
的主軸。

### 工作紀錄

| 日期 | 內容 |
|---|---|
| 2026-07-15 | 建立 TC-STR OCR 模型評估 pipeline |
| 2026-07-15 | 撰寫各檔案用途說明，並提出跨團隊分數分歧的成因分析 |

---

## 黃以信（HUANG, Yi-Hsin）

[@hyslchs](https://github.com/hyslchs)｜[sixhuang.com](https://sixhuang.com)｜社群暱稱：阿六

> repo 裡的 [`Sixhuang/`](Sixhuang/) 目錄名稱即來自他的社群 handle。

### 背景

輔仁大學 **圖書資訊學系** 三年級。學習方向：圖書資訊學、自然語言處理、數據分析、
機器學習。

**NSTC 大專學生研究計畫**（研究者：黃以信）——〈建置基於語意相似性的書籍推薦系統〉：
以輔仁大學圖書館約 30 萬筆中文書目為基礎，用 `text-embedding-3-large` 產生向量、
以 Chroma DB 做相似度檢索，處理館藏系統中「以分類號為主的相關書推薦」在跨主題探索上的
限制。目前持續延伸研究，預計投稿圖書資訊學相關期刊。

**AI CUP 2026 ESG 永續承諾驗證競賽**（進行中）——文字分類、資訊擷取與模型評估方法，
實作 TF-IDF、Transformer 多任務分類、BERT 與錯誤分析。

長期參與開源社群，以**編輯與文件**角色為主：

| 年份 | 角色 |
|---|---|
| 2026 | SITCON 編輯組員 & 行銷組員、**SITCON 2026 講者**、SITCON Camp 編輯組長、g0v summit 社群宣傳小組 |
| 2025 | SITCON 編輯組長、SITCON Camp 編輯組員 |
| 2024 | SITCON 編輯組、SITCON Camp 編輯組長 |

- SITCON 2026 講題：〈在書海中的航行指南；打開你對圖書館的全新視野！〉
  （[議程](https://sitcon.org/2026/agenda/dbd911/)）
- [SITCON 電子月刊](https://medium.com/sitcon.tw)撰稿

技術背景：Python、pandas、JSON／CSV 資料處理；Word Embedding、語意相似度、向量檢索、
TF-IDF、Transformer；向量資料庫、Markdown、Git／GitHub、研究紀錄整理；技術文件撰寫、
社群協作與跨角色溝通。

### 在這個專案負責什麼

四個目錄，涵蓋實習的第 2 到第 5 階段：

**[`Sixhuang/`](Sixhuang/)（2026-07-16）** — 與 `eval/` **各自獨立實作**的第二套 TC-STR
跑分程式，另納入 Hugging Face 繁中合成資料集。兩套流程的差異（短 prompt、最小後處理、
不同 scorer 定義）後來直接成為消融實驗的三個受測變因。906 行程式。

**[`ablation_experiment/`](ablation_experiment/)（2026-07-21）★ 本 repo 最重要的成果**
— 把兩套獨立流程對齊成共同 baseline，做 one-factor-at-a-time 單變因消融：
GLM-OCR Q8 × TC-STR 3,706 筆 × 7 組 variant，其中 5 組需推論，合計 **18,530 次 Ollama
request，全部完成、API error 數為 0**。923 行程式。

得到的結論包括：不指定 `repeat_penalty=1.6` 讓 EM 從 47.06% 提升到 64.84%（+17.78 pp，
淨增 659 題全對）；短 prompt 讓 EM／ANLS 歸零但 CM 反而升到 79.06%（字認出來了、
格式失控）；`/api/generate` 與 `/api/chat` 的 3,706 筆 prediction 逐字完全相同。

**[`benchmark_suite/`](benchmark_suite/)（2026-07-28）** — 可長時間無人值守、append-only
斷點續跑的評測系統。5 模型 × 3 benchmark（OmniDocBench／TC-STR／VisTW-MCQ）= 15 組正式
結果。4,010 行程式、12 個測試檔。

**[`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/)（2026-07-30 ～ 08-14）** — 規模最大的一套。
TC-STR 8 個模型；OmniDocBench v1.6 全量 1,651 頁，使用**釘死版本的官方 Docker
evaluator** 計分，另備 9 個模型設定與多模型互動比較報告工具。10,105 行程式、
12 個測試檔、950 行文件。

### 他真正的貢獻

兩套評測系統的核心其實不是跑分，是**來源追溯**：

- resume key／run signature 包含 config hash、實際 model digest、dataset revision、
  prompt hash 與 dependency lock SHA256——任何一項不同就是另一次實驗，不會混入舊結果
- [`VERSION_PINNING.md`](TC-STR_and_OminDoc/OminDocBench/VERSION_PINNING.md)：釘死資料
  revision 與 evaluator commit，並拒絕混用 Docker image 內建的舊版 GT
- [`PROTOCOL_EXCEPTIONS.md`](TC-STR_and_OminDoc/TC_STR/PROTOCOL_EXCEPTIONS.md)：例外要
  具名核准、載明日期與影響範圍，並註明哪些欄位因此不可等量比較
- [`MODEL_CANDIDATES.md`](TC-STR_and_OminDoc/TC_STR/MODEL_CANDIDATES.md)：第三方 GGUF
  量化版本的來源、發布者、宣稱上游、風險與核准狀態逐一存證——因為「`mradermacher` 發布的
  Kimi-VL 量化版」不等於「Moonshot AI 的 Kimi-VL」，這個區別直接影響結果能否歸因到原模型
- 官方指標與自製診斷指標嚴格分區，evaluator 失敗時 scoring 標成 `blocked`、絕不用近似值
  頂替

這在圖書資訊學裡有名字：**編目、權威控制、來源證明（provenance）**。他等於把圖資學的
紀律搬進 ML 評測，而那正是這個領域最缺的一塊。

多年 SITCON 編輯組長的訓練也直接體現在文件上——`TC-STR_and_OminDoc/` 一個目錄就有
950 行文件，涵蓋方法論、指標定義、版本釘選與例外紀錄。他甚至在最後主動回頭修正
README 的準確性（model-set 數量、InternVL 廠商歸屬、`ensure_official_configs` 實際行為），
並把混用中英文的說明統一成繁體中文。

### 工作紀錄

| 日期 | 內容 |
|---|---|
| 2026-07-16 | 建立獨立的 OCR benchmark 程式 |
| 2026-07-21 | GLM-OCR 消融實驗與報告；補完整消融結果文件 |
| 2026-07-28 | 建立可重現 VLM 評測系統；進度顯示、暫存空間配置、HF 下載重試、Python 3.14 相容、分開執行與樣本限制（共 7 次提交） |
| 2026-07-30 | TC-STR 與 OmniDocBench 執行工具；方法論文件 |
| 2026-08-06 | 修正硬編碼路徑；多模型比較報告工具與共用權限修正腳本；共用多人設定文件 |
| 2026-08-14 | 加入 Qwen3-VL／InternVL3.5 模型測試；修正 README 準確性；統一為繁體中文 |

---

## 林柏儒（LIN, Bo-Ru）

### 背景

國立臺灣大學（2021–2026），主修**地質學系**，輔修生物產業傳播暨發展學系及傳播學程。
2024 年於**國家災害防救科技中心**暑期實習，彙整 11 個國家的自然災害防治與救助方案
進行跨國比較。曾任臺大地質標本館導覽員。

技術背景偏 **GenAI 落地與自動化工作流**：ComfyUI 開發、n8n 跨系統自動化（整合 GCP、
Google Workspace 及多個社群平台 API）、**開源多模態模型的本地端部署**（曾為教育機構
建立滿足隱私要求的會議轉錄與摘要方案）。另有 GenAI 自由接案與 AI 工具教學經驗。

### 在這個專案負責什麼

**計畫前期的硬體評估與詢價**——成果整理在 [HARDWARE.md](HARDWARE.md)。

- 各模型 × 訓練方法的**顯存需求試算**（QLoRA 4-bit／LoRA 16-bit／全參數微調），
  以及每參數記憶體成本的拆解——這張表說明了為什麼全參數微調的記憶體成本是 QLoRA 的
  約 29 倍
- **七套工作站配置的比較與實際廠商詢價**：1／2／3／4 卡，AMD 與 Intel 平台，
  從 NT$65 萬的單卡家用桌機到 NT$277 萬的四卡工作站
- 連帶的規格判斷：VRAM:RAM 比例、電源與 220V 供電需求、機箱尺寸與散熱形式

### 這份評估的價值在哪裡

**OCF 最後沒有依這份規格採購**，改為委請廠商規劃整機。

但評估本身並沒有白做——它提供的是**議價與判斷的基準**。知道 GPU 佔總價六到七成、
知道 2 卡與 4 卡的實際差距、知道電源必須 220V 且會連動到機房配電，才有辦法判斷廠商
開出來的整機報價合不合理、規格有沒有被偷換。

他的本地部署背景也直接對應這件事：**評估「要買什麼卡」的人，最好是實際在本地跑過
開源模型的人**——他知道顯存不夠時會發生什麼事。

這條線同時連到計畫的另一半：本 repo 的評測全部跑在 AWS 雲端單卡上，正是因為本地機
尚未到位。[HARDWARE.md](HARDWARE.md) 裡「先租，不要先買」那一節，是這次實際走過一遍
之後的結論。

---

## 工作量

以下數字全部可由 repo 直接驗證。

| 項目 | 數量 |
|---|---|
| 實習期間 | 2026-07-15 ～ 08-19（約 5 週） |
| 程式碼 | 15,909 行 Python（90 檔）＋ 1,347 行 Shell（34 檔） |
| 文件 | 2,196 行 Markdown（14 份） |
| 測試 | 29 個測試檔 |
| 設定檔 | 19 份 YAML／JSON |
| 完成推論 | 18,530 次（消融實驗），全部成功、0 API error |
| 評測資料規模 | TC-STR 3,706 張 × 7 組變因；OmniDocBench 全量 1,651 頁 |
| 涵蓋開源 VLM | 16 個（Gemma 4 五種尺寸、Qwen2.5-VL／Qwen3-VL、GLM 三種、InternVL3.5 兩種、Kimi-VL、SmolVLM2、SEA-LION、chandra-ocr-2） |

程式碼行數只反映投入，不反映價值——這份研究最重要的成果是一個
[方法論的發現](README.md#核心發現)，它的證據是 18,530 次推論，而不是 15,909 行程式。

---

## 分支與合併紀錄

兩人的工作原本分別在 `main` 與 `Sixhuang` 分支上進行，於 2026-08-19 透過
[PR #1](https://github.com/ocftw/2026-OCF-Intern/pull/1) 合併。**`main` 目前包含全部研究
成果**；`Sixhuang` 分支僅作為歷史紀錄保留，內容與 `main` 完全一致
（`git diff main origin/Sixhuang` 為空）。

---

*本頁僅收錄兩位實習生履歷中原本即為公開的資訊（就讀學校科系、公開演講與議程連結、
公開研究計畫、GitHub 帳號、個人網頁）。聯絡方式未列於此，如需聯繫請透過 OCF 或其
GitHub 個人頁面。*
