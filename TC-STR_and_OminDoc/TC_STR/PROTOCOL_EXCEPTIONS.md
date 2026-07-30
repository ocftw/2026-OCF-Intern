# TC-STR protocol exceptions

## GLM OCR BF16：Ollama completion metadata

- 核准日期：2026-07-29
- 核准者：使用者
- Logical model id：`glm_ocr_bf16`
- Exact Ollama tag：`glm-ocr:bf16`
- 模型大小：Ollama 約 1.1B（GLM-OCR 官方介紹約 0.9B／1B 級）
- 影響範圍：只限 Ollama `/api/generate` completion envelope 的成功判定

### 觀察到的行為

在 `stream=false`、top-level `think=false` 與本 Benchmark 的共同 generation
options 下，模型可回傳 HTTP 200 與非空 OCR `response`，但回應可能是：

```json
{
  "response": "雜貨舗",
  "done": false
}
```

亦即缺少其他模型正常提供的：

- `done_reason`
- `prompt_eval_count`
- `eval_count`

BF16 與另行診斷的 Q8_0 版本都呈現相同行為，因此這被視為目前
GLM-OCR × Ollama completion／thinking 整合差異，而不是量化版本的能力差異。

### 已核准的接受規則

只有 `glm_ocr_bf16` 使用
`allow_nonempty_response_without_terminal_metadata`：

1. HTTP request 必須成功，且回應不得包含 API error。
2. 正式 `response` 必須在 outer trim 後非空。
3. 即使 `done` 不是 `true`，仍可保存該筆 prediction。
4. `done_reason`、`prompt_eval_count`、`eval_count` 缺失時一律保存為 `null`，
   不推估、不補值。
5. 每筆資料保存 `completion_validation`，明確標示是否由例外接受。
6. 空字串、純空白、API error、HTTP error 與 timeout 仍依共同規則重試；重試耗盡後
   保存為 terminal failure 並納入分數。

### 沒有放寬的項目

- 完整共同 OCR prompt
- 圖片原始 bytes 與 batch size 1
- `num_ctx=65536`、`num_predict=80`、temperature、repeat penalty、top-k、top-p
- top-level `think=false`
- 100% GPU 要求
- primary normalization、aligned supplementary 與四項 scorer
- smoke 的人工逐筆檢視、異常標記與完整 raw response 保存

### 對分析的影響

- OCR prediction 與 EM／CM／ANLS／Character F1 仍按相同資料、prompt、參數及
  scorer 計算，所以可納入主要「辨識分數」比較。
- `prompt_eval_count`、`eval_count` 與由這些欄位推導的 token 效率不能和其他模型
  比較。
- 沒有 `done_reason`／`eval_count` 時，無法自動證明輸出未碰到 `num_predict`。
  報告必須顯示 `truncation_unknown=true`，不可把它解讀成
  `truncated=false` 的正面證據。
- raw response、HTTP status、latency、例外接受狀態與缺失欄位仍逐筆保留，方便日後
  Ollama 修正後重跑或做敏感度分析。

這項 policy 會納入 run signature。移除、修改或套用到其他模型都會產生不同 signature，
避免與舊 checkpoint 混用。

## 全模型輸出失敗處理政策

- 核准日期：2026-07-29
- 核准者：使用者
- Policy version：`common_conditions_score_failures_v1`
- 適用範圍：全部 8 個 logical models

為了回答「市面上不同模型在同一組使用條件下的 Benchmark 分數」，所有模型繼續使用
相同 prompt、原圖、endpoint 與 generation options。重複、prompt echo、拒答、格式
包裝、thinking、無關輸出、空輸出與 token-limit 都是可觀察的模型結果，不再因達到比例
門檻而阻擋其他模型或整套正式評測。

重試耗盡後，每題立即寫入 EBS SQLite：

- 有 HTTP 200 且非空但帶 anomaly：`completed_with_anomaly`，依原輸出正常計分。
- HTTP 200 但空白／control-only：`model_output_failure`，空 prediction 計分。
- 沒有可接受 HTTP 200：`request_failure`，空 prediction 計分並保存最後錯誤。

仍屬硬阻擋的是資料／signature 不一致、exact tag 或 vision 不符、模型完全無法產生
HTTP 200 response、或 `ollama ps` 不是精確 `100% GPU`。這項全模型政策也納入 run
signature，避免與舊 checkpoint 混用。
