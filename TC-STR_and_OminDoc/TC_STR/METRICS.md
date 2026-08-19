# TC-STR metrics and normalization

這四項是本研究為八個模型採用的統一診斷指標，不是 TC-STR 官方唯一指定的
leaderboard protocol。主榜與 aligned supplementary 永遠分開顯示。

## Primary prediction

`prediction_raw` 永久保存，不覆寫。`prediction_primary` 對所有模型只執行：

1. `null` 轉為空字串。
2. CRLF／CR 統一為 LF。
3. Unicode 正規化為 NFC。
4. 移除整個字串最外層的 whitespace。
5. API 的獨立 thinking 欄位不會被併入 `response`；若 thinking 混入正文，不猜測答案。

不忽略標點或內部空白，不做雙向 contain、重複字壓縮、長度截斷、解釋抽取或
模型專用處理。

所有 terminal sample 都進入模型的固定分母。HTTP／API／timeout 重試耗盡或模型回傳
空白時，`prediction_primary=""`，因此有非空 ground truth 的 EM、CM、ANLS、F1 都為
0。重複、prompt echo、拒答、格式或 token-limit 等非空異常則直接以未猜測、未截斷的
`prediction_primary` 評分。這些 anomaly flags 只協助事後分析，不從分母排除樣本。

四個主指標：

- EM：trim 後 prediction 與 ground truth 完全一致。
- CM：單向判斷完整 ground truth 是否出現在 prediction (`gt in pred`)。
- ANLS：`1 - Levenshtein / max(len(pred), len(gt))`，低於 0.5 計 0。
- Character F1：Unicode 字元 multiset overlap 的 precision／recall harmonic mean。

## Aligned supplementary

為重現 repo 內參考消融實驗，`prediction_aligned_supplementary` 依序執行：

1. 若有閉合 `</think>`，移除其前方 thinking。
2. 偵測共同 OCR prompt 的固定片語，從 prompt echo 起截斷。
3. 若完整輸出為 JSON array，抽取字串元素或 `text/content/value/caption` 欄位。
4. 移除最外層 Markdown code fence。
5. 移除 HTML tags。
6. 移除 `文字/文本/辨識結果/OCR結果/answer/text/ocr:` 前綴。
7. 移除成對的最外層引號。
8. 移除 CR、LF、tab（保留普通空格）。
9. 將連續四次以上的同一字元壓縮為一次。
10. 截斷至 200 Unicode 字元。

每筆結果的 `aligned_changes` 會列出實際觸發規則。第 9、10 項會在 HTML 顯示警示；
補充分數不得覆蓋或被稱為主榜分數。
