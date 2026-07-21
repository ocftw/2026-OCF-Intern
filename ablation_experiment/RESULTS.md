# 消融實驗結果摘要

## 結論

- 3,706 筆資料全部完成，所有推論 variant 都是 0 errors。
- `/api/generate` 與 `/api/chat` 的 3,706 筆 processed prediction 完全相同，endpoint 不是這次分差來源。
- 短 prompt 會讓模型在答案後輸出解釋、Markdown、英文分析與 prompt 複誦；EM／ANLS 歸零不代表完全沒有辨識出文字，而是輸出格式失控且現有後處理無法完整清除。
- 不指定 `repeat_penalty=1.6` 是本次提升最大的單一變因：EM 增加 17.78 個百分點，ANLS 增加 17.53 個百分點。
- `num_predict=25` 將平均延遲縮短約 51%，但 EM 與 ANLS 小幅下降，而且被截斷的 prompt 複誦可能躲過後處理規則。
- aligned 後處理修改了 baseline 的 3,704／3,706 筆 raw response，是目前評測流程能取得合理 EM／ANLS 的關鍵。
- Sixhuang scorer 讓同一批 prediction 得到稍高分；這是 normalization／CM 定義不同造成的報告差異，不是模型能力提升。

## 總覽

| 實驗 | EM | CM | ANLS | F1 | 平均延遲 |
|---|---:|---:|---:|---:|---:|
| Aligned baseline | 47.06% | 76.52% | 54.54% | 59.17% | 0.602 s |
| Endpoint: `/api/chat` | 47.06% | 76.52% | 54.54% | 59.17% | 0.582 s |
| Sixhuang short prompt | 0.00% | 79.06% | 0.00% | 13.56% | 0.594 s |
| 不指定 `repeat_penalty=1.6` | **64.84%** | **82.46%** | **72.08%** | **76.62%** | 0.583 s |
| `num_predict=25` | 46.03% | 76.34% | 53.29% | 65.07% | **0.293 s** |
| Sixhuang minimal postprocess | 0.00% | 76.79% | 0.00% | 6.88% | 離線重算 |
| Sixhuang scorer | 47.87% | 77.63% | 55.02% | 59.96% | 離線重算 |

## 關鍵逐題差異

- 移除顯式 `repeat_penalty=1.6` 後，EM 有 779 題由錯轉對、120 題由對轉錯，淨增加 659 題完全正確。
- `num_predict=25` 後，EM 有 5 題由錯轉對、43 題由對轉錯。
- 短 prompt 只有 2／3,706 筆 prediction 與 baseline 完全相同；1,744 題由 EM 正確轉為錯誤。
- Sixhuang scorer 沒有改變任何 prediction，但額外判定 30 題 EM 正確。

## 建議

目前已測設定中，分數最佳的是 aligned 長 prompt、`num_predict=80`、不顯式傳入 `repeat_penalty=1.6`，搭配 aligned 後處理與統一 scorer。下一步可針對 repeat penalty 的預設值、1.0、1.1、1.3、1.6 做更細的 sweep，並另外測試 prompt 與後處理的交互作用。

完整設定、delta 與代表案例請看 [`results/ablation_report.html`](results/ablation_report.html)。
