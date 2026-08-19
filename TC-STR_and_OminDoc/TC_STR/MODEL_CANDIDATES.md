# 非官方模型候選與核准紀錄

最後查核：2026-07-29。使用者已明確核准下載及驗證候選 A 與 Kimi Q4_K_M；
兩者已下載並以 exact tag 寫入 `models.yaml`。本文件保留來源、風險與選擇紀錄。

## GLM-4.6V-Flash 9B

官方上游權重存在於 `zai-org/GLM-4.6V-Flash`，但 Ollama 官方 `library/` 沒有對應
模型。可見的 Ollama tags 都是 community conversion。

候選 A：

- exact tag：`haervwe/GLM-4.6V-Flash-9B:latest`
- 狀態：**已核准、已下載、已映射**
- 本機 digest：`ad2e2e374c6bcda3739e6563a5d0edf231d3deb0e5beeae27e89d6e5c16f8715`
- 本機 metadata：9.4B、Q4_K_M、131072 context、vision
- 發布者：Ollama community user `haervwe`，不是 Z.ai
- 宣稱來源：Unsloth Q4_K_M
- 宣稱能力：vision、tools、thinking
- 風險：自訂 template 與 sampling defaults；需要實際 `/api/show`、image request、
  `think:false`、共同 options 與 65K `100% GPU` 驗證
- 頁面：https://ollama.com/haervwe/GLM-4.6V-Flash-9B/tags

候選 B：

- exact tag：`sparksammy/glm-4.6v-flash-unsloth:latest`
- 發布者：Ollama community user `sparksammy`
- 大小：約 8.0 GB，128K，頁面標示 Text/Image
- 風險：第三方 Unsloth conversion；量化與 template 必須下載後再驗證
- 頁面：https://ollama.com/sparksammy/glm-4.6v-flash-unsloth/tags

不建議：

- `MedAIBase/GLM-4.6V-Flash:9b` 約 19 GB、F16，但 Ollama 頁面把 input 標成 Text，
  details family 為 `glm4`，無法在下載前證明 projector/vision capability。
- 其他 abliterated、uncensored、Q8 或不明匯入版本不符合控制變因要求。

## Kimi-VL-A3B-Instruct Q4

官方上游為 `moonshotai/Kimi-VL-A3B-Instruct`，但官方 repo 是約 32.8 GB 的原始權重，
沒有 Moonshot 官方 Q4 GGUF/Ollama tag。

唯一明確對應需求的候選：

- exact tag：`hf.co/mradermacher/Kimi-VL-A3B-Instruct-GGUF:Q4_K_M`
- 狀態：**已核准、已下載、已映射**
- 本機 digest：`252db10777f74ff9a13f5fe24efdea51aedc546eebac8ea14becec99c9b70698`
- 本機 metadata：Moonlight 16B total / A3B、Q4_K_M、131072 context、vision
- 發布者：Hugging Face community user `mradermacher`，不是 Moonshot AI
- 上游：宣稱直接量化自 `moonshotai/Kimi-VL-A3B-Instruct`
- 量化：static Q4_K_M
- 風險：第三方 quant；必須驗證 Ollama 是否同時取得 vision projector、template、
  image request、`think:false` 與 65K `100% GPU`
- 頁面：https://huggingface.co/mradermacher/Kimi-VL-A3B-Instruct-GGUF

不使用 Thinking-2506、abliterated 或其他模型變體代替 Instruct。
