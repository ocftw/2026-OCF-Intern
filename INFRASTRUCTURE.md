# 執行環境與主機配置

本研究的所有評測都在 OCF 於 AWS 開設的一台 GPU 主機上執行。這份文件記錄該主機的規格、
儲存架構與服務設定。

它有兩個用途：

1. **重現實驗的前提。** 評測數字只有連同執行環境才有意義——這正是
   [本研究最主要的結論](README.md#核心發現)。
2. **可重用的操作經驗。** 「在 EC2 上開一台跑 local LLM 評測的機器」有幾個容易踩的坑
   （instance store 會消失、Ollama 不會自己用你要的 context 長度、模型權重放錯磁碟每次
   開機要重抓幾十 GB），這裡把當時的決策與實作都寫下來。

> 內容整理自實習期間的內部維運文件（最後更新 2026-07-29），已移除聯絡方式與內部溝通
> 用語。機器本身為實習期間的臨時資源，文件保留作為方法紀錄。

---

## 一、機器規格

| 項目 | 規格 |
|---|---|
| Instance | AWS EC2 **`g6e.2xlarge`** |
| GPU | 1× **NVIDIA L40S**，46,068 MiB VRAM（約 45 GB） |
| NVIDIA driver | 610.43.02 |
| vCPU | 8（AMD EPYC 7R13；4 cores、2 threads/core） |
| RAM | 64 GB 標示／61 GiB 可用，無 swap |
| Instance store | 1× 450 GB NVMe（約 419 GB 可用） |
| Root volume | EBS（持久） |
| OS | Ubuntu 26.04 LTS（Resolute Raccoon），kernel `7.0.0-1009-aws` |
| Python | 3.14.4 |
| Ollama | 0.31.1 |
| Docker Engine | 29.6.2 |

### 為什麼選 L40S 48GB

評測是純推論、不是訓練，顯存需求比微調小很多。當時的尺寸估算（bf16 權重約為「參數量 ×2 GB」，
再加 KV cache 開銷）：

| 模型尺寸 | bf16 權重 | 單卡最低顯存 |
|---|---|---|
| E4B（約 4.5B） | 約 9 GB | 24 GB 就夠 |
| 12B | 約 24 GB | 48 GB 舒適（24 GB 需 4-bit） |
| 26B MoE（A4B，每 token 啟用約 3.8B） | 約 52 GB | 80 GB，或 48 GB 跑 4-bit／fp8 |
| 31B Dense | 約 62 GB | 80 GB，或 48 GB 跑 4-bit |

> MoE 的全部參數都要進顯存，不會因為只啟用一部分而變小。

結論是**一張 L40S 48 GB 就能涵蓋小尺寸的全精度評測與大尺寸的量化評測，中途不必換機重裝
環境**。`g6e.2xlarge` 相對 `g6e.xlarge` 多的是 CPU RAM，載大模型時較順。當時 us-east-1
的 on-demand 參考價約 $1.9–2.2/hr（實際以 AWS 定價為準）。

---

## 二、儲存架構：最重要的一節

機器上有**兩塊性質完全不同**的儲存。放錯地方會讓跑了一週的結果直接消失。

### `/mnt/nvme/` — 本機 NVMe instance store，約 419 GB，快，**但會消失**

Instance store 實體上綁在當下那台實體主機。

```
reboot（重開機）        -> 資料還在
stop / start（關再開）  -> 整顆清空，開機時自動重新格式化
terminate（終止）       -> 沒了
```

| 目錄 | 放什麼 |
|---|---|
| `/mnt/nvme/scratch/` | 中間檔、rasterize 出來的圖、任何暫存 |
| `/mnt/nvme/datasets/` | OmniDocBench 等資料集本體 |
| `/mnt/nvme/outputs/` | 跑到一半的逐頁輸出 |
| `/mnt/nvme/containerd/`、`/mnt/nvme/docker/` | container image 儲存，**請勿手動更動** |

### `/opt/ocf-ai/` — EBS，**會保留**

reboot 與 stop/start 都不受影響。

| 目錄 | 放什麼 |
|---|---|
| `/opt/ocf-ai/outputs/` | **最終成果放這裡** |
| `/opt/ocf-ai/models/` | Ollama 模型權重（由服務管理） |

### 判斷標準

> **資料集重抓得回來，跑一週的結果重跑不回來。**

跑分過程中的逐頁輸出可以先寫 `/mnt/nvme/outputs/` 求速度，但**一個 run 結束就要把成果搬到
EBS**：

```bash
cp -a /mnt/nvme/outputs/<run_name> /opt/ocf-ai/outputs/
```

機器上 `/mnt/nvme/README.txt` 存有同一份摘要，忘記時可直接在機器上查。

### 這個架構如何反映在本 repo 的程式裡

這不只是維運細節，它直接寫進了評測程式的設計：

- [`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/) 的 runner 把 SQLite checkpoint、raw
  response、prediction、metadata 與報告**直接寫入 EBS**，`/mnt/nvme` 只放可重新下載或
  重建的資料。其 README 的〈儲存空間的角色〉表格就是這張表。
- [`benchmark_suite/`](benchmark_suite/) 在另一種主機配置下處理同樣的問題：當
  ephemeral volume 掛在 `/srv` 時，用
  [`scripts/configure_srv_scratch.sh`](benchmark_suite/scripts/configure_srv_scratch.sh)
  把 Ollama models、Docker data-root、datasets 與 cache 移過去，但**正式 prediction／
  result 一律留在持久磁碟**，`runs/` 不放 ephemeral 空間。
- 兩者都設計成 instance store 被清空後仍能依相同 resume identity 續跑——模型與資料重抓，
  結果不受影響。

---

## 三、instance store 的自動掛載

Instance store 在 stop/start 後會變成一顆**全新未格式化**的裝置，所以不能用一般的
`/etc/fstab` 掛載流程。當時的作法是一支開機腳本加一個 systemd unit。

### 三個關鍵限制

- **不要把 instance store 寫進 `/etc/fstab`。** stop/start 之後會換一台實體機，裝置是
  全新空白的，fstab 裡的 UUID 掛不起來，開機會失敗。
- **不要 hardcode `/dev/nvme1n1`。** Nitro 上的裝置編號不保證固定，要用 MODEL 字串動態
  偵測（instance store 是 `Amazon EC2 NVMe Instance Storage`，EBS 是
  `Amazon Elastic Block Store`）。
- **不要把 `OLLAMA_MODELS` 指到 instance store。** 模型權重只在載入時讀一次，放在上面
  唯一的結果是每次 stop/start 都要重抓數十 GB。**模型放 EBS。**

### 掛載腳本

`/usr/local/sbin/mount-instance-store.sh`（需 `chmod +x`）：

```bash
#!/bin/bash
set -euo pipefail

MNT=/mnt/nvme
GRP=ocfai

DEV=$(lsblk -dno NAME,MODEL | awk '/Amazon EC2 NVMe Instance Storage/{print "/dev/"$1; exit}')
if [ -z "${DEV:-}" ]; then
  echo "no instance store device found" >&2
  exit 0
fi

# 沒有檔案系統就代表是 stop/start 之後的全新裝置，格式化
if ! blkid "$DEV" >/dev/null 2>&1; then
  mkfs.xfs -f -L nvmescratch "$DEV"
fi

mkdir -p "$MNT"
mountpoint -q "$MNT" || mount -o discard,noatime "$DEV" "$MNT"

getent group "$GRP" >/dev/null || groupadd "$GRP"

# setgid 讓新建檔案自動繼承群組，同組成員互相讀得到
for d in scratch outputs datasets; do
  mkdir -p "$MNT/$d"
  chgrp "$GRP" "$MNT/$d"
  chmod 2775 "$MNT/$d"
done
```

需要 `xfsprogs`（Ubuntu：`apt-get install -y xfsprogs`）。

### systemd unit

`/etc/systemd/system/instance-store.service`：

```ini
[Unit]
Description=Format and mount EC2 instance store NVMe
After=local-fs.target
Before=ollama.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/mount-instance-store.sh

[Install]
WantedBy=multi-user.target
```

`Before=ollama.service` 這行必須保留，確保 Ollama 啟動時目錄已經存在。

```bash
systemctl daemon-reload && systemctl enable --now instance-store.service
```

### 驗收

```bash
df -h /mnt/nvme /opt/ocf-ai
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,MODEL
systemctl status instance-store.service --no-pager
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

`df -h /mnt/nvme` 應顯示約 419 GB（450 GB 是十進位標示容量）。建議實際 `reboot` 一次
驗證自動掛載真的生效。

---

## 四、Ollama

以 systemd 服務常駐執行，模型權重放在 EBS（`OLLAMA_MODELS=/opt/ocf-ai/models`），不會
因 stop/start 消失。

```bash
ollama list      # 看有哪些模型
ollama ps        # 看目前載入了什麼、跑在 GPU 還是 CPU
ollama pull <tag>
```

### `ollama ps` 的 PROCESSOR 欄位必須是 100% GPU

跑任何正式評測前先確認。如果出現 CPU 的比例，代表模型加上 KV cache 超出 45 GB VRAM、
被切一部分到 CPU 上跑——**速度會掉一個數量級，而且跑出來的數字不能用**。

這條規則後來被寫進評測程式本身：[`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/) 的 runner
會在正式推論前檢查 `ollama ps` 是否精確顯示 `100% GPU`，不符就擋下評測——這被歸類為
「非模型因素的工程錯誤」，與「模型自己造成的失敗」（空輸出、拒答、重複、格式錯誤）
分開處理，後者照樣計分。

### Context 長度要自己指定

專案目標是 65K context。**Ollama 不會自己用 65K**——預設值會依 VRAM 自動選
（4k／32k／256k），跟 65K 對不上，所以每次呼叫都要明確指定 `options.num_ctx`：

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma4:12b-it-qat",
  "prompt": "...",
  "options": { "num_ctx": 65536 }
}'
```

### KV cache 維持 fp16

當時刻意不開量化（未設 `OLLAMA_KV_CACHE_TYPE`），因為第一輪 baseline 要用未量化的設定跑，
數字才有可比性。若 65K 下 VRAM 不足，`q8_0` 大約可省一半 KV cache VRAM——但這屬於會改變
結果可比性的變更，應視為另一組實驗。

本 repo 的 runner 只**記錄** KV cache 設定，不修改服務設定。

---

## 五、Docker 與 OmniDocBench 官方評分器

OmniDocBench 的官方評分需要 TeX Live、ImageMagick 與 Ghostscript，因此固定在官方建議的
Docker image 內執行：

```
ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204
```

已驗證的 image 內 runtime：Python 3.10.16、TeX Live 2025（pdfTeX 3.141592653-2.6-1.40.28）、
ImageMagick 7.1.1-47、Ghostscript 9.55.0。

這個 image 的 entrypoint 預設會直接跑評測：

```
Entrypoint:  python pdf_validation.py
Cmd:         --config configs/reproduce_chutao_v16_repro_conda_20260408.yaml
WorkingDir:  /workspace
```

```bash
# 換 config
docker run --rm -v /mnt/nvme:/mnt/nvme <image> --config <你的 config>

# 進去看環境、不跑評測（要覆蓋 entrypoint）
docker run --rm -it --entrypoint bash <image>
```

> ⚠️ image 內建的 1,355 頁 GT **不是** 本研究使用的 v1.6 full set（1,651 頁），runner 會
> 拒絕混用。詳見
> [`VERSION_PINNING.md`](TC-STR_and_OminDoc/OminDocBench/VERSION_PINNING.md)。

Image 約 32 GB，存在 instance store 上，**stop/start 後需要重抓**。

---

## 六、多人共用

共用目錄屬於 `ocfai` 群組並開了 setgid（`chmod 2775`），在裡面建立的檔案會自動繼承群組，
同組成員互相讀得到，不需要自己 `chmod`。

新加入群組後**必須登出再重新登入才會生效**：

```bash
id
# groups 裡要看得到 ocfai

# 確認能寫入
touch /mnt/nvme/scratch/test && ls -l /mnt/nvme/scratch/test && rm /mnt/nvme/scratch/test
# 檔案的群組應顯示 ocfai
```

Docker 也採同樣作法（使用者加入 `docker` 群組，**不需要 sudo**）。本 repo 的正式 runner
一律不會自行 `sudo`；若遇到 Docker permission denied，是重新登入或 `newgrp docker` 的問題。

`TC-STR_and_OminDoc/OminDocBench/` 另有一套讓其他人在**不需要擁有者權限**的情況下讀取
評測結果、產生多模型比較報告的機制，見該目錄 README 的〈生成多模型比較報告〉與
`tools/fix_shared_report_permissions.py`。

---

## 七、stop / start 之後的復原步驟

機器**關機再開機**（不是 reboot）之後，`/mnt/nvme` 會是全新空的。systemd unit 會自動
重新格式化並建好目錄，但**內容要自己補回來**：

```bash
# 1. 重抓 container image（約 32 GB，要等幾分鐘）
docker pull ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204
docker images

# 2. 把資料集重新放回 /mnt/nvme/datasets/

# 3. 模型不用重抓（在 EBS 上），確認還在就好
ollama list
```

本 repo 的兩套 runner 都能在這之後依相同 resume identity 從 checkpoint 續跑。

---

## 八、幾件不要做的事

- **不要把要保留的東西只放在 `/mnt/nvme/`。** 這是最容易犯、而且犯了沒得救的錯。
- **不要改 `/etc/fstab`、Ollama 服務設定、Docker `daemon.json`。** 這幾個檔案的配置是為了
  stop/start 之後還能正常開機，改動很容易讓整台機器起不來（實習期間曾因此掛過一次）。
- **不要手動動 `/mnt/nvme/containerd/` 與 `/mnt/nvme/docker/`。** 那是 Docker 的內部儲存。
- **不要在硬碟吃緊時自己刪系統檔案。**
- **不要在評測跑到一半改參數。** 本 repo 的 runner 不會自動降 `num_ctx`／`num_predict`
  或換量化來遷就 OOM；正確作法是停下來、改用 constrained profile，那會形成**另一次**
  實驗。

遇到儲存、GPU 或服務起不來的狀況，先回報再動手。

---

## 九、當時已知的限制

以下是實習期間這台機器的狀態，記錄下來以免誤讀結果：

- **成果備份只靠 EBS（`/opt/ocf-ai/outputs/`），沒有 S3 同步。** 也沒有自動把
  `/mnt/nvme/outputs/` 定時搬到 EBS 的排程，需要手動搬。
- **model tag 曾有落差。** 規劃文件寫的 `gemma4:31b` 在 Ollama registry 上找不到對應，
  機器上實際是 `gemma4:{e2b,e4b,12b,26b-a4b,31b}-it-qat` 這組 QAT 版本。這正是本 repo
  後來堅持記錄 **exact tag + digest** 而非只記型號的原因——見
  [`TC_STR/models.yaml`](TC-STR_and_OminDoc/TC_STR/models.yaml) 與
  [`MODEL_CANDIDATES.md`](TC-STR_and_OminDoc/TC_STR/MODEL_CANDIDATES.md)。
- 本機工作站到貨前，AWS 機器是暫時方案；`benchmark_suite/` 的 `/srv` scratch 配置對應的
  是另一種主機環境。

---

## 相關文件

- [根目錄 README](README.md) — 研究成果與核心發現
- [`TC-STR_and_OminDoc/README.md`](TC-STR_and_OminDoc/README.md) — 完整的系統環境紀錄與
  可比性原則
- [`benchmark_suite/README.md`](benchmark_suite/README.md) — 另一種主機配置下的
  scratch／持久儲存分離
