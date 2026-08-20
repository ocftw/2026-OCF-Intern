# 執行環境與主機配置

本研究的評測作業全部執行於 OCF 在 AWS 開設的一台 GPU 主機。本文記錄該主機的規格、儲存
架構與服務設定。

文件有兩項用途。其一為重現實驗的前提，蓋評測數字唯有連同執行環境方具意義，亦即本研究
最主要的結論。其二為可重用的操作經驗：在 EC2 上開設一台執行本地 LLM 評測的主機，有若干
容易疏漏之處，包括 instance store 會於 stop/start 之後清空、Ollama 不會自行採用所需的
context 長度、模型權重存放位置錯誤將導致每次開機重新下載數十 GB，以下一併記錄當時的
決策與實作。

> 內容整理自實習期間的內部維運文件（最後更新 2026-07-29），已移除聯絡方式與內部溝通
> 用語。主機本身為實習期間的臨時資源，文件保留作為方法紀錄。

---

## 一、機器規格

| 項目 | 規格 |
|---|---|
| Instance | AWS EC2 **`g6e.2xlarge`** |
| GPU | 1× **NVIDIA L40S**，46,068 MiB VRAM（約 45 GB） |
| NVIDIA driver | 610.43.02 |
| vCPU | 8（AMD EPYC 7R13；4 cores、2 threads/core） |
| RAM | 標示 64 GB，可用 61 GiB，無 swap |
| Instance store | 1× 450 GB NVMe（可用約 419 GB） |
| Root volume | EBS（持久） |
| OS | Ubuntu 26.04 LTS（Resolute Raccoon），kernel `7.0.0-1009-aws` |
| Python | 3.14.4 |
| Ollama | 0.31.1 |
| Docker Engine | 29.6.2 |

### 機型選擇的依據

評測屬純推論作業，並非訓練，顯存需求遠低於微調。當時的尺寸估算以 bf16 權重約為「參數量
× 2 GB」再加 KV cache 開銷為基礎：

| 模型尺寸 | bf16 權重 | 單卡最低顯存 |
|---|---|---|
| E4B（約 4.5B） | 約 9 GB | 24 GB 即可 |
| 12B | 約 24 GB | 48 GB 為宜；24 GB 須採 4-bit |
| 26B MoE（A4B，每 token 啟用約 3.8B） | 約 52 GB | 80 GB，或 48 GB 搭配 4-bit 與 fp8 |
| 31B Dense | 約 62 GB | 80 GB，或 48 GB 搭配 4-bit |

MoE 的全部參數均須進入顯存，不因僅啟用一部分而減少。

據此，單張 L40S 48 GB 已足以涵蓋小尺寸的全精度評測與大尺寸的量化評測，中途無須更換
機型並重建環境。`g6e.2xlarge` 相對 `g6e.xlarge` 增加的是 CPU RAM，載入大模型時較為順暢。
當時 us-east-1 的 on-demand 參考價約每小時 1.9 至 2.2 美元，實際以 AWS 定價為準。

---

## 二、儲存架構

主機上有兩塊性質截然不同的儲存，存放位置錯誤將導致長時間執行的結果全數消失。

### `/mnt/nvme/`：本機 NVMe instance store，約 419 GB

Instance store 在實體上綁定於當下的實體主機，其生命週期如下：

```
reboot（重開機）        資料保留
stop / start（關再開）  整顆清空，開機時重新格式化
terminate（終止）       資料消滅
```

| 目錄 | 用途 |
|---|---|
| `/mnt/nvme/scratch/` | 中間檔、rasterize 產生的圖片與各類暫存 |
| `/mnt/nvme/datasets/` | OmniDocBench 等資料集本體 |
| `/mnt/nvme/outputs/` | 執行中的逐頁輸出 |
| `/mnt/nvme/containerd/`、`/mnt/nvme/docker/` | container image 儲存，不得手動更動 |

### `/opt/ocf-ai/`：EBS，資料持久保留

reboot 與 stop/start 均不受影響。

| 目錄 | 用途 |
|---|---|
| `/opt/ocf-ai/outputs/` | 最終成果 |
| `/opt/ocf-ai/models/` | Ollama 模型權重，由服務管理 |

### 判斷準則

資料集可重新下載，執行一週所得的結果則無法重現。

跑分過程中的逐頁輸出得先寫入 `/mnt/nvme/outputs/` 以取得速度，惟每一次 run 結束即應將
成果搬移至 EBS：

```bash
cp -a /mnt/nvme/outputs/<run_name> /opt/ocf-ai/outputs/
```

主機上的 `/mnt/nvme/README.txt` 存有同一份摘要，可隨時於機器上查閱。

### 儲存架構在程式設計上的反映

上述配置並非單純的維運細節，而是直接寫入了評測程式的設計：

- [`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/) 的 runner 將 SQLite checkpoint、raw
  response、prediction、metadata 與報告直接寫入 EBS，`/mnt/nvme` 僅存放可重新下載或
  重建的資料。該目錄 README 的〈儲存空間的角色〉表格即源於此。
- [`benchmark_suite/`](benchmark_suite/) 在另一種主機配置下處理相同問題：當 ephemeral
  volume 掛載於 `/srv` 時，以
  [`scripts/configure_srv_scratch.sh`](benchmark_suite/scripts/configure_srv_scratch.sh)
  將 Ollama models、Docker data-root、datasets 與 cache 移入其中，惟正式 prediction 與
  result 一律留在持久磁碟，`runs/` 不置於 ephemeral 空間。
- 兩者均設計為 instance store 遭清空之後仍可依相同 resume identity 續跑：模型與資料重新
  下載，結果不受影響。

---

## 三、instance store 的自動掛載

Instance store 於 stop/start 之後將成為一顆全新未格式化的裝置，無法沿用一般的
`/etc/fstab` 掛載流程。當時採行的作法為一支開機腳本搭配一個 systemd unit。

### 三項限制

- **不得將 instance store 寫入 `/etc/fstab`。** stop/start 之後將更換實體主機，裝置為
  全新空白，fstab 之中的 UUID 無法掛載，開機將失敗。
- **不得寫死 `/dev/nvme1n1`。** Nitro 平台的裝置編號並不保證固定，須以 MODEL 字串動態
  偵測；instance store 為 `Amazon EC2 NVMe Instance Storage`，EBS 為
  `Amazon Elastic Block Store`。
- **不得將 `OLLAMA_MODELS` 指向 instance store。** 模型權重僅於載入時讀取一次，置於
  instance store 的唯一後果為每次 stop/start 均須重新下載數十 GB。模型應存放於 EBS。

### 掛載腳本

`/usr/local/sbin/mount-instance-store.sh`，須賦予執行權限：

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

# 未含檔案系統即代表為 stop/start 之後的全新裝置，予以格式化
if ! blkid "$DEV" >/dev/null 2>&1; then
  mkfs.xfs -f -L nvmescratch "$DEV"
fi

mkdir -p "$MNT"
mountpoint -q "$MNT" || mount -o discard,noatime "$DEV" "$MNT"

getent group "$GRP" >/dev/null || groupadd "$GRP"

# setgid 使新建檔案自動繼承群組，同組成員得以互相讀取
for d in scratch outputs datasets; do
  mkdir -p "$MNT/$d"
  chgrp "$GRP" "$MNT/$d"
  chmod 2775 "$MNT/$d"
done
```

需安裝 `xfsprogs`（Ubuntu 執行 `apt-get install -y xfsprogs`）。

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

其中 `Before=ollama.service` 須予保留，以確保 Ollama 啟動時目錄業已存在。

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

`df -h /mnt/nvme` 應顯示約 419 GB，450 GB 為十進位標示容量。建議實際執行一次 `reboot`，
以驗證自動掛載確實生效。

---

## 四、Ollama

Ollama 以 systemd 服務常駐執行，模型權重存放於 EBS（`OLLAMA_MODELS=/opt/ocf-ai/models`），
不因 stop/start 而消失。

```bash
ollama list      # 列出既有模型
ollama ps        # 檢視目前載入的模型及其執行於 GPU 或 CPU
ollama pull <tag>
```

### PROCESSOR 欄位須為 100% GPU

執行任何正式評測之前均應確認。若出現 CPU 的比例，即代表模型連同 KV cache 超出 45 GB
VRAM，部分已切至 CPU 執行，速度將下降一個數量級，所得數字亦不可採用。

上述規則後續寫入評測程式本身：[`TC-STR_and_OminDoc/`](TC-STR_and_OminDoc/) 的 runner
於正式推論之前檢查 `ollama ps` 是否精確顯示 `100% GPU`，不符即擋下評測。該情形歸類為
非模型因素的工程錯誤，與模型自身造成的失敗（空輸出、拒答、重複、格式錯誤）分開處理，
後者照常計分。

### Context 長度須明確指定

專案目標為 65K context。Ollama 不會自行採用 65K，其預設值依 VRAM 自動選定（4k、32k 或
256k），與 65K 並不相符，故每次呼叫均須明確指定 `options.num_ctx`：

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma4:12b-it-qat",
  "prompt": "...",
  "options": { "num_ctx": 65536 }
}'
```

### KV cache 維持 fp16

當時刻意不啟用量化（未設定 `OLLAMA_KV_CACHE_TYPE`），以使第一輪 baseline 採未量化設定
執行，數字方具可比性。65K 之下若 VRAM 不足，`q8_0` 約可節省一半的 KV cache VRAM，惟該項
變更將改變結果的可比性，應視為另一組實驗。

本專案的 runner 僅記錄 KV cache 設定，不修改服務設定。

---

## 五、Docker 與 OmniDocBench 官方評分器

OmniDocBench 的官方評分需要 TeX Live、ImageMagick 與 Ghostscript，故固定於官方建議的
Docker image 之內執行：

```
ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204
```

image 之內已驗證的 runtime 為 Python 3.10.16、TeX Live 2025（pdfTeX
3.141592653-2.6-1.40.28）、ImageMagick 7.1.1-47 與 Ghostscript 9.55.0。

image 的 entrypoint 預設直接執行評測：

```
Entrypoint:  python pdf_validation.py
Cmd:         --config configs/reproduce_chutao_v16_repro_conda_20260408.yaml
WorkingDir:  /workspace
```

```bash
# 更換 config
docker run --rm -v /mnt/nvme:/mnt/nvme <image> --config <指定的 config>

# 進入環境檢視而不執行評測，須覆蓋 entrypoint
docker run --rm -it --entrypoint bash <image>
```

image 內建的 1,355 頁 GT 並非本研究所使用的 v1.6 full set（1,651 頁），runner 拒絕混用，
版本判定與差異詳見
[`VERSION_PINNING.md`](TC-STR_and_OminDoc/OminDocBench/VERSION_PINNING.md)。

image 體積約 32 GB，存放於 instance store，stop/start 之後須重新下載。

---

## 六、多人共用

共用目錄隸屬 `ocfai` 群組並開啟 setgid（`chmod 2775`），於其中建立的檔案將自動繼承群組，
同組成員得以互相讀取，無須自行 `chmod`。

新加入群組之後，必須登出再重新登入方能生效：

```bash
id
# groups 之中應可見 ocfai

# 確認寫入權限
touch /mnt/nvme/scratch/test && ls -l /mnt/nvme/scratch/test && rm /mnt/nvme/scratch/test
# 檔案的群組應顯示 ocfai
```

Docker 亦採相同作法，使用者加入 `docker` 群組即無須 sudo。本專案的正式 runner 一律不會
自行 `sudo`；遭遇 Docker permission denied 時，係重新登入或執行 `newgrp docker` 的問題。

`TC-STR_and_OminDoc/OminDocBench/` 另備一套機制，使其他人在無須擁有者權限的情況下讀取
評測結果並產生多模型比較報告，詳見該目錄 README 的〈生成多模型比較報告〉一節與
`tools/fix_shared_report_permissions.py`。

---

## 七、stop / start 之後的復原步驟

主機關機再開機（並非 reboot）之後，`/mnt/nvme` 將為全新空白。systemd unit 會自動重新
格式化並建立目錄，惟內容須自行補回：

```bash
# 1. 重新下載 container image（約 32 GB，需時數分鐘）
docker pull ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204
docker images

# 2. 將資料集重新放回 /mnt/nvme/datasets/

# 3. 模型無須重新下載（存放於 EBS），確認仍在即可
ollama list
```

本專案的兩套 runner 均可於復原之後依相同 resume identity 自 checkpoint 續跑。

---

## 八、應予避免的操作

- **不得將須保留的資料僅存放於 `/mnt/nvme/`。** 為最易發生且無法補救的錯誤。
- **不得修改 `/etc/fstab`、Ollama 服務設定或 Docker `daemon.json`。** 上述檔案的配置係
  為使 stop/start 之後仍能正常開機，變更極易導致主機無法啟動；實習期間曾因此故障一次。
- **不得手動更動 `/mnt/nvme/containerd/` 與 `/mnt/nvme/docker/`**，該處為 Docker 的內部
  儲存。
- **磁碟空間不足時不得自行刪除系統檔案。**
- **不得於評測執行途中變更參數。** 本專案的 runner 不會為遷就 OOM 而自動調降 `num_ctx`
  或 `num_predict`，亦不會自行更換量化；正確作法為停止執行、改用 constrained profile，
  並視為另一次實驗。

遭遇儲存、GPU 或服務無法啟動的狀況，應先回報再行處置。

---

## 九、當時已知的限制

以下為實習期間該主機的狀態，一併記錄以免誤讀結果：

- **成果備份僅仰賴 EBS（`/opt/ocf-ai/outputs/`），並無 S3 同步**，亦無將
  `/mnt/nvme/outputs/` 定時搬移至 EBS 的排程，須手動搬移。
- **model tag 曾出現落差。** 規劃文件所載的 `gemma4:31b` 在 Ollama registry 之上並無
  對應，主機上實際存在的為 `gemma4:{e2b,e4b,12b,26b-a4b,31b}-it-qat` 一組 QAT 版本。
  本專案後續堅持記錄 exact tag 與 digest 而非僅記型號，原因即在於此，參見
  [`TC_STR/models.yaml`](TC-STR_and_OminDoc/TC_STR/models.yaml) 與
  [`MODEL_CANDIDATES.md`](TC-STR_and_OminDoc/TC_STR/MODEL_CANDIDATES.md)。
- 本地工作站到貨之前，AWS 主機為暫時方案；`benchmark_suite/` 的 `/srv` scratch 配置
  對應的則是另一種主機環境。

---

## 相關文件

- [README.md](README.md)：研究成果與核心發現
- [`TC-STR_and_OminDoc/README.md`](TC-STR_and_OminDoc/README.md)：完整的系統環境紀錄與
  可比性原則
- [`benchmark_suite/README.md`](benchmark_suite/README.md)：另一種主機配置下的 scratch
  與持久儲存分離
- [HARDWARE.md](HARDWARE.md)：硬體規劃與成本參考
