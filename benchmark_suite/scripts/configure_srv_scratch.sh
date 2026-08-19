#!/usr/bin/env bash
# 將可重建的大型 artifacts 放到 instance-local /srv。
# 此腳本會修改 systemd/Docker 設定，必須由 repository 使用者以 sudo 執行。
set -euo pipefail

SRV_ROOT="/srv/ocf-benchmark"
OLLAMA_TARGET="${SRV_ROOT}/ollama-models"
DOCKER_TARGET="${SRV_ROOT}/docker"
WORK_TARGET="${SRV_ROOT}/work"
OLLAMA_SOURCE="/opt/ollama/models"
OLLAMA_BACKUP="/opt/ollama/models.pre-srv-backup"
DOCKER_CONFIG="/etc/docker/daemon.json"
OLLAMA_OVERRIDE="/etc/systemd/system/ollama.service.d/override.conf"
DOCKER_OVERRIDE="/etc/systemd/system/docker.service.d/ocf-srv.conf"
PREPARE_UNIT="/etc/systemd/system/ocf-srv-prepare.service"
OLD_TMPFILES_CONFIG="/etc/tmpfiles.d/ocf-benchmark.conf"
MODE="configure"
if (($#)); then
  [[ $# -eq 1 && "$1" == "--cleanup-backup" ]] ||
    { echo "用法: sudo $0 [--cleanup-backup]" >&2; exit 2; }
  MODE="cleanup"
fi
OLLAMA_STOPPED=0
SOURCE_MOVED=0
CONFIG_WRITTEN=0

restore_service_on_exit() {
  local rc=$?
  if ((rc != 0 && SOURCE_MOVED && !CONFIG_WRITTEN)) &&
    [[ ! -e "$OLLAMA_SOURCE" && -d "$OLLAMA_BACKUP" ]]; then
    echo "[rollback] 還原原始 Ollama model 目錄" >&2
    mv "$OLLAMA_BACKUP" "$OLLAMA_SOURCE" || true
  fi
  if ((OLLAMA_STOPPED)); then
    systemctl start ollama || true
  fi
  exit "$rc"
}
trap restore_service_on_exit EXIT

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

wait_for_ollama() {
  local attempt
  for attempt in {1..24}; do
    if curl --fail --silent --max-time 5 http://localhost:11434/api/tags >/dev/null; then
      return 0
    fi
    echo "[wait] Ollama 正在啟動（${attempt}/24）"
    sleep 5
  done
  return 1
}

[[ ${EUID} -eq 0 ]] || fail "請使用 sudo $0"
[[ "$(findmnt -n -o TARGET --target /srv 2>/dev/null)" == "/srv" ]] ||
  fail "/srv 不是獨立 mount，拒絕變更"
findmnt -n -o OPTIONS --target /srv | grep -qw rw || fail "/srv 不是可寫 mount"

OWNER="${SUDO_USER:-}"
[[ -n "$OWNER" && "$OWNER" != "root" ]] ||
  fail "無法判斷 repository 使用者；請用一般帳號執行 sudo $0"
OWNER_GROUP="$(id -gn "$OWNER")"
id ollama >/dev/null 2>&1 || fail "找不到 ollama system user"
getent group docker >/dev/null 2>&1 || fail "找不到 docker group"
command -v rsync >/dev/null 2>&1 ||
  fail "找不到 rsync；請先執行 sudo apt-get install -y rsync"

if [[ "$MODE" == "cleanup" ]]; then
  [[ -d "$OLLAMA_BACKUP" && ! -L "$OLLAMA_BACKUP" ]] ||
    fail "找不到安全備份目錄 $OLLAMA_BACKUP"
  [[ -d "$OLLAMA_TARGET" && ! -L "$OLLAMA_TARGET" ]] ||
    fail "找不到 /srv model 目錄 $OLLAMA_TARGET"
  systemctl is-active --quiet ollama || fail "Ollama 未 active，拒絕清除備份"
  wait_for_ollama || fail "Ollama API 無法連線"
  if [[ -n "$(rsync -aHAXn --numeric-ids --delete --itemize-changes \
    "${OLLAMA_BACKUP}/" "${OLLAMA_TARGET}/")" ]]; then
    fail "/srv models 與舊備份不一致，拒絕清除"
  fi
  echo "[stage] 清除已驗證的舊 Ollama model 備份"
  find "$OLLAMA_BACKUP" -xdev -mindepth 1 -delete
  rmdir "$OLLAMA_BACKUP"
  echo "[OK] 已清除 $OLLAMA_BACKUP"
  exit 0
fi

if [[ -s "$DOCKER_CONFIG" ]] &&
  ! grep -Eq '"data-root"[[:space:]]*:[[:space:]]*"/srv/ocf-benchmark/docker"' "$DOCKER_CONFIG"; then
  fail "$DOCKER_CONFIG 已有其他設定，為避免覆蓋請先人工合併 data-root"
fi

echo "[stage] 建立 /srv ephemeral 目錄"
install -d -m 0755 -o root -g root "$SRV_ROOT"
install -d -m 0755 -o ollama -g ollama "$OLLAMA_TARGET"
install -d -m 0710 -o root -g docker "$DOCKER_TARGET"
install -d -m 0755 -o "$OWNER" -g "$OWNER_GROUP" "$WORK_TARGET"
install -d -m 0755 -o "$OWNER" -g "$OWNER_GROUP" \
  "${WORK_TARGET}/data" "${WORK_TARGET}/cache"

if [[ -d "$OLLAMA_BACKUP" ]]; then
  if [[ -n "$(rsync -aHAXn --numeric-ids --delete --itemize-changes \
    "${OLLAMA_BACKUP}/" "${OLLAMA_TARGET}/")" ]]; then
    fail "先前遷移備份與 /srv models 不一致，拒絕接續"
  fi
  echo "[stage] 偵測到完整的先前複製，從 service 配置階段接續"
elif [[ -f "$OLLAMA_OVERRIDE" ]] &&
  grep -Fq "OLLAMA_MODELS=${OLLAMA_TARGET}" "$OLLAMA_OVERRIDE"; then
  echo "[stage] Ollama 已指向 /srv；保留目前 ephemeral model store"
elif [[ -d "$OLLAMA_SOURCE" ]]; then
  echo "[stage] 複製既有 Ollama models（來源暫時保留）"
  rsync -aHAX --numeric-ids --info=progress2 "${OLLAMA_SOURCE}/" "${OLLAMA_TARGET}/"
  systemctl stop ollama
  OLLAMA_STOPPED=1
  rsync -aHAX --numeric-ids --delete "${OLLAMA_SOURCE}/" "${OLLAMA_TARGET}/"
  if [[ -n "$(rsync -aHAXn --numeric-ids --delete --itemize-changes \
    "${OLLAMA_SOURCE}/" "${OLLAMA_TARGET}/")" ]]; then
    systemctl start ollama
    fail "Ollama model 複製驗證失敗；未切換設定"
  fi
  mv "$OLLAMA_SOURCE" "$OLLAMA_BACKUP"
  SOURCE_MOVED=1
fi

echo "[stage] 寫入開機重建與 service mount dependency"
install -d -m 0755 /etc/systemd/system/ollama.service.d
install -d -m 0755 /etc/systemd/system/docker.service.d
printf '%s\n' \
  "[Unit]" \
  "Description=Prepare ephemeral /srv directories for OCF benchmark" \
  "RequiresMountsFor=/srv" \
  "Before=docker.service ollama.service" \
  "" \
  "[Service]" \
  "Type=oneshot" \
  "ExecStart=/usr/bin/install -d -m 0755 -o root -g root ${SRV_ROOT}" \
  "ExecStart=/usr/bin/install -d -m 0755 -o ollama -g ollama ${OLLAMA_TARGET}" \
  "ExecStart=/usr/bin/install -d -m 0710 -o root -g docker ${DOCKER_TARGET}" \
  "ExecStart=/usr/bin/install -d -m 0755 -o ${OWNER} -g ${OWNER_GROUP} ${WORK_TARGET}" \
  "ExecStart=/usr/bin/install -d -m 0755 -o ${OWNER} -g ${OWNER_GROUP} ${WORK_TARGET}/data" \
  "ExecStart=/usr/bin/install -d -m 0755 -o ${OWNER} -g ${OWNER_GROUP} ${WORK_TARGET}/cache" \
  "RemainAfterExit=yes" \
  "" \
  "[Install]" \
  "WantedBy=multi-user.target" \
  >"$PREPARE_UNIT"
printf '%s\n' \
  "[Unit]" \
  "RequiresMountsFor=/srv" \
  "Requires=ocf-srv-prepare.service" \
  "After=ocf-srv-prepare.service" \
  "" \
  "[Service]" \
  "Environment=\"OLLAMA_MODELS=${OLLAMA_TARGET}\"" \
  >"$OLLAMA_OVERRIDE"
printf '%s\n' \
  "[Unit]" \
  "RequiresMountsFor=/srv" \
  "Requires=ocf-srv-prepare.service" \
  "After=ocf-srv-prepare.service" \
  >"$DOCKER_OVERRIDE"
printf '%s\n' \
  "{" \
  "  \"data-root\": \"${DOCKER_TARGET}\"" \
  "}" \
  >"$DOCKER_CONFIG"
CONFIG_WRITTEN=1

rm -f "$OLD_TMPFILES_CONFIG"
systemctl daemon-reload
systemctl enable ocf-srv-prepare.service
systemctl restart ocf-srv-prepare.service
systemctl restart docker
systemctl restart ollama
OLLAMA_STOPPED=0

echo "[stage] 驗證 service"
systemctl is-active --quiet docker || fail "Docker restart 後未 active"
systemctl is-active --quiet ollama || fail "Ollama restart 後未 active"
ACTUAL_DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
[[ "$ACTUAL_DOCKER_ROOT" == "$DOCKER_TARGET" ]] ||
  fail "Docker data-root 仍是 $ACTUAL_DOCKER_ROOT"
wait_for_ollama || fail "Ollama API 無法連線"

echo "[OK] /srv scratch 已配置"
echo "Docker data-root: ${ACTUAL_DOCKER_ROOT}"
echo "Ollama models: ${OLLAMA_TARGET}"
echo "Benchmark scratch: ${WORK_TARGET}"
if [[ -d "$OLLAMA_BACKUP" ]]; then
  echo "舊模型安全備份仍在: ${OLLAMA_BACKUP}"
  echo "驗證 model tags/digests 後，可執行：sudo $0 --cleanup-backup"
fi
