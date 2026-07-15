#!/usr/bin/env bash
# 依序跑完 config.py 裡的每個模型，一次跑一個（同一個模型湊滿 LIMIT 筆「真正成功」才換
# 下一個），避免同一輪內來回切換模型造成 Ollama 不穩定。斷線/Ollama掛掉都自動重試，直到
# 完成為止。失敗滿 MAX_RETRIES_PER_ITEM 次的圖片會自動放棄，改用資料集後面的圖遞補，確保
# 最終真的湊滿 LIMIT 筆成功結果，不是「成功+放棄」混著算。
#
# 用法：
#   chmod +x run_until_done.sh
#   tmux new -s tcstreval
#   ./run_until_done.sh
#   (Ctrl+B 再按 D 離開 tmux，安全斷線)
#
# 回來看進度：
#   cat results/RUN_STATUS.txt
#   tail -f results/logs/until_done_*.log

set -uo pipefail
cd "$(dirname "$0")"

# 自動偵測可用的 python 指令（有些系統只有 python3，沒有 python 這個別名；
# 用 ./script.sh 這種非互動式方式執行時，也可能吃不到 .bashrc 裡設定的 PATH）。
# 如果偵測到的不是你要的那個（例如你在用 conda 環境），可以用
# PYTHON_BIN=/path/to/python ./run_until_done.sh 覆蓋。
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
if [ -z "$PYTHON_BIN" ]; then
    echo "[!] 找不到 python3 或 python，請用 PYTHON_BIN=/path/to/python ./run_until_done.sh 指定直譯器路徑。" >&2
    exit 1
fi
echo "[*] 使用直譯器: $PYTHON_BIN ($(command -v "$PYTHON_BIN"))"

# 改成你要跑的筆數，0 = 整個 test_labels.txt（正式跑，會很耗時，建議先用
# run_eval.py --limit 20 手動測過三個模型都正常之後，再用這支腳本跑正式的量）。
LIMIT=200
MODELS_TO_RUN=("qwen" "chandra" "glm_ocr")
# 已知有問題、Ollama 一律回錯的圖片檔名，排除掉並用下一張遞補，不然會永遠卡在重試迴圈裡
# 出不來。發現新的問題圖片就加進這個陣列（不過下面的 MAX_RETRIES_PER_ITEM 機制通常會自動
# 處理掉這類狀況，不一定要手動加）。
EXCLUDE_IMAGES=()
# 同一張圖同一個模型連續失敗幾次後就自動放棄重試（例如圖片本身有問題，重試也不會好）
MAX_RETRIES_PER_ITEM=3
OUT_CSV="results/raw_results.csv"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/until_done_${TS}.log"
STATUS_FILE="results/RUN_STATUS.txt"
SLEEP_SECONDS=120
MAX_ATTEMPTS=100

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"; }

check_model_done() {
    # 回傳 0 = 該模型已經達到「目標成功筆數」，1 = 還沒；同時印出目前進度。
    # 注意：完成判定只看「真正成功」的數量，不是 success+放棄 湊數——因為
    # run_eval.py 自己會用資料池後面的圖遞補放棄掉的，所以只要資料池夠大，
    # 最終目標就是真的湊滿 --limit 筆成功結果，不是「成功+放棄」都算數。
    local model_key="$1"
    "$PYTHON_BIN" - "$OUT_CSV" "$LIMIT" "$model_key" "$MAX_RETRIES_PER_ITEM" <<'PYEOF'
import sys
import pandas as pd

out_csv, limit, model_key, max_retries = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
expected = limit

try:
    df = pd.read_csv(out_csv, keep_default_na=False, na_values=[])
except FileNotFoundError:
    print(f"{model_key}: 0/{expected}")
    sys.exit(1)

sub = df[df["model_key"] == model_key]
success = 0
gave_up = 0
for _, group in sub.groupby("image_filename"):
    if (group["error"].astype(str) == "").any():
        success += 1
    elif len(group) >= max_retries:
        gave_up += 1

suffix = f"（另有{gave_up}張已放棄、用其他圖遞補中）" if gave_up else ""
print(f"{model_key}: {success}/{expected}{suffix}")
sys.exit(0 if success >= expected else 1)
PYEOF
}

# 把「所有」模型目前的進度（不只是正在跑的那個）都寫進 STATUS_FILE，
# 隨時 cat 都能看到完整全貌，不用等到整個腳本結束才更新。
write_status() {
    local overall_state="$1"      # 執行中 / 完成 / 失敗
    local current_model="$2"      # 目前正在處理的模型（可空白）
    local current_attempt="$3"    # 目前第幾次嘗試（可空白）
    {
        echo "狀態: ${overall_state}"
        echo "最後更新: $(date '+%Y-%m-%d %H:%M:%S')"
        if [ -n "$current_model" ]; then
            echo "目前處理模型: ${current_model}（第 ${current_attempt} 次嘗試）"
        fi
        for m in "${MODELS_TO_RUN[@]}"; do
            local prog
            prog=$(check_model_done "$m" 2>/dev/null)
            echo "  - ${prog}"
        done
    } > "$STATUS_FILE"
}

EXCLUDE_ARGS=()
if [ "${#EXCLUDE_IMAGES[@]}" -gt 0 ]; then
    EXCLUDE_ARGS=(--exclude "${EXCLUDE_IMAGES[@]}")
fi

write_status "執行中" "" ""
log "===== 開始跑 --limit ${LIMIT}，模型：${MODELS_TO_RUN[*]}（一次一個，跑完才換下一個）====="

for model_key in "${MODELS_TO_RUN[@]}"; do
    log "===== 現在處理模型: ${model_key} ====="
    attempt=0
    model_done=0

    while [ $attempt -lt $MAX_ATTEMPTS ]; do
        attempt=$((attempt + 1))
        write_status "執行中" "$model_key" "$attempt"
        log "[*] [${model_key}] 第 ${attempt} 次嘗試 ($PYTHON_BIN -u run_eval.py --limit ${LIMIT} --models ${model_key} --resume --max-retries ${MAX_RETRIES_PER_ITEM})..."

        # -u：強制 Python 即時輸出（不整批緩衝），這樣 log 檔案 / tail -f 才看得到即時進度。
        # 丟到背景執行，讓下面的迴圈可以每 30 秒更新一次 STATUS_FILE——不然單次
        # run_eval.py 可能要跑很久，RUN_STATUS.txt 會一直卡在這次開始前的舊數字。
        "$PYTHON_BIN" -u run_eval.py --limit "$LIMIT" --out "$OUT_CSV" --models "$model_key" --resume \
            --max-retries "$MAX_RETRIES_PER_ITEM" "${EXCLUDE_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
        eval_pid=$!
        while kill -0 "$eval_pid" 2>/dev/null; do
            sleep 30
            write_status "執行中" "$model_key" "$attempt"
        done
        wait "$eval_pid"
        ret=$?

        check_output=$(check_model_done "$model_key")
        check_ret=$?
        log "[*] [${model_key}] 目前狀態: ${check_output}"
        write_status "執行中" "$model_key" "$attempt"

        if [ $check_ret -eq 0 ]; then
            model_done=1
            log "[+] [${model_key}] 已全部成功完成！"
            break
        fi

        if [ $ret -eq 0 ]; then
            log "[!] [${model_key}] run_eval.py 正常跑完一輪，但還有組合沒成功，${SLEEP_SECONDS} 秒後重跑補齊..."
        else
            log "[!] [${model_key}] run_eval.py 提早中止（很可能 Ollama 斷線），${SLEEP_SECONDS} 秒後重試..."
        fi
        sleep $SLEEP_SECONDS
    done

    if [ $model_done -ne 1 ]; then
        log "[!] [${model_key}] 已經嘗試 ${MAX_ATTEMPTS} 次仍未全部完成，先停下來讓你檢查。"
        write_status "未完成（${model_key} 重試 ${MAX_ATTEMPTS} 次後仍失敗，詳見 $LOG_FILE）" "" ""
        exit 1
    fi
done

log "[*] ${MODELS_TO_RUN[*]} 都完成了，開始產生報告 ($PYTHON_BIN -u build_report.py)..."
"$PYTHON_BIN" -u build_report.py --raw "$OUT_CSV" --out results/report.html >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    write_status "完成" "" ""
    echo "報告: results/report.html" >> "$STATUS_FILE"
    log "[+] 全部完成！報告：results/report.html"
else
    log "[!] build_report.py 失敗，詳見 $LOG_FILE"
    write_status "失敗（build_report.py，詳見 $LOG_FILE）" "" ""
    exit 1
fi
