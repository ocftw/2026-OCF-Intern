# -*- coding: utf-8 -*-
"""
TC-STR OCR 模型評估主腳本
用法（在 eval/ 目錄下執行，或用 --labels/--images-dir 指定路徑）：

    python run_eval.py --limit 50
    python run_eval.py --limit 50 --models qwen chandra
    python run_eval.py --limit 50 --resume        # 中斷了可以接續跑
    python run_eval.py --labels ../train_labels.txt --limit 50   # 改跑 train split

--limit 代表「目標成功筆數」，不是圖片數量上限：如果某張圖片一直失敗（達到
--max-retries 次），會自動放棄它、改用資料集後面的下一張圖遞補，直到湊滿
--limit 筆成功結果為止（前提是資料集裡有足夠多圖片可以遞補）。

跑完之後用 build_report.py 產生一頁式 HTML 報告。
"""

import argparse
import csv
import os
import sys

import pandas as pd

from config import (
    MODELS, OLLAMA_HOST, OCR_PROMPT, DATASET_LABELS, IMAGES_DIR,
    RAW_RESULTS_CSV, ANLS_THRESHOLD,
)
from ollama_client import call_ollama
from metrics import score_all
from postprocess import clean_prediction


def load_pool(labels_path, exclude=None):
    """載入整個候選圖片池（不受 --limit 限制），失敗的圖片會從這裡動態遞補下一張。

    labels_path 是 TC-STR 的 tab 分隔標註檔，每行格式為：
        images/xxx.jpg<TAB>對應文字
    """
    df = pd.read_csv(
        labels_path, sep="\t", header=None, names=["image_path", "text"],
        dtype=str, keep_default_na=False, na_values=[], encoding="utf-8",
    )
    df["image_path"] = df["image_path"].str.strip()
    df["text"] = df["text"].str.strip()
    # 標註檔裡的路徑已經含 "images/" 前綴，IMAGES_DIR 又會再加一次，這裡先把
    # 前綴去掉，只留檔名本身，後面統一用 os.path.join(images_dir, image_filename)。
    df["image_filename"] = df["image_path"].str.replace(r"^images/", "", regex=True)
    df = df[df["image_filename"] != ""]
    if exclude:
        df = df[~df["image_filename"].isin(exclude)]
    return df.reset_index(drop=True)


def load_model_progress(out_path, model_key, max_retries):
    """讀取某個模型目前為止的進度（只看這個 model_key 的歷史紀錄）：
    - resolved: 已經成功或已經放棄重試的圖片檔名集合（本次執行不會再處理）
    - fail_counts: 尚未解決（失敗次數 < max_retries）的圖片 -> 目前失敗次數
    - success_count: 已成功筆數
    """
    resolved = set()
    fail_counts = {}
    success_count = 0
    if not os.path.exists(out_path):
        return resolved, fail_counts, success_count

    prev = pd.read_csv(out_path, keep_default_na=False, na_values=[])
    prev = prev[prev["model_key"] == model_key]
    for img, group in prev.groupby("image_filename"):
        has_success = (group["error"].astype(str) == "").any()
        n_fail = int((group["error"].astype(str) != "").sum())
        if has_success:
            resolved.add(img)
            success_count += 1
        elif n_fail >= max_retries:
            resolved.add(img)
        else:
            fail_counts[img] = n_fail
    return resolved, fail_counts, success_count


FIELDNAMES = [
    "image_filename", "model_key", "model_tag", "ground_truth",
    "prediction", "prediction_raw",
    "em", "cm", "anls", "f1", "latency_sec", "error",
]

# 連續幾次「連不上 Ollama」就提早中止，避免傻傻把剩下的呼叫全部跑成失敗
CONSECUTIVE_CONNECTION_FAILURE_LIMIT = 8
_CONNECTION_ERROR_MARKERS = (
    "Connection refused", "Max retries exceeded",
    "ConnectionError", "Failed to establish a new connection",
)


def _is_connection_error(err):
    return bool(err) and any(marker in err for marker in _CONNECTION_ERROR_MARKERS)


def main():
    parser = argparse.ArgumentParser(description="TC-STR OCR 模型評估：EM / CM(Containment) / ANLS / F1")
    parser.add_argument("--labels", default=DATASET_LABELS,
                         help="tab 分隔標註檔路徑，預設 test_labels.txt")
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    parser.add_argument("--limit", type=int, default=50,
                         help="目標成功筆數（不是圖片數量上限）：失敗且放棄重試的圖片會自動"
                              "用資料集後面的圖遞補，直到湊滿這個成功數；0 = 以整個資料集為目標")
    parser.add_argument("--host", default=OLLAMA_HOST)
    parser.add_argument("--out", default=RAW_RESULTS_CSV)
    parser.add_argument("--models", nargs="*", default=None,
                         help="只跑指定 model key（見 config.py 的 MODELS），例如 --models qwen chandra")
    parser.add_argument("--resume", action="store_true",
                         help="接續 --out 裡面已經有的進度（成功/放棄重試的組合不會重跑）")
    parser.add_argument("--timeout", type=int, default=180, help="每次 Ollama 呼叫的逾時秒數")
    parser.add_argument("--exclude", nargs="*", default=[],
                         help="排除已知有問題的圖片檔名（例如尺寸異常導致 Ollama 一律回 400），"
                              "例如 --exclude billboard_00131_120_某某.jpg")
    parser.add_argument("--max-retries", type=int, default=3,
                         help="同一張圖同一個模型連續失敗幾次後就放棄、自動用下一張圖遞補，預設 3 次")
    args = parser.parse_args()

    models = [m for m in MODELS if not args.models or m["key"] in args.models]
    if not models:
        print(f"[!] 沒有符合的 model，可用 key: {[m['key'] for m in MODELS]}")
        sys.exit(1)

    if not os.path.exists(args.labels):
        print(f"[!] 找不到標註檔: {args.labels}")
        sys.exit(1)

    pool = load_pool(args.labels, exclude=set(args.exclude))
    if len(pool) == 0:
        print("[!] 標註檔篩選之後沒有圖片可評估")
        sys.exit(1)

    print(f"[*] 候選圖片池共 {len(pool)} 張，將用 {len(models)} 個模型評估：{[m['tag'] for m in models]}")
    print(f"[*] Ollama host: {args.host}")
    print(f"[*] 標註檔: {args.labels}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    file_exists = os.path.exists(args.out)
    mode = "a" if (args.resume and file_exists) else "w"

    with open(args.out, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()

        for m in models:
            target = args.limit if args.limit else len(pool)

            if args.resume and file_exists:
                resolved, fail_counts, success_count = load_model_progress(
                    args.out, m["key"], args.max_retries
                )
            else:
                resolved, fail_counts, success_count = set(), {}, 0

            print(f"[*] [{m['key']}] 起始進度: {success_count}/{target} 成功"
                  f"（已放棄 {len(resolved) - success_count} 張）")

            error_count = 0
            consecutive_conn_errors = 0
            aborted = False

            for _, row in pool.iterrows():
                if success_count >= target:
                    break

                img_name = row["image_filename"]
                if img_name in resolved:
                    continue

                image_path = os.path.join(args.images_dir, img_name)
                gt = str(row["text"])

                pred_raw, latency, err = call_ollama(
                    args.host, m["tag"], OCR_PROMPT, image_path, timeout=args.timeout,
                    extra_options=m.get("options"),
                )
                pred = clean_prediction(pred_raw) if not err else ""
                scores = score_all(pred, gt, ANLS_THRESHOLD)

                if err:
                    error_count += 1
                    fail_counts[img_name] = fail_counts.get(img_name, 0) + 1
                    if fail_counts[img_name] >= args.max_retries:
                        resolved.add(img_name)
                else:
                    success_count += 1
                    resolved.add(img_name)
                consecutive_conn_errors = consecutive_conn_errors + 1 if _is_connection_error(err) else 0

                writer.writerow({
                    "image_filename": img_name,
                    "model_key": m["key"], "model_tag": m["tag"],
                    "ground_truth": gt, "prediction": pred,
                    "prediction_raw": pred_raw if not err else "",
                    "em": scores["em"], "cm": scores["cm"],
                    "anls": scores["anls"], "f1": scores["f1"],
                    "latency_sec": round(latency, 3), "error": err or "",
                })
                f.flush()

                tag = f"[錯誤:{err}]" if err else ""
                print(f"    - [{m['key']}] 進度 {success_count}/{target} | {img_name} "
                      f"EM={scores['em']:.0f} CM={scores['cm']:.0f} ANLS={scores['anls']:.2f} "
                      f"F1={scores['f1']:.2f} 累計錯誤={error_count} {tag}", flush=True)

                if consecutive_conn_errors >= CONSECUTIVE_CONNECTION_FAILURE_LIMIT:
                    print(f"\n[!] [{m['key']}] 連續 {consecutive_conn_errors} 次連不上 Ollama"
                          f"（{args.host}），懷疑服務已經掛掉，提早中止，避免浪費剩下的呼叫。")
                    print("[!] 請確認 Ollama 正常運作（例如 curl "
                          f"{args.host}/api/tags）之後，用同一行指令加 --resume 接續。")
                    aborted = True
                    break

            if aborted:
                sys.exit(1)

            if success_count < target:
                print(f"[!] [{m['key']}] 資料池已經用完（共 {len(pool)} 張圖），"
                      f"只湊到 {success_count}/{target} 筆成功，其餘都已放棄重試。")
            else:
                print(f"[+] [{m['key']}] 已達成目標 {target} 筆成功！")

    print(f"\n[+] 完成！原始結果已存到 {args.out}")
    print("[*] 下一步：python build_report.py 產生一頁式 HTML 報告")


if __name__ == "__main__":
    main()
