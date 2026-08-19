#!/usr/bin/env python3
"""下載所有資料後才允許推論；revision 由 YAML 明確指定。"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

# OmniDocBench 有大量小檔。匿名 Xet token endpoint 容易先於檔案下載本身觸發 429；
# Hugging Face 官方提供此開關以退回一般 HTTP。使用者仍可明確設為 0 啟用 Xet。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

from huggingface_hub import snapshot_download

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE / "src"))

from ocf_benchmark.config import load_config, resolve_path  # noqa: E402

DEFAULT_HF_RETRY_DELAYS = (60, 120, 300, 600, 900, 1200, 1800)


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    base = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(base):
            raise ValueError(f"不安全的 tar member: {member.name}")
    # 已在上方逐一驗證 resolved target；不用 Python 3.12 才加入的 filter 參數，
    # 以維持宣告的 Python 3.10+ 相容性。
    archive.extractall(destination)


def _retryable_download_error(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    # hf-xet 將 token/transport failure 包成 built-in ConnectionError。
    return isinstance(exc, (ConnectionError, TimeoutError, requests.RequestException))


def snapshot_download_with_retry(
    *,
    snapshot_fn: Callable[..., Any] = snapshot_download,
    sleep_fn: Callable[[float], None] = time.sleep,
    retry_delays: tuple[int, ...] = DEFAULT_HF_RETRY_DELAYS,
    **kwargs: Any,
) -> Any:
    for attempt in range(len(retry_delays) + 1):
        try:
            return snapshot_fn(**kwargs)
        except Exception as exc:
            if not _retryable_download_error(exc) or attempt == len(retry_delays):
                raise
            delay = retry_delays[attempt]
            print(
                f"[retry] Hugging Face download attempt {attempt + 1} failed: "
                f"{type(exc).__name__}: {exc}; {delay}s 後沿用 cache 重試",
                file=sys.stderr,
                flush=True,
            )
            sleep_fn(delay)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SUITE / "configs/experiment.yaml"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=["omnidocbench", "tc_str", "vistw_mcq"],
        help="只準備指定 benchmark；可重複傳入",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    cache_dir = resolve_path(cfg, cfg["paths"]["cache_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for benchmark in cfg["benchmarks"]:
        if args.benchmark and benchmark["id"] not in args.benchmark:
            continue
        if benchmark["id"] == "tc_str":
            root = data_dir / "TC-STR"
            if not (root / "test_labels.txt").exists():
                if args.verify_only:
                    raise FileNotFoundError(root / "test_labels.txt")
                response = requests.get(benchmark["download_url"], timeout=180)
                response.raise_for_status()
                with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
                    safe_extract(archive, data_dir)
            count = len((root / "test_labels.txt").read_text(encoding="utf-8").splitlines())
            if count != 3706:
                raise ValueError(f"TC-STR test count={count}，預期 3706")
            manifest.append({"id": "tc_str", "revision": benchmark["revision"], "count": count})
        else:
            local = cache_dir / "hf" / benchmark["id"] / benchmark["revision"]
            if args.verify_only and not local.exists():
                raise FileNotFoundError(local)
            if not args.verify_only:
                max_workers = int(os.environ.get("HF_DOWNLOAD_MAX_WORKERS", "2"))
                if max_workers < 1:
                    raise ValueError("HF_DOWNLOAD_MAX_WORKERS 必須至少為 1")
                snapshot_download_with_retry(
                    repo_id=benchmark["dataset"],
                    repo_type="dataset",
                    revision=benchmark["revision"],
                    local_dir=local,
                    max_workers=max_workers,
                )
            manifest.append(
                {"id": benchmark["id"], "revision": benchmark["revision"], "snapshot": str(local)}
            )
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
