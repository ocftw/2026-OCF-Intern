#!/usr/bin/env python3
"""下載所有資料後才允許推論；revision 由 YAML 明確指定。"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE / "src"))

from ocf_benchmark.config import load_config, resolve_path  # noqa: E402


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    base = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(base):
            raise ValueError(f"不安全的 tar member: {member.name}")
    # 已在上方逐一驗證 resolved target；不用 Python 3.12 才加入的 filter 參數，
    # 以維持宣告的 Python 3.10+ 相容性。
    archive.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SUITE / "configs/experiment.yaml"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    cache_dir = resolve_path(cfg, cfg["paths"]["cache_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for benchmark in cfg["benchmarks"]:
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
                snapshot_download(
                    repo_id=benchmark["dataset"],
                    repo_type="dataset",
                    revision=benchmark["revision"],
                    local_dir=local,
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
