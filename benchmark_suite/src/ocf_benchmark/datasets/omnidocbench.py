"""OmniDocBench full-page loader。

HF v1.7 可能調整 imagefolder 欄位名稱，因此只接受明確 image/page_info 欄位，
不猜測 ground truth。官方 JSON 仍由 evaluator 直接使用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .common import Sample


def locate_ground_truth(snapshot: Path) -> Path:
    candidates = [
        path
        for path in snapshot.rglob("*.json")
        if "demo" not in path.name.lower() and "readme" not in path.name.lower()
    ]
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, list) and data and "layout_dets" in data[0]:
            return path
    raise FileNotFoundError("HF snapshot 中找不到 OmniDocBench full annotation JSON")


def load_samples(
    dataset_name: str,
    revision: str,
    image_cache: Path,
    split: str = "train",
    snapshot: Path | None = None,
) -> Iterable[Sample]:
    del dataset_name, revision, image_cache, split
    if snapshot is None:
        raise ValueError("OmniDocBench 必須先 snapshot_download 固定 revision")
    gt_path = locate_ground_truth(snapshot)
    pages = json.loads(gt_path.read_text(encoding="utf-8"))
    if len(pages) != 1651:
        raise ValueError(f"OmniDocBench v1.7 full 應為 1651 頁，實際 {len(pages)}")
    for index, row in enumerate(pages):
        page_info = row.get("page_info") or {}
        key = str(page_info.get("image_path") or row.get("image_path") or "")
        if not key:
            raise ValueError(f"OmniDocBench annotation index={index} 缺少 image_path")
        path = snapshot / key
        if not path.exists():
            path = snapshot / "images" / Path(key).name
        if not path.exists():
            raise FileNotFoundError(path)
        yield Sample(Path(key).name, index, path, row, dataset_key=key)
