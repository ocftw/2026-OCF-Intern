"""TC-STR 7k-word raw format loader（正式 test 固定 3,706 筆）。"""

from __future__ import annotations

from pathlib import Path

from .common import Sample


def load_samples(root: Path, split: str = "test") -> list[Sample]:
    labels = root / f"{split}_labels.txt"
    if not labels.exists():
        raise FileNotFoundError(f"找不到 {labels}；請先執行 prepare-data")
    samples = []
    for index, line in enumerate(labels.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        relative, ground_truth = line.split("\t", 1)
        image_path = root / relative.strip()
        samples.append(
            Sample(
                sample_id=relative.strip(),
                sample_index=index,
                image_path=image_path,
                ground_truth=ground_truth,
                dataset_key=relative.strip(),
            )
        )
    if split == "test" and len(samples) != 3706:
        raise ValueError(f"TC-STR test 應為 3706 筆，實際 {len(samples)}")
    return samples
