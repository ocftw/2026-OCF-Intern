"""VisTW-MCQ 21 subjects loader，固定 HF revision 與各 subject 原始順序。"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from PIL import Image

from .common import Sample

SUBJECTS = [
    "accounting",
    "arts",
    "biology",
    "chemistry",
    "chinese_literature",
    "dentistry",
    "electronic_circuits",
    "fundamentals_of_physical_therapy",
    "geography",
    "mathematics",
    "mechanics",
    "medical",
    "music",
    "natural_science",
    "navigation",
    "pharmaceutical_chemistry",
    "physics",
    "sociology",
    "statistics",
    "structural_engineering",
    "veterinary_medicine",
]


def load_samples(
    dataset_name: str,
    revision: str,
    image_cache: Path,
    split: str = "test",
    snapshot: Path | None = None,
) -> Iterable[Sample]:
    del dataset_name, revision
    if snapshot is None:
        raise ValueError("VisTW-MCQ 必須先 snapshot_download 固定 revision")
    image_cache.mkdir(parents=True, exist_ok=True)
    index = 0
    for subject in SUBJECTS:
        parquet = snapshot / subject / f"{split}-00000-of-00001.parquet"
        if not parquet.exists():
            raise FileNotFoundError(parquet)
        parquet_file = pq.ParquetFile(parquet)
        for batch in parquet_file.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                yield _sample_from_row(row, subject, image_cache, index)
                index += 1


def _sample_from_row(row: dict, subject: str, image_cache: Path, index: int) -> Sample:
    qid = str(row["qid"])
    path = image_cache / subject / f"{hashlib.sha256(qid.encode()).hexdigest()}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        image_data = row["image"]
        encoded = image_data.get("bytes") if isinstance(image_data, dict) else None
        if not encoded:
            raise ValueError(f"VisTW {subject}:{qid} parquet image 缺少 embedded bytes")
        with Image.open(io.BytesIO(encoded)) as image:
            image.save(path, format="PNG")
    ground_truth = {
        "answer": str(row["answer"]),
        "question": str(row["question"]),
        "A": str(row["A"]),
        "B": str(row["B"]),
        "C": str(row["C"]),
        "D": str(row["D"]),
    }
    sample_id = f"{subject}:{qid}"
    return Sample(sample_id, index, path, ground_truth, subject, qid)
