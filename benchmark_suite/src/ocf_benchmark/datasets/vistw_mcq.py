"""VisTW-MCQ 21 subjects loader，固定 HF revision 與各 subject 原始順序。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

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
    image_cache.mkdir(parents=True, exist_ok=True)
    index = 0
    for subject in SUBJECTS:
        parquet = (snapshot / subject / f"{split}-00000-of-00001.parquet") if snapshot else None
        dataset = (
            load_dataset("parquet", data_files=str(parquet), split="train")
            if parquet is not None and parquet.exists()
            else load_dataset(dataset_name, subject, split=split, revision=revision)
        )
        for row in dataset:
            qid = str(row["qid"])
            path = image_cache / subject / f"{hashlib.sha256(qid.encode()).hexdigest()}.png"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                row["image"].save(path, format="PNG")
            ground_truth = {
                "answer": str(row["answer"]),
                "question": str(row["question"]),
                "A": str(row["A"]),
                "B": str(row["B"]),
                "C": str(row["C"]),
                "D": str(row["D"]),
            }
            yield Sample(qid, index, path, ground_truth, subject, qid)
            index += 1
