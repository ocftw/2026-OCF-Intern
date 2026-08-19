from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    sample_id: str
    sample_index: int
    image_path: Path
    ground_truth: Any
    subject: str = ""
    dataset_key: str = ""
