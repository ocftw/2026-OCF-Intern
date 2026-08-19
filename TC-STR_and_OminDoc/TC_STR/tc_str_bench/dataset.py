from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path
from typing import Any

from .util import atomic_write_json, canonical_json, sha256_file, utc_now


def image_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:2] == b"\xff\xd8":
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            offset += 2
            if marker in {0xD8, 0xD9}:
                continue
            if offset + 2 > len(data):
                break
            size = struct.unpack(">H", data[offset : offset + 2])[0]
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
                return width, height
            offset += size
    if data[:6] in {b"GIF87a", b"GIF89a"}:
        return struct.unpack("<HH", data[6:10])
    raise ValueError(f"無法讀取圖片尺寸: {path}")


def _choose_smoke(samples: list[dict[str, Any]], count: int = 20) -> tuple[list[int], dict[str, Any]]:
    """Fixed feature-stratified selection; official order is the tie-breaker."""
    chosen: list[int] = []

    def take(predicate: Any, n: int) -> None:
        candidates = [s for s in samples if s["index"] not in chosen and predicate(s)]
        if not candidates:
            return
        if n == 1:
            positions = [len(candidates) // 2]
        else:
            positions = [round(i * (len(candidates) - 1) / (n - 1)) for i in range(n)]
        for position in positions:
            index = candidates[position]["index"]
            if index not in chosen:
                chosen.append(index)

    take(lambda s: len(s["ground_truth"]) <= 2, 4)
    take(lambda s: len(s["ground_truth"]) >= 10, 4)
    take(lambda s: bool(re.search(r"[A-Za-z]", s["ground_truth"])), 3)
    take(lambda s: bool(re.search(r"\d", s["ground_truth"])), 3)
    take(lambda s: bool(re.search(r"[\s，。！？、：；,.!?():/&+\-]", s["ground_truth"])), 3)
    take(lambda s: s["width"] / max(s["height"], 1) >= 8, 2)
    take(lambda s: s["height"] / max(s["width"], 1) >= 1.5, 2)
    areas = sorted(s["width"] * s["height"] for s in samples)
    high_area = areas[int(len(areas) * 0.85)]
    take(lambda s: s["width"] * s["height"] >= high_area, 3)
    take(lambda s: True, count - len(chosen))
    selected = sorted(chosen[:count])
    definitions = {
        "short_len_le_2": lambda s: len(s["ground_truth"]) <= 2,
        "long_len_ge_10": lambda s: len(s["ground_truth"]) >= 10,
        "latin": lambda s: bool(re.search(r"[A-Za-z]", s["ground_truth"])),
        "digits": lambda s: bool(re.search(r"\d", s["ground_truth"])),
        "punctuation_or_space": lambda s: bool(re.search(r"[\s，。！？、：；,.!?():/&+\-]", s["ground_truth"])),
        "very_wide": lambda s: s["width"] / max(s["height"], 1) >= 8,
        "tall": lambda s: s["height"] / max(s["width"], 1) >= 1.5,
        "large_area_top_15_percent": lambda s: s["width"] * s["height"] >= high_area,
    }
    selected_rows = [samples[index] for index in selected]
    coverage = {
        name: {
            "available_in_test": sum(bool(predicate(sample)) for sample in samples),
            "selected": sum(bool(predicate(sample)) for sample in selected_rows),
        }
        for name, predicate in definitions.items()
    }
    unavailable = [name for name, counts in coverage.items() if counts["available_in_test"] == 0]
    return selected, {"strata": coverage, "unavailable_strata": unavailable}


def build_manifest(dataset_dir: Path, output: Path | None = None, expected: int = 3706) -> dict[str, Any]:
    label_path = dataset_dir / "test_labels.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"找不到官方 test split: {label_path}")
    raw = label_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    samples: list[dict[str, Any]] = []
    for source_line, line in enumerate(lines, 1):
        if not line:
            raise ValueError(f"test_labels.txt 第 {source_line} 行為空白")
        if "\t" not in line:
            raise ValueError(f"test_labels.txt 第 {source_line} 行沒有 tab")
        relative, ground_truth = line.split("\t", 1)
        if not relative or not ground_truth:
            raise ValueError(f"test_labels.txt 第 {source_line} 行 path 或 label 為空")
        image = dataset_dir / relative
        if not image.is_file():
            raise FileNotFoundError(f"test image 不存在: {image}")
        width, height = image_dimensions(image)
        samples.append(
            {
                "index": len(samples),
                "source_line": source_line,
                "image_relative_path": relative,
                "ground_truth": ground_truth,
                "image_sha256": sha256_file(image),
                "image_bytes": image.stat().st_size,
                "width": width,
                "height": height,
            }
        )
    if len(samples) != expected:
        raise ValueError(f"TC-STR test 有效樣本必須為 {expected}，實際 {len(samples)}")
    fingerprint_material = [
        [s["index"], s["image_relative_path"], s["ground_truth"], s["image_sha256"]]
        for s in samples
    ]
    fingerprint = hashlib.sha256(canonical_json(fingerprint_material).encode()).hexdigest()
    smoke_indices, smoke_coverage = _choose_smoke(samples, 20)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "dataset": "TC-STR 7k-word",
        "source_url": "https://github.com/esun-ai/traditional-chinese-text-recogn-dataset",
        "split": "test",
        "label_file": "test_labels.txt",
        "label_file_sha256": sha256_file(label_path),
        "sample_count": len(samples),
        "dataset_fingerprint": fingerprint,
        "smoke_selection": {
            "version": "tcstr_stratified_v1",
            "rule": "official-order deterministic strata: short, long, Latin, digits, punctuation/space, large image; quantile positions with official index tie-break",
            "sample_indices": smoke_indices,
            "coverage": smoke_coverage,
            "warning": (
                "官方 test split 不含 Latin、digits、punctuation/space ground truth；"
                "遵守 test-only 規則，不從 train 補樣本。"
                if smoke_coverage["unavailable_strata"] else None
            ),
        },
        "samples": samples,
    }
    if output:
        atomic_write_json(output, manifest)
    return manifest
