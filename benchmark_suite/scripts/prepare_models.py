#!/usr/bin/env python3
"""依 YAML 順序先 pull 全部模型，再保存實際 digest/show metadata。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import requests

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE / "src"))

from ocf_benchmark.config import load_config  # noqa: E402
from ocf_benchmark.ollama_client import OllamaClient  # noqa: E402


def cli_quantization(model: str) -> str:
    proc = subprocess.run(
        ["ollama", "show", model, "--verbose"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        return ""
    match = re.search(r"^\s*quantization\s+(\S+)\s*$", proc.stdout, flags=re.MULTILINE)
    return match.group(1) if match else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SUITE / "configs/experiment.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    client = OllamaClient(cfg["ollama"]["host"])
    for model in cfg["models"]:
        if not args.verify_only:
            subprocess.run(["ollama", "pull", model["tag"]], check=True)
    tags = requests.get(f"{cfg['ollama']['host'].rstrip('/')}/api/tags", timeout=30)
    tags.raise_for_status()
    listed = tags.json().get("models", [])
    metadata = []
    for model in cfg["models"]:
        entry = next(
            (
                item
                for item in listed
                if item.get("name") == model["tag"] or item.get("model") == model["tag"]
            ),
            None,
        )
        if entry is None:
            raise RuntimeError(f"ollama list 找不到已準備模型 {model['tag']}")
        show = client.model_metadata(model["tag"])
        show["digest"] = entry.get("digest") or show.get("digest")
        quantization = (entry.get("details") or {}).get("quantization_level") or show.get(
            "quantization"
        )
        if not quantization or str(quantization).lower() == "unknown":
            quantization = cli_quantization(model["tag"])
        if not quantization:
            raise RuntimeError(f"無法解析實際 quantization: {model['tag']}")
        show["quantization"] = quantization
        show["logical_id"] = model["id"]
        show["third_party_quantization"] = model.get("third_party_quantization", False)
        metadata.append(show)
    Path(args.output).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
