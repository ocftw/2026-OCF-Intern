#!/usr/bin/env python3
"""Fetch the four new comparison models (Qwen3-VL 4B/32B, InternVL3.5 4B/38B).

Qwen3-VL ships as a single-file Ollama-native GGUF, so those two entries are a
plain `ollama pull`. InternVL3.5 has no official Ollama entry -- it is
imported from community GGUF quantizations (bartowski / mradermacher) that
ship the language-model weights and the CLIP vision projector (`mmproj`) as
two separate files. Both files are downloaded with resume + SHA-256
verification (mirroring scripts/fetch_dataset.py), then combined into one
local Ollama model with a two-`FROM` Modelfile.

Verified manually on this host on 2026-08-07 with Ollama 0.31.1:
  - qwen3-vl:4b            -> architecture qwen3vl, vision capability present,
                              real (non-hallucinated) OCR output on a sample
                              image.
  - internvl3.5:4b-q4_k_m  -> architecture qwen3 + clip projector, vision
                              capability present, real OCR output on the same
                              sample image.
InternVL3.5-38B uses the same dense Qwen3 backbone as the 4B checkpoint (it is
not one of the "-A" MoE variants), so the same two-FROM Modelfile path is
expected to work; this script does not re-verify that automatically -- run
`--smoke-test` after fetching to confirm on this exact model before flipping
`supervisor_approved: true` in the config_variants file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import urllib.request

SCRATCH_ROOT = pathlib.Path("/mnt/nvme/scratch/model_import")
OLLAMA_API = "http://localhost:11434/api"

# (repo, filename, sha256, size_bytes) for the language-model weights and the
# vision projector of each InternVL import. Pinned from the HF API tree
# listing on 2026-08-07 -- a content change on either repo will fail the
# checksum rather than silently substituting a different file.
INTERNVL_FILES = {
    "internvl3.5_4b": {
        "repo": "bartowski/OpenGVLab_InternVL3_5-4B-GGUF",
        "main": (
            "OpenGVLab_InternVL3_5-4B-Q4_K_M.gguf",
            "7c1612b6896ad14caa501238e72afa17a600651d0984225e3ff78b39de86099c",
            2716065472,
        ),
        "mmproj": (
            "mmproj-OpenGVLab_InternVL3_5-4B-bf16.gguf",
            "c697d9fb3090cde25cd3c24c1cfad6970ef823e6f3fd072174200034f667d143",
            646227360,
        ),
        "ollama_tag": "internvl3.5:4b-q4_k_m",
    },
    "internvl3.5_38b": {
        "repo": "mradermacher/InternVL3_5-38B-GGUF",
        "main": (
            "InternVL3_5-38B.Q4_K_M.gguf",
            "94f43c1e9f2cbd8d192a1b966ad7778e70b9981a5aecb1b4479af59586545774",
            19762146688,
        ),
        "mmproj": (
            "InternVL3_5-38B.mmproj-Q8_0.gguf",
            "e36451c93bf11103e3586632ec415d2f7ff68df8e49fa31d994832d263348ff2",
            6000453152,
        ),
        "ollama_tag": "internvl3.5:38b-q4_k_m",
    },
}

QWEN_TAGS = ["qwen3-vl:4b", "qwen3-vl:32b"]

# A billboard photo from the sibling TC-STR dataset -- not a document scan,
# just enough to prove the vision path is wired up (image in -> grounded text
# out, not a hallucinated stock answer).
SMOKE_TEST_IMAGE = pathlib.Path("/mnt/nvme/datasets/TC-STR/images/billboard_00000_010_雜貨舖.jpg")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: pathlib.Path, expected_sha256: str, expected_size: int) -> None:
    if target.is_file() and target.stat().st_size == expected_size and sha256_file(target) == expected_sha256:
        print(f"  already present, verified: {target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        digest = hashlib.sha256()
        written = 0
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=900) as response:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                print(f"\r  {target.name}: {written / 1e9:.2f} / {expected_size / 1e9:.2f} GB", end="", flush=True)
            out.flush()
            os.fsync(out.fileno())
        print()
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(f"SHA mismatch for {target.name}: {actual} != {expected_sha256}")
        if written != expected_size:
            raise RuntimeError(f"size mismatch for {target.name}: {written} != {expected_size}")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ollama_model_present(tag: str) -> bool:
    with urllib.request.urlopen(f"{OLLAMA_API}/tags", timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    return any(m["name"] == tag for m in data["models"])


def fetch_qwen(tag: str) -> None:
    print(f"[qwen3-vl] ollama pull {tag}")
    subprocess.run(["ollama", "pull", tag], check=True)


def fetch_internvl(key: str, spec: dict) -> None:
    tag = spec["ollama_tag"]
    if ollama_model_present(tag):
        print(f"[{key}] {tag} already exists in ollama, skipping import")
        return
    work_dir = SCRATCH_ROOT / key
    work_dir.mkdir(parents=True, exist_ok=True)
    main_name, main_sha, main_size = spec["main"]
    proj_name, proj_sha, proj_size = spec["mmproj"]
    base_url = f"https://huggingface.co/{spec['repo']}/resolve/main"
    print(f"[{key}] downloading {main_name} ({main_size / 1e9:.1f} GB)")
    download(f"{base_url}/{main_name}", work_dir / "main.gguf", main_sha, main_size)
    print(f"[{key}] downloading {proj_name} ({proj_size / 1e9:.1f} GB)")
    download(f"{base_url}/{proj_name}", work_dir / "mmproj.gguf", proj_sha, proj_size)
    modelfile = work_dir / "Modelfile"
    modelfile.write_text("FROM ./main.gguf\nFROM ./mmproj.gguf\n")
    print(f"[{key}] ollama create {tag}")
    subprocess.run(["ollama", "create", tag, "-f", str(modelfile)], check=True, cwd=work_dir)


def smoke_test(tag: str) -> None:
    if not SMOKE_TEST_IMAGE.is_file():
        print(f"  [skip] sample image not found: {SMOKE_TEST_IMAGE}")
        return
    import base64

    image_b64 = base64.b64encode(SMOKE_TEST_IMAGE.read_bytes()).decode("ascii")
    payload = json.dumps(
        {
            "model": tag,
            "prompt": "What text do you see in this image? List it exactly.",
            "images": [image_b64],
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(f"{OLLAMA_API}/generate", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
    print(f"  [{tag}] response: {result['response'].strip()!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["qwen3-vl:4b", "qwen3-vl:32b", "internvl3.5_4b", "internvl3.5_38b"],
        action="append",
        help="fetch only these (repeatable); default is all four",
    )
    parser.add_argument("--smoke-test", action="store_true", help="after fetching, run a real image through each model")
    args = parser.parse_args()
    selected = set(args.only) if args.only else {"qwen3-vl:4b", "qwen3-vl:32b", "internvl3.5_4b", "internvl3.5_38b"}

    for tag in QWEN_TAGS:
        if tag in selected:
            fetch_qwen(tag)
    for key, spec in INTERNVL_FILES.items():
        if key in selected:
            fetch_internvl(key, spec)

    if args.smoke_test:
        print("\n-- smoke test (real image -> model -> text) --")
        tags = [t for t in QWEN_TAGS if t in selected] + [
            spec["ollama_tag"] for key, spec in INTERNVL_FILES.items() if key in selected
        ]
        for tag in tags:
            smoke_test(tag)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
