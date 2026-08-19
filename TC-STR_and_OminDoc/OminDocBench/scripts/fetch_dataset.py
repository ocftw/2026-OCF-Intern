#!/usr/bin/env python3
"""Download the exact pinned v1.6 GT and page images with resume and SHA checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from omnidocbench.core import Settings, atomic_write, sha256_file


def api_json(url: str) -> tuple[object, dict[str, str]]:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8")), dict(response.headers)


def list_images(repo: str, revision: str) -> list[dict]:
    base = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/images"
    url = f"{base}?recursive=false&expand=false&limit=1000"
    files: list[dict] = []
    while url:
        page, headers = api_json(url)
        files.extend(x for x in page if x.get("type") == "file")
        link = next((value for key, value in headers.items() if key.lower() == "link"), "")
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part.split(";", 1)[0].strip().strip("<>")
        url = next_url
    return files


def download(url: str, target: pathlib.Path, expected_sha256: str | None = None) -> None:
    if target.is_file() and (expected_sha256 is None or sha256_file(target) == expected_sha256):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=900) as response:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                digest.update(chunk)
            out.flush()
            os.fsync(out.fileno())
        actual = digest.hexdigest()
        if expected_sha256 and actual != expected_sha256:
            raise RuntimeError(f"SHA mismatch for {target.name}: {actual} != {expected_sha256}")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    settings = Settings.load()
    benchmark = settings.raw["benchmark"]
    repo, revision = benchmark["dataset_repo"], benchmark["dataset_revision"]
    root = pathlib.Path(settings.raw["paths"]["dataset_root"])
    files = list_images(repo, revision)
    if len(files) != benchmark["expected_pages"]:
        raise RuntimeError(f"pinned image tree has {len(files)} files, expected {benchmark['expected_pages']}")
    manifest = {
        "repo": repo,
        "revision": revision,
        "files": [
            {
                "path": x["path"],
                "size": x["size"],
                "sha256": (x.get("lfs") or {}).get("oid"),
            }
            for x in files
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write(root / "huggingface_revision_manifest.json", json.dumps(manifest, indent=2) + "\n")
    if args.manifest_only:
        print(root / "huggingface_revision_manifest.json")
        return 0
    gt_url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{benchmark['ground_truth_filename']}"
    download(gt_url, settings.gt_path, benchmark["ground_truth_sha256"])
    for index, item in enumerate(manifest["files"], 1):
        encoded = urllib.parse.quote(item["path"], safe="/")
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{encoded}"
        target = root / item["path"]
        download(url, target, item["sha256"])
        print(f"[{index}/{len(files)}] {item['path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
