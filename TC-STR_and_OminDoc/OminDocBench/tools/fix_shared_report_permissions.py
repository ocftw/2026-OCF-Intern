#!/usr/bin/env python3
"""Grant other-read on exactly the files build_comparison_report.py (and
build_visual_report.py) need to read, so users who are NOT this box's
dataset/output owner -- but who can already traverse the world-readable
directory chain under paths.dataset_root / paths.output_root -- can generate
reports too, without ever needing this user's login.

This never copies or moves data (see README's "共用資料維護" section for why:
the system disk backing /opt and /home has very little free space, and the
dataset/output trees already live on /mnt/nvme with plenty of room -- only
their file permissions are the blocker). It only adds the "other" read bit
(o+r) on:

  - the ground-truth JSON (paths.dataset_root/<ground_truth_filename>)
  - every file under paths.dataset_root/images/
  - every file under paths.output_root/v1_6_*/models/*/predictions/

Directories in that chain are already world-traversable (o+rx) on this box;
this script never touches directory permissions, and never touches any file
outside the three groups above (raw_responses.jsonl, docker logs, batch
internals, etc. are left exactly as they are -- least exposure).

Safe to re-run any time (idempotent) -- e.g. after a new model finishes
evaluation, so its predictions become readable too.
"""

from __future__ import annotations

import pathlib
import stat
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omnidocbench.core import Settings  # noqa: E402


def add_other_read(path: pathlib.Path) -> bool:
    """chmod o+r on a single file. Returns True if a change was actually made."""
    current = stat.S_IMODE(path.stat().st_mode)
    if current & stat.S_IROTH:
        return False
    path.chmod(current | stat.S_IROTH)
    return True


def fix_files(paths: list[pathlib.Path], label: str) -> None:
    changed = 0
    for p in paths:
        if not p.is_file():
            continue
        if add_other_read(p):
            changed += 1
    print(f"{label}: {changed} file(s) changed, {len(paths)} checked")


def main() -> int:
    settings = Settings.load()
    output_root = pathlib.Path(settings.raw["paths"]["output_root"])

    fix_files([settings.gt_path], "ground truth JSON")
    fix_files(sorted(settings.image_dir.glob("*")), "dataset images")

    prediction_files: list[pathlib.Path] = []
    for run_dir in sorted(output_root.glob("v1_6_*")):
        models_root = run_dir / "models"
        if not models_root.is_dir():
            continue
        for model_dir in sorted(models_root.iterdir()):
            pred_dir = model_dir / "predictions"
            if pred_dir.is_dir():
                prediction_files.extend(sorted(pred_dir.glob("*")))
    fix_files(prediction_files, "model prediction files (all run ids)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
