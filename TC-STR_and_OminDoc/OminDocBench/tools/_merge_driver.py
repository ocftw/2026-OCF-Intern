#!/usr/bin/env python3
"""Merge per-batch official results into one corpus-wide official result.

This script is never run on the host. It is bind-mounted into the exact
pinned OmniDocBench evaluator image and executed there (via
``--entrypoint python3``) so that every piece of scoring/aggregation math it
touches is the official, pinned implementation itself -- not a re-typed copy
of it. It only:

  1. Imports the official aggregation helpers directly from the image's own
     ``src`` package (unmodified).
  2. For Edit_dist, re-runs the real ``call_Edit_dist(...).evaluate()`` on the
     merged samples -- correct and cheap, since Levenshtein distance is fast.
  3. For TEDS/TEDS_structure_only/CDM, it does NOT re-score (that would
     re-render every table/formula from scratch); each sample already carries
     its official per-sample score from whichever batch computed it
     (batching never changes a per-sample score, only which docker
     invocation produced it). It reproduces only the official aggregation
     *tail* -- the same "collect scores into 'all' plus any configured
     group, then average" pattern used verbatim in cal_metric.py's
     call_TEDS/call_CDM -- using the official ``_safe_average`` and
     ``_sample_matches_group`` helpers imported unchanged from the image.
  4. Calls the official ``get_full_labels_results``/``get_page_split`` for
     the 'group'/'page' report sections, over the full merged sample list,
     exactly as a single non-batched run would.

Batching only changes how the corpus is split across docker invocations
during scoring; every score is produced by the same pinned code either way,
and this step performs the same one aggregation the official pipeline
would have performed if it had scored the whole corpus in one invocation.
"""

from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, "/workspace")

import yaml  # noqa: E402

from src.core.metrics import get_full_labels_results, get_page_split  # noqa: E402
from src.dataset.end2end_dataset import End2EndDataset  # noqa: E402
from src.metrics.cal_metric import (  # noqa: E402
    _safe_average,
    _sample_matches_group,
    call_Edit_dist,
)


def build_page_info(full_gt: list) -> dict:
    page_info = {}
    for page in full_gt:
        img_path = os.path.basename(page["page_info"]["image_path"])
        page_info[img_path] = page["page_info"]["page_attribute"]
    return page_info


def aggregate_precomputed_metric(samples: list, metric_names: list, group_info: list) -> dict:
    """Reproduce the official 'collect into all + groups, then average' tail
    for a metric whose per-sample score is already computed (TEDS/CDM), i.e.
    the part of call_TEDS/call_CDM that runs after per-sample scoring.
    """
    group_scores = {name: {"all": []} for name in metric_names}
    for sample in samples:
        metric_map = sample.get("metric") or {}
        for name in metric_names:
            if name in metric_map:
                group_scores[name]["all"].append(metric_map[name])
        for group in group_info:
            if _sample_matches_group(sample, group):
                for name in metric_names:
                    if name in metric_map:
                        group_scores[name].setdefault(str(group), []).append(metric_map[name])
    return {
        name: {bucket: _safe_average(scores) for bucket, scores in buckets.items()}
        for name, buckets in group_scores.items()
    }


def main() -> int:
    with open("/workspace/configs/run.yaml", "r", encoding="utf-8") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    task_cfg = cfg["end2end_eval"]
    metrics_list = task_cfg["metrics"]
    prediction_path = task_cfg["dataset"]["prediction"]["data_path"]
    match_method = task_cfg["dataset"].get("match_method", "quick_match")
    save_name = os.path.basename(prediction_path) + "_" + match_method

    with open("/workspace/gt/OmniDocBench.json", "r", encoding="utf-8") as handle:
        full_gt = json.load(handle)
    gt_pages_by_element = End2EndDataset._collect_gt_pages_by_element(None, full_gt)
    page_info = build_page_info(full_gt)

    result_all = {}
    os.makedirs("/workspace/result", exist_ok=True)
    for element, metric_cfg in metrics_list.items():
        merged_path = f"/workspace/merge_input/{element}_samples.json"
        with open(merged_path, "r", encoding="utf-8") as handle:
            samples = json.load(handle)

        group_info = metric_cfg.get("group", [])
        expected_metrics = list(metric_cfg.get("metric", []))
        result = {}
        for metric_name in expected_metrics:
            if metric_name == "Edit_dist":
                _, edit_result = call_Edit_dist(copy.deepcopy(samples), {}).evaluate(
                    group_info=group_info, save_name=f"{save_name}_{element}"
                )
                result.update(edit_result)
            elif metric_name == "TEDS":
                result.update(
                    aggregate_precomputed_metric(
                        samples, ["TEDS", "TEDS_structure_only"], group_info
                    )
                )
            elif metric_name not in result:
                result.update(aggregate_precomputed_metric(samples, [metric_name], group_info))

        group_result = get_full_labels_results(samples)
        page_result = get_page_split(
            samples,
            page_info,
            gt_page_names=gt_pages_by_element.get(element),
            expected_metrics=expected_metrics,
        )
        result_all[element] = {"all": result, "group": group_result, "page": page_result}

        with open(f"/workspace/result/{save_name}_{element}_result.json", "w", encoding="utf-8") as handle:
            json.dump(samples, handle, indent=4, ensure_ascii=False)

    with open(f"/workspace/result/{save_name}_metric_result.json", "w", encoding="utf-8") as handle:
        json.dump(result_all, handle, indent=4, ensure_ascii=False)

    print(f"[merge-driver] wrote {save_name}_metric_result.json and per-element result files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
