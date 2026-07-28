"""固定 seed 的 bootstrap CI 與 paired model deltas。"""

from __future__ import annotations

import itertools
import random
from typing import Any, Callable


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[min(len(ordered) - 1, int(probability * len(ordered)))]


def bootstrap_ci(
    values: list[float], iterations: int = 10000, seed: int = 2026
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(iterations)]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def stratified_macro_bootstrap(
    subject_values: dict[str, list[float]], iterations: int = 10000, seed: int = 2026
) -> tuple[float, float]:
    if not subject_values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        subject_scores = []
        for values in subject_values.values():
            draw = [rng.choice(values) for _ in values]
            subject_scores.append(sum(draw) / len(draw))
        estimates.append(sum(subject_scores) / len(subject_scores))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_deltas(
    by_model: dict[str, dict[str, float]],
    statistic: Callable[[list[float]], float] | None = None,
    iterations: int = 10000,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    statistic = statistic or (lambda values: sum(values) / len(values))
    rng = random.Random(seed)
    output = []
    for left, right in itertools.combinations(sorted(by_model), 2):
        common = sorted(set(by_model[left]) & set(by_model[right]))
        if not common:
            output.append({"model_a": left, "model_b": right, "status": "N/A"})
            continue
        deltas = [by_model[left][key] - by_model[right][key] for key in common]
        draws = [statistic([rng.choice(deltas) for _ in deltas]) for _ in range(iterations)]
        output.append(
            {
                "model_a": left,
                "model_b": right,
                "n": len(common),
                "delta": statistic(deltas),
                "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
            }
        )
    return output


def stratified_paired_delta(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    iterations: int = 10000,
    seed: int = 2026,
) -> dict[str, Any]:
    """subject -> sample_id -> score；subject 內 paired resample，再取 macro delta。"""
    rng = random.Random(seed)
    paired: dict[str, list[float]] = {}
    for subject in sorted(set(left) & set(right)):
        common = sorted(set(left[subject]) & set(right[subject]))
        if common:
            paired[subject] = [left[subject][key] - right[subject][key] for key in common]
    if not paired:
        return {"status": "N/A"}
    observed = sum(sum(values) / len(values) for values in paired.values()) / len(paired)
    draws = []
    for _ in range(iterations):
        subject_deltas = [
            sum(rng.choice(values) for _ in values) / len(values) for values in paired.values()
        ]
        draws.append(sum(subject_deltas) / len(subject_deltas))
    return {
        "delta": observed,
        "n": sum(len(values) for values in paired.values()),
        "subjects": len(paired),
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (len(pvalues) - rank) * value))
        adjusted[index] = running
    return adjusted
