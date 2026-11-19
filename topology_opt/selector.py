"""Reward-based topology selection and simple measured-data baselines."""

from __future__ import annotations

import random
from typing import Any

# When no topology reaches the absolute quality gate (it is calibrated for MNIST; a
# short CIFAR-10 sweep reaches ~0.55), keep only topologies whose weakest worker is
# within this margin of the best weakest worker, so bandwidth cannot outvote quality.
RELATIVE_QUALITY_MARGIN = 0.05


def select_candidate(candidates: list[dict[str, Any]], seed: int = 2026) -> dict[str, Any]:
    valid = [candidate for candidate in candidates if candidate.get("completed_runs", 0) > 0]
    if not valid:
        raise ValueError("Topology selection requires at least one candidate with real completed runs")
    method = "weighted_measured_reward"
    quality_gate_met = None
    if all(candidate.get("reward_version") == 2 for candidate in valid):
        # Reward v2: only topologies whose weakest worker reached the accuracy gate,
        # unless none did (then a gate relative to the best weakest worker); within
        # one standard deviation of the best, the cheaper wins.
        passing = [candidate for candidate in valid if candidate.get("passes_quality_gate")]
        quality_gate_met = bool(passing)
        pool = passing or _relative_quality_pool(valid)
        best = max(pool, key=lambda item: item["mean_reward"])
        tolerance = float(best.get("std_reward") or 0.0)
        tied = [item for item in pool if best["mean_reward"] - item["mean_reward"] <= tolerance]
        reward_choice = max(
            tied,
            key=lambda item: (
                item.get("mean_components", {}).get("bandwidth_efficiency", 0.0),
                item["mean_reward"],
            ),
        )
        method = "quality_gated_reward_v2"
    else:
        reward_choice = max(valid, key=lambda item: item["mean_reward"])
    if quality_gate_met is False and len(pool) < len(valid):
        method = "relative_quality_gated_reward_v2"
    best_observed = max(valid, key=lambda item: item["mean_final_accuracy"])
    random_choice = random.Random(seed).choice(valid)
    return {
        "method": method,
        "quality_gate_met": quality_gate_met,
        "selected_candidate": {
            "topology": reward_choice["topology"],
            "alpha": reward_choice["alpha"],
            "mean_reward": reward_choice["mean_reward"],
        },
        "baselines": {
            "best_observed": {
                "topology": best_observed["topology"],
                "alpha": best_observed["alpha"],
                "mean_final_accuracy": best_observed["mean_final_accuracy"],
            },
            "random_seeded": {
                "topology": random_choice["topology"],
                "alpha": random_choice["alpha"],
                "mean_reward": random_choice["mean_reward"],
            },
        },
        "candidate_count": len(valid),
    }


def _relative_quality_pool(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured = [
        item for item in candidates if isinstance(item.get("mean_min_worker_accuracy"), (int, float))
    ]
    if len(measured) != len(candidates):
        return candidates
    best = max(float(item["mean_min_worker_accuracy"]) for item in measured)
    return [
        item for item in measured if float(item["mean_min_worker_accuracy"]) >= best - RELATIVE_QUALITY_MARGIN
    ]
