"""Reward-based topology selection and simple measured-data baselines."""

from __future__ import annotations

import random
from typing import Any


def select_candidate(candidates: list[dict[str, Any]], seed: int = 2026) -> dict[str, Any]:
    valid = [candidate for candidate in candidates if candidate.get("completed_runs", 0) > 0]
    if not valid:
        raise ValueError("Topology selection requires at least one candidate with real completed runs")
    reward_choice = max(valid, key=lambda item: item["mean_reward"])
    best_observed = max(valid, key=lambda item: item["mean_final_accuracy"])
    random_choice = random.Random(seed).choice(valid)
    return {
        "method": "weighted_measured_reward",
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
