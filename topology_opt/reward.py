"""Configurable reward calculation for measured distributed-learning runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    weights: dict[str, float]
    target_accuracy: float
    accuracy_gain_scale: float
    max_bytes_per_worker_round: float
    max_exchange_seconds_per_worker_round: float
    max_runtime_seconds_per_round: float

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RewardConfig":
        config_path = Path(path) if path else Path(__file__).with_name("reward_config.json")
        value = json.loads(config_path.read_text(encoding="utf-8"))
        weights = {str(key): float(item) for key, item in value["weights"].items()}
        expected = {
            "accuracy_gain",
            "convergence_speed",
            "communication_efficiency",
            "runtime_efficiency",
        }
        if set(weights) != expected or any(item < 0 for item in weights.values()):
            raise ValueError("Reward weights must define the four non-negative reward components")
        if abs(sum(weights.values()) - 1) > 1e-9:
            raise ValueError("Reward weights must sum to 1")
        config = cls(
            weights=weights,
            target_accuracy=float(value["target_accuracy"]),
            accuracy_gain_scale=float(value["accuracy_gain_scale"]),
            max_bytes_per_worker_round=float(value["max_bytes_per_worker_round"]),
            max_exchange_seconds_per_worker_round=float(
                value["max_exchange_seconds_per_worker_round"]
            ),
            max_runtime_seconds_per_round=float(value["max_runtime_seconds_per_round"]),
        )
        if not 0 < config.target_accuracy <= 1 or any(
            item <= 0
            for item in (
                config.accuracy_gain_scale,
                config.max_bytes_per_worker_round,
                config.max_exchange_seconds_per_worker_round,
                config.max_runtime_seconds_per_round,
            )
        ):
            raise ValueError("Reward targets and normalization scales must be positive")
        return config

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "target_accuracy": self.target_accuracy,
            "accuracy_gain_scale": self.accuracy_gain_scale,
            "max_bytes_per_worker_round": self.max_bytes_per_worker_round,
            "max_exchange_seconds_per_worker_round": self.max_exchange_seconds_per_worker_round,
            "max_runtime_seconds_per_round": self.max_runtime_seconds_per_round,
        }


def calculate_run_reward(
    rounds: list[dict[str, Any]],
    config: RewardConfig,
) -> dict[str, Any]:
    if not rounds:
        raise ValueError("A completed run must contain at least one measured round")
    accuracies = [
        sum(float(metrics["accuracy"]) for metrics in row["workers"].values())
        / len(row["workers"])
        for row in rounds
        if row.get("workers")
    ]
    if len(accuracies) != len(rounds):
        raise ValueError("Every round must contain worker accuracy metrics")
    initial_accuracy, final_accuracy = accuracies[0], accuracies[-1]
    accuracy_gain = min(max((final_accuracy - initial_accuracy) / config.accuracy_gain_scale, 0), 1)
    convergence_round = next(
        (index + 1 for index, accuracy in enumerate(accuracies) if accuracy >= config.target_accuracy),
        None,
    )
    convergence_speed = (
        0.0
        if convergence_round is None
        else (len(rounds) - convergence_round + 1) / len(rounds)
    )
    all_metrics = [
        metrics
        for row in rounds
        for metrics in row["workers"].values()
    ]
    worker_count = max(len(row["workers"]) for row in rounds)
    bytes_sent = sum(int(metrics.get("bytes_sent", 0)) for metrics in all_metrics)
    exchange_seconds = sum(float(metrics.get("exchange_seconds", 0)) for metrics in all_metrics)
    runtime_seconds = sum(
        max(
            float(metrics.get("round_seconds", metrics.get("runtime_seconds", 0)))
            for metrics in row["workers"].values()
        )
        for row in rounds
    )
    communication_efficiency = 1 - min(
        bytes_sent
        / (config.max_bytes_per_worker_round * len(rounds) * worker_count),
        1,
    )
    exchange_efficiency = 1 - min(
        exchange_seconds
        / (
            config.max_exchange_seconds_per_worker_round
            * len(rounds)
            * worker_count
        ),
        1,
    )
    communication_efficiency = (communication_efficiency + exchange_efficiency) / 2
    runtime_efficiency = 1 - min(
        runtime_seconds
        / (config.max_runtime_seconds_per_round * len(rounds)),
        1,
    )
    components = {
        "accuracy_gain": accuracy_gain,
        "convergence_speed": convergence_speed,
        "communication_efficiency": communication_efficiency,
        "runtime_efficiency": runtime_efficiency,
    }
    reward = sum(config.weights[name] * value for name, value in components.items())
    return {
        "reward": round(reward, 6),
        "components": {key: round(value, 6) for key, value in components.items()},
        "metrics": {
            "initial_accuracy": round(initial_accuracy, 6),
            "final_accuracy": round(final_accuracy, 6),
            "accuracy_gain": round(final_accuracy - initial_accuracy, 6),
            "target_accuracy": config.target_accuracy,
            "convergence_round": convergence_round,
            "bytes_sent": bytes_sent,
            "exchange_seconds": round(exchange_seconds, 3),
            "runtime_seconds": round(runtime_seconds, 3),
        },
        "weights": dict(config.weights),
    }
