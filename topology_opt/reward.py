"""Configurable reward calculation for measured distributed-learning runs.

Version 1 (historical) rewards accuracy gain, a 50% convergence target and
cost terms normalized by fixed caps. Its accuracy gain saturates at 1.0 for
every topology, so the cheapest topology won even with a 50%-accurate worker.

Version 2 rewards what decentralized learning is for, every worker ending with
a model that is good on all classes:

    reward = w_q * Q + w_s * S + w_b * B + w_w * W

    q_r = (1 - k) * mean_f1_r + k * min_f1_r     per round r (k = worst_worker_share)
    Q   = q_R                                     quality after the last round
    S   = mean(q_1 .. q_R)                        learning speed (area under the curve)
    B   = 1 - bytes_sent / (model_bytes * N * (N - 1) * R)   0 = full mesh
    W   = 1 - sum(peer_wait) / sum(round_time)    share of time not spent waiting

F1 is measured on the shared class-balanced validation split when workers
report it, otherwise on each worker's local split over the classes it holds.
A run passes the quality gate when its weakest worker reaches the configured
accuracy; selection prefers candidates that pass it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


V2_COMPONENTS = {"quality", "learning_speed", "bandwidth_efficiency", "wait_efficiency"}


@dataclass(frozen=True)
class RewardConfig:
    weights: dict[str, float]
    target_accuracy: float = 0.5
    accuracy_gain_scale: float = 0.25
    max_bytes_per_worker_round: float = 50_000_000
    max_exchange_seconds_per_worker_round: float = 30.0
    max_peer_wait_seconds_per_worker_round: float = 30.0
    version: int = 1
    worst_worker_share: float = 0.5
    quality_gate_min_worker_accuracy: float = 0.9

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RewardConfig":
        config_path = Path(path) if path else Path(__file__).with_name("reward_config.json")
        return cls.from_dict(json.loads(config_path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RewardConfig":
        """Config from JSON; groups store theirs so they are always re-scored the same way."""
        weights = {str(key): float(item) for key, item in value["weights"].items()}
        if int(value.get("version", 1)) == 2:
            if set(weights) != V2_COMPONENTS or any(item < 0 for item in weights.values()):
                raise ValueError("Reward v2 weights must define the four non-negative v2 components")
            if abs(sum(weights.values()) - 1) > 1e-9:
                raise ValueError("Reward weights must sum to 1")
            config = cls(
                weights=weights,
                version=2,
                worst_worker_share=float(value.get("worst_worker_share", 0.5)),
                quality_gate_min_worker_accuracy=float(value.get("quality_gate_min_worker_accuracy", 0.9)),
            )
            if not 0 <= config.worst_worker_share <= 1 or not 0 <= config.quality_gate_min_worker_accuracy <= 1:
                raise ValueError("worst_worker_share and the quality gate must be within [0, 1]")
            return config
        expected = {
            "accuracy_gain",
            "convergence_speed",
            "communication_efficiency",
            "peer_wait_efficiency",
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
            max_peer_wait_seconds_per_worker_round=float(
                value["max_peer_wait_seconds_per_worker_round"]
            ),
        )
        if not 0 < config.target_accuracy <= 1 or any(
            item <= 0
            for item in (
                config.accuracy_gain_scale,
                config.max_bytes_per_worker_round,
                config.max_exchange_seconds_per_worker_round,
                config.max_peer_wait_seconds_per_worker_round,
            )
        ):
            raise ValueError("Reward targets and normalization scales must be positive")
        return config

    def as_dict(self) -> dict[str, Any]:
        if self.version == 2:
            return {
                "version": 2,
                "weights": dict(self.weights),
                "worst_worker_share": self.worst_worker_share,
                "quality_gate_min_worker_accuracy": self.quality_gate_min_worker_accuracy,
            }
        return {
            "weights": dict(self.weights),
            "target_accuracy": self.target_accuracy,
            "accuracy_gain_scale": self.accuracy_gain_scale,
            "max_bytes_per_worker_round": self.max_bytes_per_worker_round,
            "max_exchange_seconds_per_worker_round": self.max_exchange_seconds_per_worker_round,
            "max_peer_wait_seconds_per_worker_round": self.max_peer_wait_seconds_per_worker_round,
        }


def calculate_run_reward(
    rounds: list[dict[str, Any]],
    config: RewardConfig,
) -> dict[str, Any]:
    if config.version == 2:
        return _calculate_run_reward_v2(rounds, config)
    if not rounds:
        raise ValueError("A completed run must contain at least one measured round")
    accuracies = [
        sum(
            float(metrics.get("validation_accuracy", metrics.get("accuracy")))
            for metrics in row["workers"].values()
        )
        / len(row["workers"])
        for row in rounds
        if row.get("workers")
    ]
    if len(accuracies) != len(rounds):
        raise ValueError("Every round must contain worker accuracy metrics")
    first_round_workers = rounds[0]["workers"]
    initial_validation_available = all(
        metrics.get("initial_validation_accuracy") is not None
        for metrics in first_round_workers.values()
    )
    initial_accuracy = (
        sum(float(metrics["initial_validation_accuracy"]) for metrics in first_round_workers.values())
        / len(first_round_workers)
        if initial_validation_available
        else accuracies[0]
    )
    final_accuracy = accuracies[-1]
    accuracy_gain_available = len(rounds) > 1 or initial_validation_available
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
    peer_wait_seconds = sum(
        float(metrics.get("peer_wait_seconds", 0))
        for row in rounds
        for metrics in row["workers"].values()
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
    peer_wait_efficiency = 1 - min(
        peer_wait_seconds
        / (
            config.max_peer_wait_seconds_per_worker_round
            * len(rounds)
            * worker_count
        ),
        1,
    )
    components = {
        "accuracy_gain": accuracy_gain,
        "convergence_speed": convergence_speed,
        "communication_efficiency": communication_efficiency,
        "peer_wait_efficiency": peer_wait_efficiency,
    }
    reward = sum(config.weights[name] * value for name, value in components.items())
    return {
        "reward": round(reward, 6),
        "components": {key: round(value, 6) for key, value in components.items()},
        "metrics": {
            "initial_accuracy": round(initial_accuracy, 6),
            "final_accuracy": round(final_accuracy, 6),
            "accuracy_gain": round(final_accuracy - initial_accuracy, 6),
            "accuracy_gain_available": accuracy_gain_available,
            "target_accuracy": config.target_accuracy,
            "convergence_round": convergence_round,
            "bytes_sent": bytes_sent,
            "exchange_seconds": round(exchange_seconds, 3),
            "runtime_seconds": round(runtime_seconds, 3),
            "peer_wait_seconds": round(peer_wait_seconds, 3),
        },
        "weights": dict(config.weights),
    }


def worker_quality(metrics: dict[str, Any]) -> tuple[float, float, str]:
    """(macro-F1, accuracy, source) of one worker's round, preferring the shared split."""
    if metrics.get("global_validation_macro_f1") is not None:
        return (
            float(metrics["global_validation_macro_f1"]),
            float(metrics["global_validation_accuracy"]),
            "global_validation",
        )
    accuracy = float(metrics.get("validation_accuracy", metrics.get("accuracy", 0.0)))
    per_class = metrics.get("validation_per_class_f1")
    confusion = metrics.get("validation_confusion_matrix_counts")
    if per_class is not None and confusion is not None:
        values = list(per_class.values()) if isinstance(per_class, dict) else list(per_class)
        present = [float(value) for value, row in zip(values, confusion) if sum(row) > 0]
        if present:
            # Classes absent from this worker's split would otherwise count as F1 = 0.
            return sum(present) / len(present), accuracy, "local_present_classes"
    return float(metrics.get("validation_macro_f1", metrics.get("macro_f1", accuracy))), accuracy, "local"


def _calculate_run_reward_v2(rounds: list[dict[str, Any]], config: RewardConfig) -> dict[str, Any]:
    measured = [row for row in rounds if row.get("workers")]
    if not measured or len(measured) != len(rounds):
        raise ValueError("Every round must contain worker metrics")
    share = config.worst_worker_share
    round_quality = []
    sources: set[str] = set()
    for row in measured:
        scores = [worker_quality(metrics) for metrics in row["workers"].values()]
        sources.update(source for _, _, source in scores)
        f1_values = [f1 for f1, _, _ in scores]
        round_quality.append((1 - share) * (sum(f1_values) / len(f1_values)) + share * min(f1_values))
    final_scores = [worker_quality(metrics) for metrics in measured[-1]["workers"].values()]
    final_accuracies = [accuracy for _, accuracy, _ in final_scores]
    quality = round_quality[-1]
    learning_speed = sum(round_quality) / len(round_quality)

    all_metrics = [metrics for row in measured for metrics in row["workers"].values()]
    worker_count = max(len(row["workers"]) for row in measured)
    round_count = len(measured)
    bytes_sent = sum(int(metrics.get("bytes_sent", 0)) for metrics in all_metrics)
    model_bytes = [float(metrics["weight_bytes"]) for metrics in all_metrics if metrics.get("weight_bytes")]
    mesh_bytes = (
        (sum(model_bytes) / len(model_bytes)) * worker_count * (worker_count - 1) * round_count
        if model_bytes and worker_count > 1
        else 0.0
    )
    bandwidth_efficiency = 1 - min(bytes_sent / mesh_bytes, 1) if mesh_bytes > 0 else 0.0
    peer_wait_seconds = sum(float(metrics.get("peer_wait_seconds", 0)) for metrics in all_metrics)
    round_seconds = sum(
        float(metrics.get("round_seconds", metrics.get("runtime_seconds", 0))) for metrics in all_metrics
    )
    wait_efficiency = 1 - min(peer_wait_seconds / round_seconds, 1) if round_seconds > 0 else 1.0

    components = {
        "quality": quality,
        "learning_speed": learning_speed,
        "bandwidth_efficiency": bandwidth_efficiency,
        "wait_efficiency": wait_efficiency,
    }
    reward = sum(config.weights[name] * value for name, value in components.items())
    first_accuracy = [accuracy for _, accuracy, _ in (worker_quality(m) for m in measured[0]["workers"].values())]
    final_accuracy = sum(final_accuracies) / len(final_accuracies)
    min_worker_accuracy = min(final_accuracies)
    return {
        "reward": round(reward, 6),
        "components": {key: round(value, 6) for key, value in components.items()},
        "metrics": {
            "quality_source": "global_validation" if sources == {"global_validation"} else "+".join(sorted(sources)),
            "initial_accuracy": round(sum(first_accuracy) / len(first_accuracy), 6),
            "final_accuracy": round(final_accuracy, 6),
            "min_worker_accuracy": round(min_worker_accuracy, 6),
            "final_mean_f1": round(sum(f1 for f1, _, _ in final_scores) / len(final_scores), 6),
            "final_min_f1": round(min(f1 for f1, _, _ in final_scores), 6),
            "round_quality": [round(value, 6) for value in round_quality],
            "accuracy_gain_available": True,
            "passes_quality_gate": min_worker_accuracy >= config.quality_gate_min_worker_accuracy,
            "quality_gate_min_worker_accuracy": config.quality_gate_min_worker_accuracy,
            "bytes_sent": bytes_sent,
            "mesh_equivalent_bytes": int(mesh_bytes),
            "peer_wait_seconds": round(peer_wait_seconds, 3),
            "round_seconds": round(round_seconds, 3),
        },
        "weights": dict(config.weights),
    }
