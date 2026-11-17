"""Decentralized model aggregation rules run on each worker (never on the coordinator).

Each worker combines its own locally trained weights with the weights received from its
in-neighbours in the current topology. All rules return weights with the reference dtype.

- ``fedavg``: weighted by local training-sample counts (the original behaviour).
- ``uniform``: every participating model counts equally.
- ``metropolis``: Metropolis-Hastings consensus weights, w_ij = 1 / (1 + max(d_i, d_j)) for
  each received model j, where d is a node's number of in-neighbours; the node keeps the
  remaining mass. These weights form a doubly stochastic mixing matrix on undirected graphs,
  which is the standard choice for decentralized averaging.
- ``trimmed_mean``: coordinate-wise mean after dropping the single smallest and largest
  value, robust to one faulty or outlying peer. Needs at least three models; with fewer it
  falls back to ``fedavg``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

AGGREGATION_RULES = ("fedavg", "uniform", "metropolis", "trimmed_mean")
SYNC_MODES = ("sync", "semi_sync")


@dataclass(frozen=True)
class Contribution:
    node_id: str
    weights: list[np.ndarray]
    sample_count: int
    in_degree: int


def model_signature(weights: list[np.ndarray]) -> str:
    """Stable fingerprint of a model's tensor layout (shapes and dtypes, not values)."""
    layout = ";".join(f"{tuple(item.shape)}:{np.asarray(item).dtype.str}" for item in weights)
    return hashlib.sha256(layout.encode("ascii")).hexdigest()[:16]


def _check_layout(contributions: list[Contribution]) -> list[np.ndarray]:
    if not contributions:
        raise ValueError("Aggregation requires at least the local model")
    reference = contributions[0].weights
    if not reference:
        raise ValueError("Model weights must not be empty")
    for contribution in contributions:
        if len(contribution.weights) != len(reference):
            raise ValueError(
                f"Model from {contribution.node_id} has {len(contribution.weights)} tensors; "
                f"expected {len(reference)} (architecture mismatch)"
            )
        for current, expected in zip(contribution.weights, reference):
            if current.shape != expected.shape:
                raise ValueError(
                    f"Model from {contribution.node_id} has tensor shape {current.shape}; "
                    f"expected {expected.shape} (architecture mismatch)"
                )
            if not np.isfinite(current).all():
                raise ValueError(f"Model from {contribution.node_id} contains non-finite values")
        if contribution.sample_count < 1:
            raise ValueError(f"Model from {contribution.node_id} reports no training samples")
    return reference


def mixing_weights(rule: str, contributions: list[Contribution]) -> list[float]:
    """Per-model coefficients (summing to 1) for the linear rules; local model first."""
    if rule == "fedavg" or (rule == "trimmed_mean" and len(contributions) < 3):
        total = sum(item.sample_count for item in contributions)
        return [item.sample_count / total for item in contributions]
    if rule == "uniform":
        return [1.0 / len(contributions)] * len(contributions)
    if rule == "metropolis":
        local = contributions[0]
        peer_weights = [
            1.0 / (1.0 + max(local.in_degree, peer.in_degree)) for peer in contributions[1:]
        ]
        self_weight = 1.0 - sum(peer_weights)
        if self_weight < 0:
            raise ValueError("Metropolis weights exceed one; in-degrees are inconsistent")
        return [self_weight, *peer_weights]
    raise ValueError(f"Unsupported aggregation rule {rule!r}")


def aggregate(rule: str, contributions: list[Contribution]) -> list[np.ndarray]:
    """Combine the local model (first) with received peer models using ``rule``."""
    if rule not in AGGREGATION_RULES:
        raise ValueError(f"Unsupported aggregation rule {rule!r}")
    reference = _check_layout(contributions)
    if rule == "trimmed_mean" and len(contributions) >= 3:
        aggregated = []
        for index in range(len(reference)):
            stacked = np.stack([item.weights[index].astype(np.float64) for item in contributions])
            ordered = np.sort(stacked, axis=0)
            aggregated.append(ordered[1:-1].mean(axis=0).astype(reference[index].dtype))
        return aggregated
    coefficients = mixing_weights(rule, contributions)
    return [
        sum(
            item.weights[index].astype(np.float64) * coefficient
            for item, coefficient in zip(contributions, coefficients)
        ).astype(reference[index].dtype)
        for index in range(len(reference))
    ]
