"""Hyperparameter optimisation (HPO) candidates for a fixed dataset architecture.

HPO keeps the network fixed and varies only its training settings. Every worker
evaluates the same candidate list on its own data split (through the dataset NAS
search endpoint), and the coordinator picks the setting with the best mean
worker validation accuracy. No worker data or weights leave the workers.
"""

from __future__ import annotations

import math
from typing import Any

from common.model_design import validate_model_design

# Learning-rate multipliers around the architecture's current rate. A small,
# log-spaced grid is the most reliable use of a tight on-device budget.
LEARNING_RATE_FACTORS = (0.3, 1.0, 3.0)
BATCH_SIZES = (32, 64)
MNIST_MAX_CANDIDATES = 6
MEMORY_CONSTRAINED_MAX_CANDIDATES = 3
# Epochs each HPO candidate trains: enough to separate learning rates, and 6 x 2 epochs
# fits a 300 s budget on 8 MNIST workers; CIFAR-10 candidates are slower, so 1 epoch.
HPO_EPOCHS = 2
MEMORY_CONSTRAINED_HPO_EPOCHS = 1
MIN_LEARNING_RATE = 0.00001
MAX_LEARNING_RATE = 0.01
# Keys added by the coordinator for bookkeeping; they are not part of the design.
_BOOKKEEPING_KEYS = {
    "id",
    "parameter_count",
    "dataset_name",
    "input_shape",
    "classes",
    "num_classes",
    "normalization",
    "initialization_seed",
}


def _rate_key(learning_rate: float) -> str:
    """Stable, filename-safe learning-rate label such as 3e-4."""
    exponent = math.floor(math.log10(learning_rate))
    mantissa = round(learning_rate / 10**exponent)
    return f"{mantissa}e{exponent}"


def hpo_candidates(
    base_architecture: dict[str, Any],
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    memory_constrained: bool = False,
) -> list[dict[str, Any]]:
    """Return validated candidates that differ from the base only in training settings."""
    design = {key: value for key, value in base_architecture.items() if key not in _BOOKKEEPING_KEYS}
    normalized, parameter_count = validate_model_design(design, input_shape, num_classes)
    base_training = normalized["training"]
    base_rate = float(base_training["learning_rate"])
    rates = sorted(
        {
            round(min(MAX_LEARNING_RATE, max(MIN_LEARNING_RATE, base_rate * factor)), 8)
            for factor in LEARNING_RATE_FACTORS
        }
    )
    if memory_constrained:
        # CIFAR-10 on an 8 GB Jetson: keep the deployed batch size and only tune the rate.
        grid = [(rate, int(base_training["batch_size"])) for rate in rates]
        limit = MEMORY_CONSTRAINED_MAX_CANDIDATES
    else:
        grid = [(rate, batch) for rate in rates for batch in BATCH_SIZES]
        limit = MNIST_MAX_CANDIDATES
    candidates = []
    for rate, batch_size in grid[:limit]:
        training = {**base_training, "learning_rate": rate, "batch_size": batch_size}
        candidate = {
            **design,
            "training": training,
            "id": f"hpo-{base_training['optimizer']}-lr{_rate_key(rate)}-bs{batch_size}",
            "parameter_count": parameter_count,
        }
        # Validate again so a worker never receives a design it would reject.
        validate_model_design(candidate, input_shape, num_classes)
        candidates.append(candidate)
    return candidates


def select_best_hyperparameters(measured: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the candidate with the best mean worker accuracy, then macro-F1, then speed."""
    if not measured:
        raise ValueError("HPO produced no measured candidates")
    return min(
        measured,
        key=lambda item: (
            -float(item["validation_accuracy"]),
            -float(item["validation_macro_f1"]),
            float(item["elapsed_seconds"]),
            item["architecture"]["id"],
        ),
    )
