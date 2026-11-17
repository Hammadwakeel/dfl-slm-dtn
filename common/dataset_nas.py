"""Bounded architecture generation and Pareto selection for dataset NAS."""

from __future__ import annotations

import random
from typing import Any

from common.model_design import validate_model_design

MAX_DATASET_NAS_PARAMETERS = 1_000_000
INITIAL_POPULATION_SIZE = 8
CIFAR_INITIAL_POPULATION_SIZE = 3
FINAL_POPULATION_SIZE = 4
INITIAL_EPOCHS = 3
FINAL_EPOCHS = 10
MAX_BLOCKS = 3
CHANNEL_CHOICES = (16, 32, 64)
KERNEL_CHOICES = (3, 5)
HEAD_CHOICES = (0, 64)
DROPOUT_CHOICES = (0.0, 0.2, 0.35)
BLOCK_CHOICES = ("conv2d", "separable_conv2d")


def _candidate_id(
    channels: tuple[int, ...],
    kernel_size: int,
    block_type: str,
    head_units: int,
    dropout: float,
    attention: bool,
) -> str:
    width_key = "x".join(str(width) for width in channels)
    dropout_key = str(int(dropout * 100))
    candidate_id = (
        f"nas-d{len(channels)}-w{width_key}-k{kernel_size}-"
        f"{'sep' if block_type == 'separable_conv2d' else 'conv'}-"
        f"h{head_units}-drop{dropout_key}"
    )
    return f"{candidate_id}-attn" if attention else candidate_id


def build_dataset_architecture(
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    channels: tuple[int, ...],
    kernel_size: int,
    block_type: str,
    head_units: int,
    dropout: float,
    attention: bool = False,
) -> dict[str, Any]:
    if not 2 <= len(channels) <= MAX_BLOCKS:
        raise ValueError(f"Dataset NAS depth must be between 2 and {MAX_BLOCKS}")
    if any(width not in CHANNEL_CHOICES for width in channels):
        raise ValueError("Dataset NAS channel widths are not in the supported search space")
    if kernel_size not in KERNEL_CHOICES or block_type not in BLOCK_CHOICES:
        raise ValueError("Dataset NAS kernel or block type is unsupported")
    if head_units not in HEAD_CHOICES or dropout not in DROPOUT_CHOICES:
        raise ValueError("Dataset NAS classifier or dropout choice is unsupported")
    if not isinstance(attention, bool):
        raise ValueError("Dataset NAS attention choice must be a boolean")

    layers: list[dict[str, Any]] = []
    for width in channels:
        layers.append(
            {
                "type": block_type,
                "filters": width,
                "kernel_size": kernel_size,
                "strides": 1,
                "padding": "same",
                "activation": "relu",
            }
        )
        layers.append({"type": "batch_normalization"})
        layers.append({"type": "max_pooling2d", "pool_size": 2})
    if attention:
        layers.extend(
            [
                {"type": "patch_embedding", "patch_size": 1, "embed_dim": 32},
                {
                    "type": "transformer_encoder",
                    "num_heads": 4,
                    "ff_dim": 64,
                    "dropout": 0.1,
                    "activation": "gelu",
                },
                {"type": "global_average_pooling1d"},
            ]
        )
    else:
        layers.append({"type": "global_average_pooling2d"})
    if head_units:
        layers.extend(
            [
                {"type": "dense", "units": head_units, "activation": "gelu"},
                {"type": "dropout", "rate": dropout},
            ]
        )
    layers.append(
        {"type": "dense", "units": num_classes, "activation": "linear"}
    )
    architecture: dict[str, Any] = {
        "id": _candidate_id(
            channels,
            kernel_size,
            block_type,
            head_units,
            dropout,
            attention,
        ),
        "model_type": "custom",
        "blocks": len(channels),
        "channels": max(channels),
        "channel_schedule": list(channels),
        "kernel_size": kernel_size,
        "block_type": block_type,
        "head_units": head_units,
        "dropout": dropout,
        "attention": attention,
        "layers": layers,
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 32,
        },
    }
    _, parameter_count = validate_model_design(
        architecture,
        input_shape,
        num_classes,
    )
    if parameter_count > MAX_DATASET_NAS_PARAMETERS:
        raise ValueError(
            f"Dataset NAS candidate has {parameter_count:,} parameters; "
            f"cap is {MAX_DATASET_NAS_PARAMETERS:,}"
        )
    architecture["parameter_count"] = parameter_count
    return architecture


def _random_candidate(
    rng: random.Random,
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    memory_constrained: bool = False,
) -> dict[str, Any]:
    channel_choices = (16, 32) if memory_constrained else CHANNEL_CHOICES
    depth = 2 if memory_constrained else rng.randint(2, MAX_BLOCKS)
    channels = tuple(
        sorted(rng.choices(channel_choices, k=depth))
    )
    return build_dataset_architecture(
        input_shape=input_shape,
        num_classes=num_classes,
        channels=channels,
        kernel_size=3 if memory_constrained else rng.choice(KERNEL_CHOICES),
        block_type=rng.choice(BLOCK_CHOICES),
        head_units=rng.choice((0, 64) if memory_constrained else HEAD_CHOICES),
        dropout=rng.choice((0.0, 0.2) if memory_constrained else DROPOUT_CHOICES),
        attention=False if memory_constrained else rng.random() < 0.2,
    )


def initial_population(
    *,
    seed: int,
    input_shape: tuple[int, int, int],
    num_classes: int,
    size: int = INITIAL_POPULATION_SIZE,
    memory_constrained: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    population: dict[str, dict[str, Any]] = {}
    if size >= 2:
        fixed_candidates = (
            (
                ((16, 16), "conv2d", 0, 0.0, False),
                ((16, 32), "separable_conv2d", 64, 0.2, False),
            )
            if memory_constrained
            else (
                ((32, 64), "conv2d", 64, 0.2, False),
                ((32, 64), "conv2d", 64, 0.2, True),
            )
        )
        for channels, block_type, head_units, dropout, attention in fixed_candidates:
            candidate = build_dataset_architecture(
                input_shape=input_shape,
                num_classes=num_classes,
                channels=channels,
                kernel_size=3,
                block_type=block_type,
                head_units=head_units,
                dropout=dropout,
                attention=attention,
            )
            population[candidate["id"]] = candidate
    for _ in range(size * 100):
        if len(population) >= size:
            break
        try:
            candidate = _random_candidate(
                rng,
                input_shape=input_shape,
                num_classes=num_classes,
                memory_constrained=memory_constrained,
            )
        except ValueError:
            continue
        population.setdefault(candidate["id"], candidate)
        if len(population) == size:
            break
    if len(population) != size:
        raise ValueError("Could not generate enough unique feasible NAS architectures")
    return list(population.values())


def smoke_test_architecture(
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> dict[str, Any]:
    return build_dataset_architecture(
        input_shape=input_shape,
        num_classes=num_classes,
        channels=(16, 16),
        kernel_size=3,
        block_type="conv2d",
        head_units=0,
        dropout=0.0,
        attention=False,
    )


def mutate_population(
    parents: list[dict[str, Any]],
    *,
    seed: int,
    input_shape: tuple[int, int, int],
    num_classes: int,
    size: int = FINAL_POPULATION_SIZE,
    memory_constrained: bool = False,
) -> list[dict[str, Any]]:
    if not parents:
        raise ValueError("Evolutionary NAS requires at least one parent")
    rng = random.Random(seed)
    population = {parent["id"]: parent for parent in parents}
    for _ in range(size * 100):
        parent = rng.choice(parents)
        channels = list(
            int(layer["filters"])
            for layer in parent["layers"]
            if layer["type"] in BLOCK_CHOICES
        )
        kernel_size = int(parent["kernel_size"])
        block_type = str(parent["block_type"])
        head_units = int(parent["head_units"])
        dropout = float(parent["dropout"])
        attention = bool(parent.get("attention", False))
        if memory_constrained:
            mutation = rng.randrange(3)
            if mutation == 0:
                index = rng.randrange(len(channels))
                channels[index] = rng.choice((16, 32))
            elif mutation == 1:
                block_type = (
                    "separable_conv2d"
                    if block_type == "conv2d"
                    else "conv2d"
                )
            else:
                head_units = 64 if head_units == 0 else 0
                dropout = 0.2 if head_units else 0.0
            try:
                candidate = build_dataset_architecture(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    channels=tuple(sorted(channels)),
                    kernel_size=3,
                    block_type=block_type,
                    head_units=head_units,
                    dropout=dropout,
                    attention=False,
                )
            except ValueError:
                continue
            population.setdefault(candidate["id"], candidate)
            if len(population) >= len(parents) + size:
                break
            continue
        mutation = rng.randrange(6)
        if mutation == 0:
            if rng.random() < 0.5 and len(channels) > 2:
                channels.pop(rng.randrange(len(channels)))
            elif len(channels) < MAX_BLOCKS:
                channels.append(rng.choice(CHANNEL_CHOICES))
                channels.sort()
        elif mutation == 1:
            index = rng.randrange(len(channels))
            channels[index] = rng.choice(CHANNEL_CHOICES)
            channels.sort()
        elif mutation == 2:
            kernel_size = rng.choice(KERNEL_CHOICES)
        elif mutation == 3:
            block_type = rng.choice(BLOCK_CHOICES)
        else:
            if mutation == 4:
                head_units = rng.choice(HEAD_CHOICES)
                dropout = rng.choice(DROPOUT_CHOICES)
            else:
                attention = not attention
        try:
            candidate = build_dataset_architecture(
                input_shape=input_shape,
                num_classes=num_classes,
                channels=tuple(channels),
                kernel_size=kernel_size,
                block_type=block_type,
                head_units=head_units,
                dropout=dropout,
                attention=attention,
            )
        except ValueError:
            continue
        population.setdefault(candidate["id"], candidate)
        if len(population) >= len(parents) + size:
            break
    offspring = [
        candidate
        for candidate in population.values()
        if candidate["id"] not in {parent["id"] for parent in parents}
    ][:size]
    if len(offspring) != size:
        raise ValueError("Could not generate enough unique feasible NAS offspring")
    return [*parents, *offspring]


def aggregate_worker_candidates(
    worker_results: list[dict[str, Any]],
    *,
    worker_count: int,
) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for result in worker_results:
        for candidate in result.get("candidates", []):
            by_id.setdefault(candidate["architecture"]["id"], []).append(candidate)
    aggregates = []
    for candidate_id, rows in sorted(by_id.items()):
        worker_ids = [str(row["node_id"]) for row in rows]
        if len(rows) != worker_count or len(set(worker_ids)) != worker_count:
            raise ValueError(
                f"NAS candidate {candidate_id} is missing results from a worker"
            )
        metrics = {}
        for name in (
            "validation_accuracy",
            "validation_macro_f1",
            "validation_macro_precision",
            "validation_macro_recall",
            "elapsed_seconds",
        ):
            metrics[name] = sum(float(row[name]) for row in rows) / len(rows)
        metrics["peak_rss_mb"] = max(float(row["peak_rss_mb"]) for row in rows)
        architecture = rows[0]["architecture"]
        if any(row["architecture"] != architecture for row in rows[1:]):
            raise ValueError(f"NAS workers returned inconsistent architecture {candidate_id}")
        model_summaries = [row.get("model_summary") for row in rows]
        if any(summary != model_summaries[0] for summary in model_summaries[1:]):
            raise ValueError(f"NAS workers returned inconsistent model summaries for {candidate_id}")
        confusion_matrices = [
            row.get("validation_confusion_matrix") for row in rows
        ]
        aggregated_confusion_matrix = None
        if all(matrix is not None for matrix in confusion_matrices):
            matrix_size = len(confusion_matrices[0])
            if any(
                not isinstance(matrix, list)
                or len(matrix) != matrix_size
                or any(not isinstance(row, list) or len(row) != matrix_size for row in matrix)
                for matrix in confusion_matrices
            ):
                raise ValueError(
                    f"NAS workers returned inconsistent confusion matrices for {candidate_id}"
                )
            aggregated_confusion_matrix = [
                [
                    sum(int(matrix[row][column]) for matrix in confusion_matrices)
                    for column in range(matrix_size)
                ]
                for row in range(matrix_size)
            ]
        aggregates.append(
            {
                "architecture": architecture,
                "parameter_count": int(rows[0]["parameter_count"]),
                "worker_accuracies": [
                    float(row["validation_accuracy"])
                    for row in sorted(rows, key=lambda row: row["node_id"])
                ],
                "mean_validation_accuracy": metrics["validation_accuracy"],
                "mean_validation_macro_precision": metrics["validation_macro_precision"],
                "mean_validation_macro_recall": metrics["validation_macro_recall"],
                "mean_validation_macro_f1": metrics["validation_macro_f1"],
                "mean_elapsed_seconds": metrics["elapsed_seconds"],
                "max_peak_rss_mb": metrics["peak_rss_mb"],
                **(
                    {"model_summary": model_summaries[0]}
                    if isinstance(model_summaries[0], str) and model_summaries[0]
                    else {}
                ),
                **(
                    {"validation_confusion_matrix": aggregated_confusion_matrix}
                    if aggregated_confusion_matrix is not None
                    else {}
                ),
                **metrics,
            }
        )
    if not aggregates:
        raise ValueError("NAS workers returned no candidate results")
    return aggregates


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = (
        left["validation_accuracy"],
        left["validation_macro_f1"],
        -left["parameter_count"],
        -left["elapsed_seconds"],
        -left["peak_rss_mb"],
    )
    right_values = (
        right["validation_accuracy"],
        right["validation_macro_f1"],
        -right["parameter_count"],
        -right["elapsed_seconds"],
        -right["peak_rss_mb"],
    )
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def pareto_front(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in candidates
        if not any(
            other is not candidate and _dominates(other, candidate)
            for other in candidates
        )
    ]


def select_evolutionary_parents(
    candidates: list[dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    if count < 1 or len(candidates) < count:
        raise ValueError("Not enough measured candidates for evolutionary selection")
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(selected) < count:
        front = pareto_front(remaining)
        objectives = (
            lambda item: float(item["validation_accuracy"]),
            lambda item: float(item["validation_macro_f1"]),
            lambda item: -float(item["parameter_count"]),
            lambda item: -float(item["elapsed_seconds"]),
            lambda item: -float(item["peak_rss_mb"]),
        )
        distances = {id(item): 0.0 for item in front}
        for objective in objectives:
            ordered = sorted(front, key=objective)
            if len(ordered) <= 2:
                for item in ordered:
                    distances[id(item)] = float("inf")
                continue
            lower = objective(ordered[0])
            upper = objective(ordered[-1])
            distances[id(ordered[0])] = float("inf")
            distances[id(ordered[-1])] = float("inf")
            if upper > lower:
                for index in range(1, len(ordered) - 1):
                    if distances[id(ordered[index])] != float("inf"):
                        distances[id(ordered[index])] += (
                            objective(ordered[index + 1])
                            - objective(ordered[index - 1])
                        ) / (upper - lower)
        front.sort(
            key=lambda item: (
                -distances[id(item)],
                -float(item["validation_accuracy"]),
                -float(item["validation_macro_f1"]),
                item["architecture"]["id"],
            )
        )
        selected.extend(front[: count - len(selected)])
        front_ids = {item["architecture"]["id"] for item in front}
        remaining = [
            item for item in remaining if item["architecture"]["id"] not in front_ids
        ]
    return selected


def select_best_architecture(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    front = pareto_front(candidates)
    if not front:
        raise ValueError("No feasible candidate remains on the NAS Pareto front")
    return min(
        front,
        key=lambda item: (
            -float(item["validation_accuracy"]),
            -float(item["validation_macro_f1"]),
            int(item["parameter_count"]),
            float(item["elapsed_seconds"]),
            float(item["peak_rss_mb"]),
            item["architecture"]["id"],
        ),
    )
