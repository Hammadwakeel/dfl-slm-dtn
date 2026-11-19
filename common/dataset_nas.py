"""Bounded architecture generation and Pareto selection for dataset NAS."""

from __future__ import annotations

import random
from typing import Any

from common.model_design import validate_model_design

MAX_DATASET_NAS_PARAMETERS = 1_000_000
INITIAL_POPULATION_SIZE = 8
# CIFAR-10 trains in the long-lived worker service on an 8 GB Jetson. The search's
# memory is dominated by TensorFlow and the shard, not by these small CNNs, so the
# space allows up to three blocks of 64 channels under a parameter cap. The former
# space (two blocks of <= 32 channels, three candidates) used ~1 of its 10 minutes
# and could only find ~3k-parameter models.
CIFAR_INITIAL_POPULATION_SIZE = 6
MEMORY_CONSTRAINED_MAX_PARAMETERS = 150_000
MEMORY_CONSTRAINED_CHANNELS = (16, 32, 64)
MEMORY_CONSTRAINED_DROPOUT = (0.0, 0.2, 0.35)
FINAL_POPULATION_SIZE = 4
# Fidelity schedule measured on 8 Jetson workers (~7.5k MNIST samples each, ~16 s per
# epoch): 8 candidates x 1 epoch, then 4 x 3 epochs, fits the 600 s budget with margin.
# The former 3/10 schedule needed ~670 s and timed out on every worker.
INITIAL_EPOCHS = 1
FINAL_EPOCHS = 3
MAX_BLOCKS = 3
CHANNEL_CHOICES = (16, 32, 64)
KERNEL_CHOICES = (3, 5)
HEAD_CHOICES = (0, 64)
DROPOUT_CHOICES = (0.0, 0.2, 0.35)
BLOCK_CHOICES = ("conv2d", "separable_conv2d")
# Extended CIFAR-10 space (NAS v2). The compact space above never contained the
# VGG-style models that federate well on CIFAR-10 (a 6-conv baseline reached 0.67
# while the compact search kept returning ~3k-parameter models), so v2 adds
# double-conv blocks and 128 channels under a larger parameter cap.
EXTENDED_CHANNELS = (32, 64, 128)
EXTENDED_MAX_PARAMETERS = 400_000
CONVS_PER_BLOCK_CHOICES = (1, 2)
ALL_CHANNEL_CHOICES = (16, 32, 64, 128)


def _candidate_id(
    channels: tuple[int, ...],
    kernel_size: int,
    block_type: str,
    head_units: int,
    dropout: float,
    attention: bool,
    convs_per_block: int = 1,
) -> str:
    width_key = "x".join(str(width) for width in channels)
    dropout_key = str(int(dropout * 100))
    candidate_id = (
        f"nas-d{len(channels)}-w{width_key}-k{kernel_size}-"
        f"{'sep' if block_type == 'separable_conv2d' else 'conv'}-"
        f"h{head_units}-drop{dropout_key}"
    )
    if convs_per_block != 1:
        candidate_id = f"{candidate_id}-c{convs_per_block}"
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
    convs_per_block: int = 1,
) -> dict[str, Any]:
    if not 2 <= len(channels) <= MAX_BLOCKS:
        raise ValueError(f"Dataset NAS depth must be between 2 and {MAX_BLOCKS}")
    if convs_per_block not in CONVS_PER_BLOCK_CHOICES:
        raise ValueError("Dataset NAS convolutions per block must be 1 or 2")
    if any(width not in ALL_CHANNEL_CHOICES for width in channels):
        raise ValueError("Dataset NAS channel widths are not in the supported search space")
    if kernel_size not in KERNEL_CHOICES or block_type not in BLOCK_CHOICES:
        raise ValueError("Dataset NAS kernel or block type is unsupported")
    if head_units not in HEAD_CHOICES or dropout not in DROPOUT_CHOICES:
        raise ValueError("Dataset NAS classifier or dropout choice is unsupported")
    if not isinstance(attention, bool):
        raise ValueError("Dataset NAS attention choice must be a boolean")

    layers: list[dict[str, Any]] = []
    for width in channels:
        for _ in range(convs_per_block):
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
            convs_per_block,
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
    if convs_per_block != 1:
        # Recorded only for double-conv blocks so v1 designs keep their exact form.
        architecture["convs_per_block"] = convs_per_block
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
    channel_choices = MEMORY_CONSTRAINED_CHANNELS if memory_constrained else CHANNEL_CHOICES
    depth = rng.randint(2, MAX_BLOCKS)
    channels = tuple(
        sorted(rng.choices(channel_choices, k=depth))
    )
    candidate = build_dataset_architecture(
        input_shape=input_shape,
        num_classes=num_classes,
        channels=channels,
        kernel_size=3 if memory_constrained else rng.choice(KERNEL_CHOICES),
        block_type=rng.choice(BLOCK_CHOICES),
        head_units=rng.choice(HEAD_CHOICES),
        dropout=rng.choice(MEMORY_CONSTRAINED_DROPOUT if memory_constrained else DROPOUT_CHOICES),
        attention=False if memory_constrained else rng.random() < 0.2,
    )
    if memory_constrained:
        _check_memory_constrained(candidate)
    return candidate


def _check_memory_constrained(candidate: dict[str, Any]) -> None:
    if int(candidate["parameter_count"]) > MEMORY_CONSTRAINED_MAX_PARAMETERS:
        raise ValueError(
            f"{candidate['id']} exceeds the {MEMORY_CONSTRAINED_MAX_PARAMETERS:,}-parameter "
            "cap for memory-constrained search"
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
                ((32, 64), "conv2d", 64, 0.2, False),
                ((16, 32, 64), "conv2d", 64, 0.2, False),
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
            mutation = rng.randrange(5)
            if mutation == 0:
                index = rng.randrange(len(channels))
                channels[index] = rng.choice(MEMORY_CONSTRAINED_CHANNELS)
            elif mutation == 1:
                block_type = (
                    "separable_conv2d"
                    if block_type == "conv2d"
                    else "conv2d"
                )
            elif mutation == 2:
                head_units = 64 if head_units == 0 else 0
            elif mutation == 3:
                if len(channels) < MAX_BLOCKS:
                    channels.append(rng.choice(MEMORY_CONSTRAINED_CHANNELS))
                elif len(channels) > 2:
                    channels.pop(rng.randrange(len(channels)))
            else:
                dropout = rng.choice(MEMORY_CONSTRAINED_DROPOUT)
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
                _check_memory_constrained(candidate)
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


EXTENDED_INITIAL_POPULATION_SIZE = 6
EXTENDED_OFFSPRING = 2
EXTENDED_DROPOUT = (0.0, 0.2, 0.35)


def _extended_candidate(
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    channels: tuple[int, ...],
    convs_per_block: int,
    head_units: int,
    dropout: float,
) -> dict[str, Any]:
    if any(width not in EXTENDED_CHANNELS for width in channels):
        raise ValueError("Extended NAS widths must come from the extended channel set")
    candidate = build_dataset_architecture(
        input_shape=input_shape,
        num_classes=num_classes,
        channels=tuple(sorted(channels)),
        kernel_size=3,
        block_type="conv2d",
        head_units=head_units,
        dropout=dropout,
        convs_per_block=convs_per_block,
    )
    if int(candidate["parameter_count"]) > EXTENDED_MAX_PARAMETERS:
        raise ValueError(
            f"{candidate['id']} exceeds the {EXTENDED_MAX_PARAMETERS:,}-parameter extended cap"
        )
    return candidate


def extended_initial_population(
    *,
    seed: int,
    input_shape: tuple[int, int, int],
    num_classes: int,
    size: int = EXTENDED_INITIAL_POPULATION_SIZE,
) -> list[dict[str, Any]]:
    """First NAS v2 generation: two known-good anchors plus random extended designs.

    The anchors are a VGG-style double-conv CNN and the fixed 3-block CNN of the
    topology sweeps, so the search always measures at least one strong design.
    """
    if not 2 <= size <= 8:
        raise ValueError("Extended NAS population size must be between 2 and 8")
    rng = random.Random(seed)
    population: dict[str, dict[str, Any]] = {}
    for channels, convs, head, dropout in (((32, 64, 128), 2, 64, 0.2), ((32, 64, 64), 1, 64, 0.2)):
        anchor = _extended_candidate(
            input_shape=input_shape,
            num_classes=num_classes,
            channels=channels,
            convs_per_block=convs,
            head_units=head,
            dropout=dropout,
        )
        population[anchor["id"]] = anchor
    for _ in range(size * 100):
        if len(population) >= size:
            break
        depth = rng.randint(2, MAX_BLOCKS)
        try:
            candidate = _extended_candidate(
                input_shape=input_shape,
                num_classes=num_classes,
                channels=tuple(rng.choices(EXTENDED_CHANNELS, k=depth)),
                convs_per_block=rng.choice(CONVS_PER_BLOCK_CHOICES),
                head_units=rng.choice(HEAD_CHOICES),
                dropout=rng.choice(EXTENDED_DROPOUT),
            )
        except ValueError:
            continue
        population.setdefault(candidate["id"], candidate)
    if len(population) != size:
        raise ValueError("Could not generate enough unique extended NAS architectures")
    return list(population.values())


def mutate_extended_population(
    parents: list[dict[str, Any]],
    *,
    seed: int,
    input_shape: tuple[int, int, int],
    num_classes: int,
    size: int = EXTENDED_OFFSPRING,
) -> list[dict[str, Any]]:
    """Parents plus ``size`` one-step mutations (width, depth, convs, head, dropout).

    Only searched designs (with a channel schedule) are mutated; an external seed
    architecture such as the deployed baseline can still be a parent and is kept.
    """
    mutable = [parent for parent in parents if parent.get("channel_schedule")]
    if not mutable:
        raise ValueError("Extended NAS evolution needs at least one searched parent")
    rng = random.Random(seed)
    parent_ids = {parent["id"] for parent in parents}
    offspring: dict[str, dict[str, Any]] = {}
    for _ in range(size * 200):
        if len(offspring) >= size:
            break
        parent = rng.choice(mutable)
        channels = [int(width) for width in parent["channel_schedule"]]
        convs = int(parent.get("convs_per_block", 1))
        head = int(parent["head_units"])
        dropout = float(parent["dropout"])
        mutation = rng.randrange(5)
        if mutation == 0:
            channels[rng.randrange(len(channels))] = rng.choice(EXTENDED_CHANNELS)
        elif mutation == 1:
            if len(channels) < MAX_BLOCKS and (len(channels) == 2 or rng.random() < 0.5):
                channels.append(rng.choice(EXTENDED_CHANNELS))
            else:
                channels.pop(rng.randrange(len(channels)))
        elif mutation == 2:
            convs = 2 if convs == 1 else 1
        elif mutation == 3:
            head = 0 if head else 64
        else:
            dropout = rng.choice([value for value in EXTENDED_DROPOUT if value != dropout])
        try:
            candidate = _extended_candidate(
                input_shape=input_shape,
                num_classes=num_classes,
                channels=tuple(channels),
                convs_per_block=convs,
                head_units=head,
                dropout=dropout,
            )
        except ValueError:
            continue
        if candidate["id"] not in parent_ids:
            offspring.setdefault(candidate["id"], candidate)
    if len(offspring) != size:
        raise ValueError("Could not generate enough unique extended NAS offspring")
    return [*parents, *offspring.values()]


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
        # Accuracy on the shared class-balanced split (NAS v2 objective), when every
        # worker measured it; the weakest worker shows how evenly a design learns.
        if all(isinstance(row.get("global_validation_accuracy"), (int, float)) for row in rows):
            metrics["global_validation_accuracy"] = sum(
                float(row["global_validation_accuracy"]) for row in rows
            ) / len(rows)
            metrics["global_validation_macro_f1"] = sum(
                float(row["global_validation_macro_f1"]) for row in rows
            ) / len(rows)
            metrics["global_validation_min_accuracy"] = min(
                float(row["global_validation_accuracy"]) for row in rows
            )
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
