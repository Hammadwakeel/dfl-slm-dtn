"""Bounded CIFAR-10 CNN architecture search space and candidate selection."""

from __future__ import annotations

from typing import Any

ARCHITECTURE_SPACE = tuple(
    {
        "id": f"b{blocks}_c{channels}_k{kernel_size}",
        "blocks": blocks,
        "channels": channels,
        "kernel_size": kernel_size,
    }
    for blocks in (1, 2)
    for channels in (8, 16)
    for kernel_size in (3, 5)
)
MAX_PARAMETERS = 12_000
FIXED_ARCHITECTURE = "b1_c8_k3"


def architecture_parameters(architecture: str | dict[str, Any]) -> int:
    spec = architecture_spec(architecture)
    channels = spec["channels"]
    kernel_size = spec["kernel_size"]
    total = 0
    input_channels = 3
    for _ in range(spec["blocks"]):
        total += kernel_size * kernel_size * input_channels * channels + channels
        input_channels = channels
    side = 32 // (2 ** (spec["blocks"] + 1))
    total += side * side * channels * 10 + 10
    return total


def architecture_spec(architecture: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(architecture, str):
        candidate_id = architecture
    elif isinstance(architecture, dict):
        candidate_id = architecture.get("id")
    else:
        raise ValueError("Architecture must be a candidate ID or candidate specification")
    for spec in ARCHITECTURE_SPACE:
        if candidate_id == spec["id"]:
            if isinstance(architecture, dict) and architecture != spec:
                raise ValueError("Architecture parameters do not match the candidate ID")
            if architecture_parameters_unchecked(spec) > MAX_PARAMETERS:
                raise ValueError("Architecture exceeds the parameter cap")
            return dict(spec)
    raise ValueError(f"Unsupported CIFAR-10 architecture: {candidate_id}")


def architecture_parameters_unchecked(spec: dict[str, Any]) -> int:
    channels = spec["channels"]
    kernel_size = spec["kernel_size"]
    input_channels = 3
    total = 0
    for _ in range(spec["blocks"]):
        total += kernel_size * kernel_size * input_channels * channels + channels
        input_channels = channels
    side = 32 // (2 ** (spec["blocks"] + 1))
    return total + side * side * channels * 10 + 10


def dataset_architecture_parameters(
    architecture: str | dict[str, Any],
    num_classes: int,
    *,
    input_channels: int = 3,
) -> int:
    if not 2 <= num_classes:
        raise ValueError("Dataset NAS requires at least two classes")
    if input_channels not in {1, 3}:
        raise ValueError("Dataset NAS supports one-channel grayscale or three-channel RGB")
    spec = architecture_spec(architecture)
    channels = spec["channels"]
    kernel_size = spec["kernel_size"]
    previous_channels = input_channels
    total = 0
    for _ in range(spec["blocks"]):
        total += kernel_size * kernel_size * previous_channels * channels + channels
        previous_channels = channels
    return total + channels * num_classes + num_classes


def select_dataset_architecture(
    candidate_results: list[dict[str, Any]],
    *,
    num_classes: int,
    worker_count: int = 3,
    input_channels: int = 3,
) -> dict[str, Any]:
    expected_ids = {
        spec["id"]
        for spec in ARCHITECTURE_SPACE
        if dataset_architecture_parameters(
            spec, num_classes, input_channels=input_channels
        ) <= MAX_PARAMETERS
    }
    if not expected_ids:
        raise ValueError("No NAS candidate fits the 12,000-parameter cap for this dataset")
    by_id: dict[str, list[dict[str, Any]]] = {}
    seen_workers: set[str] = set()
    for result in candidate_results:
        spec = architecture_spec(result["architecture"])
        candidate_id = spec["id"]
        worker_id = str(result["node_id"])
        accuracy = float(result["validation_accuracy"])
        if (
            candidate_id not in expected_ids
            or not worker_id
            or not 0 <= accuracy <= 1
            or int(result["parameter_count"])
            != dataset_architecture_parameters(
                spec, num_classes, input_channels=input_channels
            )
        ):
            raise ValueError("Dataset NAS candidate result is invalid")
        by_id.setdefault(candidate_id, []).append(result)
        seen_workers.add(worker_id)
    if len(seen_workers) != worker_count or set(by_id) != expected_ids:
        raise ValueError("Dataset NAS results must cover every eligible candidate and worker")
    if any(
        len(rows) != worker_count
        or len({str(row["node_id"]) for row in rows}) != worker_count
        for rows in by_id.values()
    ):
        raise ValueError("Dataset NAS requires exactly one result per worker and candidate")
    selected_id = min(
        expected_ids,
        key=lambda candidate_id: (
            -sum(float(row["validation_accuracy"]) for row in by_id[candidate_id])
            / worker_count,
            dataset_architecture_parameters(
                candidate_id, num_classes, input_channels=input_channels
            ),
            candidate_id,
        ),
    )
    return {
        "architecture": architecture_spec(selected_id),
        "parameter_count": dataset_architecture_parameters(
            selected_id, num_classes, input_channels=input_channels
        ),
        "mean_validation_accuracy": sum(
            float(row["validation_accuracy"]) for row in by_id[selected_id]
        )
        / worker_count,
        "candidate_means": [
            {
                "architecture": architecture_spec(candidate_id),
                "parameter_count": dataset_architecture_parameters(
                    candidate_id, num_classes, input_channels=input_channels
                ),
                "mean_validation_accuracy": sum(
                    float(row["validation_accuracy"]) for row in by_id[candidate_id]
                )
                / worker_count,
                "worker_accuracies": [
                    float(row["validation_accuracy"])
                    for row in sorted(by_id[candidate_id], key=lambda row: row["node_id"])
                ],
            }
            for candidate_id in sorted(expected_ids)
        ],
    }


def select_architecture(candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ids = {item["id"] for item in ARCHITECTURE_SPACE}
    by_id: dict[str, list[float]] = {}
    for result in candidate_results:
        candidate_id = str(result["architecture"]["id"])
        architecture_spec(result["architecture"])
        accuracy = float(result["validation_accuracy"])
        parameters = int(result["parameter_count"])
        if candidate_id not in expected_ids or not 0 <= accuracy <= 1:
            raise ValueError("Candidate result has an invalid architecture or accuracy")
        if parameters != architecture_parameters(candidate_id):
            raise ValueError("Candidate result parameter count does not match its architecture")
        by_id.setdefault(candidate_id, []).append(accuracy)
    if set(by_id) != expected_ids or any(len(scores) != 3 for scores in by_id.values()):
        raise ValueError("Architecture selection requires one result per worker for every candidate")
    selected_id = min(
        expected_ids,
        key=lambda candidate_id: (
            -sum(by_id[candidate_id]) / len(by_id[candidate_id]),
            architecture_parameters(candidate_id),
            candidate_id,
        ),
    )
    selected = architecture_spec(selected_id)
    return {
        "architecture": selected,
        "parameter_count": architecture_parameters(selected_id),
        "mean_validation_accuracy": sum(by_id[selected_id]) / len(by_id[selected_id]),
        "candidate_means": [
            {
                "architecture": architecture_spec(candidate_id),
                "parameter_count": architecture_parameters(candidate_id),
                "mean_validation_accuracy": sum(by_id[candidate_id]) / len(by_id[candidate_id]),
                "worker_accuracies": by_id[candidate_id],
                "mean_elapsed_seconds": sum(
                    float(row["elapsed_seconds"])
                    for row in candidate_results
                    if row["architecture"]["id"] == candidate_id
                )
                / len(by_id[candidate_id]),
                "max_peak_rss_mb": max(
                    float(row["peak_rss_mb"])
                    for row in candidate_results
                    if row["architecture"]["id"] == candidate_id
                ),
            }
            for candidate_id in sorted(expected_ids)
        ],
    }
