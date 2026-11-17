"""Build bounded feedback summaries from completed dataset-training groups."""

import math
from typing import Any


class FeedbackSummaryError(ValueError):
    """Raised when a completed experiment cannot be summarized safely."""


def build_training_feedback_summary(
    group: dict[str, Any],
    *,
    iteration: int,
    previous_accuracy: float | None = None,
) -> dict[str, Any]:
    if group.get("status") != "completed":
        raise FeedbackSummaryError("Feedback requires a completed experiment group")
    if group.get("kind") != "decentralized" or not isinstance(group.get("dataset_name"), str):
        raise FeedbackSummaryError("Feedback requires a dataset-specific decentralized run")
    runs = group.get("runs")
    if not isinstance(runs, list) or not runs:
        raise FeedbackSummaryError("Experiment group is missing completed runs")

    worker_accuracies: dict[str, list[float]] = {}
    macro_f1_values: list[float] = []
    communication_bytes = 0
    for run in runs:
        if not isinstance(run, dict) or run.get("status") != "completed":
            raise FeedbackSummaryError("Feedback includes an incomplete training run")
        rounds = run.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            raise FeedbackSummaryError("Completed run is missing round metrics")
        for round_index, round_row in enumerate(rounds):
            if not isinstance(round_row, dict):
                raise FeedbackSummaryError("Training round is malformed")
            workers = round_row.get("workers")
            if not isinstance(workers, dict) or not workers:
                raise FeedbackSummaryError("Training round is missing worker metrics")
            for worker_id, metrics in workers.items():
                if not isinstance(metrics, dict):
                    raise FeedbackSummaryError(f"Metrics for {worker_id} are malformed")
                if round_index == len(rounds) - 1:
                    accuracy = metrics.get("validation_accuracy")
                    macro_f1 = metrics.get("validation_macro_f1")
                    if (
                        not isinstance(accuracy, (int, float))
                        or not isinstance(macro_f1, (int, float))
                        or not math.isfinite(float(accuracy))
                        or not math.isfinite(float(macro_f1))
                        or not 0 <= float(accuracy) <= 1
                        or not 0 <= float(macro_f1) <= 1
                    ):
                        raise FeedbackSummaryError(
                            f"Validation metrics for {worker_id} are missing or invalid"
                        )
                    worker_accuracies.setdefault(worker_id, []).append(float(accuracy))
                    macro_f1_values.append(float(macro_f1))
                sent = metrics.get("bytes_sent")
                if not isinstance(sent, int) or sent < 0:
                    raise FeedbackSummaryError(f"Communication bytes for {worker_id} are invalid")
                communication_bytes += sent

    if not worker_accuracies:
        raise FeedbackSummaryError("No final-round validation metrics were recorded")
    worker_means = [
        sum(scores) / len(scores) for scores in worker_accuracies.values()
    ]
    accuracy = sum(worker_means) / len(worker_means)
    if previous_accuracy is not None and (
        not math.isfinite(previous_accuracy) or not 0 <= previous_accuracy <= 1
    ):
        raise FeedbackSummaryError("Previous accuracy is outside the valid range")

    started_at = group.get("started_at")
    completed_at = group.get("completed_at")
    topology = group.get("topology")
    rounds = group.get("rounds")
    epochs = group.get("epochs")
    nas_enabled = group.get("nas_enabled")
    if (
        not isinstance(started_at, (int, float))
        or not isinstance(completed_at, (int, float))
        or completed_at < started_at
        or not isinstance(topology, str)
        or not isinstance(rounds, int)
        or not isinstance(epochs, int)
        or not isinstance(nas_enabled, bool)
    ):
        raise FeedbackSummaryError("Experiment group is missing valid run configuration")

    return {
        "dataset_name": group["dataset_name"],
        "topology": topology,
        "rounds": rounds,
        "epochs": epochs,
        "nas_enabled": nas_enabled,
        "accuracy": accuracy,
        "macro_f1": sum(macro_f1_values) / len(macro_f1_values),
        "communication_bytes": communication_bytes,
        "duration_seconds": float(completed_at - started_at),
        "worker_accuracy_min": min(worker_means),
        "worker_accuracy_max": max(worker_means),
        "worker_accuracy_spread": max(worker_means) - min(worker_means),
        "previous_accuracy": previous_accuracy,
        "accuracy_delta": (
            accuracy - previous_accuracy if previous_accuracy is not None else None
        ),
        "iteration": iteration,
    }
