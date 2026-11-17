"""Combine persisted real TASK K evidence and compare the two MNIST NAS runs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import Settings  # noqa: E402
from m5_feedback_loop import BackendClient  # noqa: E402

OLD_GROUP_ID = "dataset-nas-1793260918-14afda"
NAS_TELEMETRY_PATH = ROOT / "data" / "mnist_nas_telemetry_rerun.json"
TRAINING_FEEDBACK_PATH = ROOT / "data" / "mnist_task_k_training_feedback.json"
OUTPUT_PATH = ROOT / "data" / "task_k_mnist_results.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path.name}")
    return value


def _nas_projection(group: dict[str, Any]) -> dict[str, Any]:
    candidates = group.get("nas_candidates")
    workers = group.get("worker_search")
    if not isinstance(candidates, list) or not isinstance(workers, list):
        raise RuntimeError("NAS group omitted candidates or worker_search")
    rows = []
    for candidate in candidates:
        architecture_id = candidate.get("architecture", {}).get("id")
        worker_results = []
        for worker in workers:
            detail = next(
                (
                    item
                    for item in worker.get("candidates", [])
                    if item.get("architecture", {}).get("id") == architecture_id
                ),
                None,
            )
            if detail is None:
                raise RuntimeError(f"NAS group omitted {architecture_id} for a worker")
            worker_results.append(
                {
                    "node_id": detail.get("node_id"),
                    "validation_accuracy": detail.get("validation_accuracy"),
                    "validation_samples": detail.get("validation_samples"),
                }
            )
        rows.append(
            {
                "architecture_id": architecture_id,
                "parameter_count": candidate.get("parameter_count"),
                "mean_validation_accuracy": candidate.get("mean_validation_accuracy"),
                "workers": worker_results,
            }
        )
    return {
        "group_id": group.get("group_id"),
        "status": group.get("status"),
        "dataset_name": group.get("dataset_name"),
        "selected_architecture": group.get("selected_architecture"),
        "parameter_count": group.get("selected_parameter_count"),
        "duration_seconds": group.get("search_elapsed_seconds"),
        "candidates": rows,
    }


def _training_projection(group: dict[str, Any], tegrastats: dict[str, Any]) -> dict[str, Any]:
    runs = group.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or runs[0].get("status") != "completed":
        raise RuntimeError("Expected exactly one completed training run")
    rounds = runs[0].get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise RuntimeError("Training group has no round results")
    final_workers = rounds[-1].get("workers", {})
    per_worker: dict[str, dict[str, Any]] = {}
    bytes_total = 0
    for node in ("worker1", "worker2", "worker3"):
        if node not in final_workers:
            raise RuntimeError(f"Training group omitted {node}")
        final = final_workers[node]
        accuracy = final.get("validation_accuracy", final.get("accuracy"))
        macro_f1 = final.get("validation_macro_f1", final.get("macro_f1"))
        if not isinstance(accuracy, (int, float)) or not isinstance(macro_f1, (int, float)):
            raise RuntimeError(f"Final metrics are missing for {node}")
        sent = sum(row.get("workers", {}).get(node, {}).get("bytes_sent", 0) for row in rounds)
        received = sum(row.get("workers", {}).get(node, {}).get("bytes_received", 0) for row in rounds)
        bytes_total += sent + received
        per_worker[node] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "bytes_sent": sent,
            "bytes_received": received,
        }
    return {
        "group_id": group.get("group_id"),
        "status": group.get("status"),
        "dataset_name": group.get("dataset_name"),
        "topology": group.get("topology"),
        "rounds": group.get("rounds"),
        "epochs": group.get("epochs"),
        "seed": runs[0].get("seed"),
        "accuracy": sum(row["accuracy"] for row in per_worker.values()) / 3,
        "macro_f1": sum(row["macro_f1"] for row in per_worker.values()) / 3,
        "per_worker": per_worker,
        "communication_bytes_total_sent_plus_received": bytes_total,
        "duration_seconds": group.get("completed_at", 0) - group.get("started_at", 0),
        "tegrastats": tegrastats,
    }


def main() -> None:
    telemetry = _read_json(NAS_TELEMETRY_PATH)
    acceptance = _read_json(TRAINING_FEEDBACK_PATH)
    settings = Settings.from_env()
    client = BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)
    results = client.request("GET", "/api/slm/results")
    old_group = next(
        (item for item in results.get("groups", []) if item.get("group_id") == OLD_GROUP_ID),
        None,
    )
    if not isinstance(old_group, dict):
        raise RuntimeError("Recovered first NAS group is no longer present")

    new_nas = {
        "group_id": telemetry["group_id"],
        "status": telemetry["status"],
        "dataset_name": telemetry["dataset_name"],
        "selected_architecture": telemetry["selected_architecture"],
        "parameter_count": telemetry["parameter_count"],
        "duration_seconds": telemetry["search_duration_seconds"],
        "candidates": [
            {
                "architecture_id": item["architecture"]["id"],
                "parameter_count": item["parameter_count"],
                "mean_validation_accuracy": item["mean_validation_accuracy"],
                "workers": item["workers"],
            }
            for item in telemetry["candidates"]
        ],
        "tegrastats": telemetry["tegrastats"],
    }
    old_nas = _nas_projection(old_group)
    same_architecture = (
        old_nas["selected_architecture"].get("id")
        == new_nas["selected_architecture"].get("id")
    )
    old_candidates = {item["architecture_id"]: item for item in old_nas["candidates"]}
    new_candidates = {item["architecture_id"]: item for item in new_nas["candidates"]}
    common_ids = sorted(set(old_candidates) & set(new_candidates))
    same_candidate_accuracies = all(
        math.isclose(
            float(old_candidates[key]["mean_validation_accuracy"]),
            float(new_candidates[key]["mean_validation_accuracy"]),
            rel_tol=0,
            abs_tol=1e-12,
        )
        for key in common_ids
    )
    old_selected_mean = old_candidates[
        old_nas["selected_architecture"]["id"]
    ]["mean_validation_accuracy"]
    new_selected_mean = new_candidates[
        new_nas["selected_architecture"]["id"]
    ]["mean_validation_accuracy"]

    training = acceptance.get("steps", {}).get("training", {})
    feedback = acceptance.get("steps", {}).get("feedback", {})
    feedback_group = feedback.get("follow_up_group")
    if not isinstance(feedback_group, dict):
        raise RuntimeError("Feedback action's completed follow-up group was not persisted")
    feedback_tegrastats = feedback.get("tegrastats")
    if not isinstance(feedback_tegrastats, dict):
        raise RuntimeError("Feedback tegrastats peaks are missing")
    feedback_training = _training_projection(feedback_group, feedback_tegrastats)

    state = client.request("GET", "/api/autonomy/state")
    if state.get("pending_decisions") != 0 or any(
        loop.get("status") == "RUNNING" for loop in state.get("loops", [])
    ):
        raise RuntimeError("Autonomy is not idle after TASK K runs")
    output = {
        "dataset_name": "mnist_demo",
        "comparison_with_cifar": "NOT MADE",
        "recovered_nas": old_nas,
        "telemetry_nas_rerun": new_nas,
        "nas_comparison": {
            "same_selected_architecture": same_architecture,
            "same_candidate_mean_accuracies": same_candidate_accuracies,
            "old_selected_mean_validation_accuracy": old_selected_mean,
            "new_selected_mean_validation_accuracy": new_selected_mean,
            "selected_accuracy_delta": new_selected_mean - old_selected_mean,
            "clearly_above_10_percent_chance": False,
        },
        "short_training": training,
        "feedback": {
            **{key: value for key, value in feedback.items() if key != "follow_up_group"},
            "follow_up_training": feedback_training,
        },
        "fallback_enabled": settings.feedback_rule_fallback_enabled,
        "final_autonomy_state": {
            "mode": state.get("mode"),
            "stop_requested": state.get("stop_requested"),
            "pending_decisions": state.get("pending_decisions"),
            "running_loops": sum(loop.get("status") == "RUNNING" for loop in state.get("loops", [])),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"task_k_summary_json={OUTPUT_PATH}")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
