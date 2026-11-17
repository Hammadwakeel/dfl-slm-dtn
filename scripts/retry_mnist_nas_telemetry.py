"""Run one ASK_HUMAN MNIST dataset-NAS rerun for complete tegrastats capture."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import Settings  # noqa: E402
from m5_feedback_loop import BackendClient, _redact_tokens  # noqa: E402
from mnist_task_k_acceptance import (  # noqa: E402
    DATASET_NAME,
    TaskKError,
    _dispatch,
    _preflight,
    _run_sampled_approval,
    _workers,
)
from app.services.autonomy import validate_action  # noqa: E402

OUTPUT_PATH = ROOT / "data" / "mnist_nas_telemetry_rerun.json"


def _summary(group: dict[str, Any], sampling: str) -> dict[str, Any]:
    selected = group.get("selected_architecture")
    candidates = group.get("nas_candidates")
    worker_search = group.get("worker_search")
    if not isinstance(selected, dict) or not isinstance(candidates, list) or not isinstance(worker_search, list):
        raise TaskKError("NAS results omitted the selected architecture or candidate worker results")
    candidate_rows = []
    for candidate in candidates:
        architecture = candidate.get("architecture", {})
        architecture_id = architecture.get("id")
        workers = []
        for worker_result in worker_search:
            match = next(
                (
                    row
                    for row in worker_result.get("candidates", [])
                    if row.get("architecture", {}).get("id") == architecture_id
                ),
                None,
            )
            if match is None:
                raise TaskKError(f"NAS worker result omitted candidate {architecture_id}")
            workers.append(
                {
                    "node_id": match.get("node_id"),
                    "validation_accuracy": match.get("validation_accuracy"),
                    "validation_samples": match.get("validation_samples"),
                }
            )
        candidate_rows.append(
            {
                "architecture": architecture,
                "parameter_count": candidate.get("parameter_count"),
                "mean_validation_accuracy": candidate.get("mean_validation_accuracy"),
                "workers": workers,
            }
        )
    scores = [
        row["mean_validation_accuracy"]
        for row in candidate_rows
        if isinstance(row.get("mean_validation_accuracy"), (int, float))
    ]
    selected_row = next(
        (row for row in candidate_rows if row["architecture"].get("id") == selected.get("id")),
        None,
    )
    if selected_row is None:
        raise TaskKError("Selected NAS architecture is not in candidate results")
    sample = json.loads(sampling)
    return {
        "status": group.get("status"),
        "dataset_name": group.get("dataset_name"),
        "group_id": group.get("group_id"),
        "search_time_limit_seconds": group.get("search_time_limit_seconds"),
        "search_duration_seconds": group.get("search_elapsed_seconds"),
        "approval_wall_seconds": sample.get("elapsed_seconds"),
        "selected_architecture": selected,
        "parameter_count": group.get("selected_parameter_count"),
        "selected_mean_validation_accuracy": selected_row["mean_validation_accuracy"],
        "chance_accuracy": 0.1,
        "descriptively_above_chance": selected_row["mean_validation_accuracy"] > 0.1,
        "candidate_scores_differ": len(set(scores)) > 1,
        "candidates": candidate_rows,
        "tegrastats": sample.get("nodes"),
    }


def main() -> None:
    settings = Settings.from_env()
    client = BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)
    workers = _workers(settings)
    result: dict[str, Any] = {
        "dataset_name": DATASET_NAME,
        "status": "RUNNING",
        "search_time_limit_seconds": 120,
    }
    try:
        _preflight(client, settings, workers)
        action = {
            "action": "enable_nas",
            "parameters": {
                "enabled": True,
                "dataset_name": DATASET_NAME,
                "search_time_limit_seconds": 120,
            },
        }
        validate_action(action)
        decision = _dispatch(
            client,
            action,
            "Rerun dataset-specific NAS once to capture full-interval Jetson RAM and swap telemetry for MNIST.",
            {"dataset_name": DATASET_NAME, "search_time_limit_seconds": 120, "telemetry_rerun": True},
        )
        result["decision_id"] = decision["id"]
        _, group, sampling = _run_sampled_approval(
            client,
            settings,
            workers,
            decision,
            label="mnist-nas-telemetry-rerun",
            timeout_seconds=300,
        )
        if group is None or group.get("status") != "completed":
            raise TaskKError("The MNIST NAS rerun did not complete successfully")
        result.update(_summary(group, sampling))
        result["status"] = "PASS"
    except Exception as error:
        result["status"] = "FAIL"
        result["error_type"] = type(error).__name__
        result["error"] = str(_redact_tokens(str(error), settings))
        raise
    finally:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        safe_result = _redact_tokens(result, settings)
        OUTPUT_PATH.write_text(json.dumps(safe_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"nas_rerun_json={OUTPUT_PATH}")
        print(json.dumps(safe_result, sort_keys=True))


if __name__ == "__main__":
    main()
