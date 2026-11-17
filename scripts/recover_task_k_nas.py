"""Read-only recovery report for a previously accepted MNIST NAS group."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import Settings  # noqa: E402
from m5_feedback_loop import BackendClient  # noqa: E402

GROUP_ID = "dataset-nas-1793260918-14afda"
DECISION_ID = "00769179-e2a0-40f2-9451-4bc579f47004"


def main() -> None:
    settings = Settings.from_env()
    client = BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)
    results = client.request("GET", "/api/slm/results")
    groups = results.get("groups")
    if not isinstance(groups, list):
        raise RuntimeError("Results response omitted groups")
    group = next(
        (item for item in groups if isinstance(item, dict) and item.get("group_id") == GROUP_ID),
        None,
    )
    if group is None:
        print(f"group_id={GROUP_ID}; status=NOT_FOUND")
    else:
        print(
            f"group_id={GROUP_ID}; status={group.get('status')}; "
            f"selected_architecture={json.dumps(group.get('selected_architecture'), sort_keys=True)}; "
            f"parameter_count={group.get('selected_parameter_count')}; "
            f"duration_seconds={group.get('search_elapsed_seconds')}; error={group.get('error')}"
        )
        workers = group.get("worker_search")
        candidates = group.get("nas_candidates")
        if not isinstance(workers, list) or not isinstance(candidates, list):
            print("candidate_results=UNAVAILABLE")
        else:
            for candidate in candidates:
                architecture = candidate.get("architecture", {})
                architecture_id = architecture.get("id")
                per_worker = []
                for worker_result in workers:
                    detail = next(
                        (
                            item
                            for item in worker_result.get("candidates", [])
                            if item.get("architecture", {}).get("id") == architecture_id
                        ),
                        None,
                    )
                    if detail is None:
                        raise RuntimeError(f"Missing worker result for candidate {architecture_id}")
                    per_worker.append(
                        {
                            "worker": detail.get("node_id"),
                            "validation_accuracy": detail.get("validation_accuracy"),
                            "validation_samples": detail.get("validation_samples"),
                        }
                    )
                print(
                    "candidate="
                    + json.dumps(
                        {
                            "architecture": architecture_id,
                            "parameter_count": candidate.get("parameter_count"),
                            "mean_validation_accuracy": candidate.get("mean_validation_accuracy"),
                            "workers": per_worker,
                        },
                        sort_keys=True,
                    )
                )

    history = client.request("GET", "/api/autonomy/history")
    matches = [
        {
            "event": entry.get("event"),
            "details": entry.get("details"),
            "created_at": entry.get("created_at"),
        }
        for entry in history
        if entry.get("decision_id") == DECISION_ID
    ]
    print(f"decision_id={DECISION_ID}; history={json.dumps(matches, sort_keys=True)}")

    state = client.request("GET", "/api/autonomy/state")
    pending = int(state.get("pending_decisions", -1))
    running_loops = [
        loop.get("id") for loop in state.get("loops", []) if loop.get("status") == "RUNNING"
    ]
    print(
        f"autonomy_mode={state.get('mode')}; global_stop={state.get('stop_requested')}; "
        f"pending_decisions={pending}; running_loops={json.dumps(running_loops)}"
    )
    if pending != 0 or running_loops:
        raise RuntimeError("Autonomy has pending decisions or running loops")


if __name__ == "__main__":
    main()
