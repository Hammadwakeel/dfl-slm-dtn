"""Read-only inspection of TASK K autonomy and training state after an acceptance error."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import Settings  # noqa: E402
from app.services.worker_shards import get_worker_training_status  # noqa: E402
from m5_feedback_loop import BackendClient  # noqa: E402

DECISION_ID = "00769179-e2a0-40f2-9451-4bc579f47004"


def main() -> None:
    settings = Settings.from_env()
    client = BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)
    state = client.request("GET", "/api/autonomy/state")
    results = client.request("GET", "/api/slm/results")
    group_id = results.get("active_group_id")
    print(
        f"autonomy_mode={state.get('mode')}; global_stop={state.get('stop_requested')}; "
        f"pending_decisions={state.get('pending_decisions')}; active_group_id={group_id}"
    )
    if group_id:
        group = next(
            (item for item in results.get("groups", []) if item.get("group_id") == group_id),
            None,
        )
        if group is None:
            raise RuntimeError("Active group id is missing from results")
        print(
            f"active_group_status={group.get('status')}; kind={group.get('kind')}; "
            f"dataset={group.get('dataset_name')}; error={group.get('error')}"
        )
    history = client.request("GET", "/api/autonomy/history")
    related = [
        item
        for item in history
        if item.get("decision_id") == DECISION_ID
    ]
    for item in related:
        print(
            f"decision_event={item.get('event')}; details={item.get('details')}"
        )
    for worker in sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    ):
        status = get_worker_training_status(worker)
        print(
            f"{worker.node_id}: active_run_id={status.get('active_run_id')}; "
            f"scheduled_run_id={status.get('scheduled_run_id')}"
        )


if __name__ == "__main__":
    main()
