"""Pause the failed post-reboot loop after confirming all hardware is idle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import Settings  # noqa: E402
from app.services.worker_shards import get_worker_training_status  # noqa: E402
from m5_feedback_loop import BackendClient  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    backend = BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)
    state = backend.request("GET", "/api/autonomy/state")
    results = backend.request("GET", "/api/slm/results")
    if state.get("mode") != "ASK_HUMAN" or state.get("stop_requested") is not False:
        raise RuntimeError("Safe cleanup requires ASK_HUMAN and global STOP inactive")
    if state.get("pending_decisions") != 0:
        raise RuntimeError("Safe cleanup requires no pending decisions")
    if results.get("active_group_id") is not None:
        raise RuntimeError("Master still has an active group; refusing loop cleanup")
    for worker in settings.nodes:
        if worker.role == "worker":
            get_worker_training_status(worker)

    failed_group = next(
        (
            group
            for group in results.get("groups", [])
            if group.get("group_id") == "bidirectional_ring-1793177958-9444d2"
        ),
        None,
    )
    if not isinstance(failed_group, dict) or failed_group.get("status") != "failed":
        raise RuntimeError("Expected failed post-reboot recovery group was not found")
    loop = next(
        (
            row
            for row in state.get("loops", [])
            if row.get("status") == "RUNNING"
        ),
        None,
    )
    if not isinstance(loop, dict):
        raise RuntimeError("No running loop remains to pause")
    reason = (
        f"Post-reboot recovery group {failed_group['group_id']} failed: "
        f"{failed_group.get('error')}"
    )[:1000]
    paused = backend.request(
        "POST",
        f"/api/autonomy/loops/{loop['id']}/iterations",
        {"worker_failure": reason},
    )
    if paused.get("status") != "PAUSED":
        raise RuntimeError("Failed post-reboot loop did not pause")
    history = backend.request("GET", "/api/autonomy/history")
    event = next(
        (
            row
            for row in history
            if row.get("event") == "loop_paused"
            and row.get("details", {}).get("loop_id") == loop["id"]
        ),
        None,
    )
    if event is None:
        raise RuntimeError("Loop pause was not recorded in decision history")
    final = backend.request("GET", "/api/autonomy/state")
    if any(row.get("status") == "RUNNING" for row in final.get("loops", [])):
        raise RuntimeError("A running autonomy loop remains after cleanup")
    print(
        json.dumps(
            {
                "group_id": failed_group["group_id"],
                "group_status": failed_group["status"],
                "group_error": failed_group.get("error"),
                "loop_id": loop["id"],
                "loop_status": paused["status"],
                "loop_failure_reason": paused.get("failure_reason"),
                "history_event": event.get("event"),
                "workers_idle": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
