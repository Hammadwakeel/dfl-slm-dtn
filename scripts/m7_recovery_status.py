"""Read current M7 recovery state without changing the backend or devices."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import Settings  # noqa: E402
from m5_feedback_loop import BackendClient  # noqa: E402
from m7_worker_failure import _worker_status  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    backend = BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)
    state = backend.request("GET", "/api/autonomy/state")
    results = backend.request("GET", "/api/slm/results")
    print(
        "autonomy="
        + json.dumps(
            {
                "mode": state.get("mode"),
                "stop_requested": state.get("stop_requested"),
                "pending_decisions": state.get("pending_decisions"),
                "loops": [
                    {
                        "id": loop.get("id"),
                        "status": loop.get("status"),
                        "failure_reason": loop.get("failure_reason"),
                    }
                    for loop in state.get("loops", [])
                ],
            },
            sort_keys=True,
        )
    )
    print(f"master_active_group={results.get('active_group_id')}")
    print(
        "groups="
        + json.dumps(
            [
                {
                    "group_id": group.get("group_id"),
                    "status": group.get("status"),
                    "error": group.get("error"),
                    "rounds": group.get("rounds"),
                    "epochs": group.get("epochs"),
                }
                for group in results.get("groups", [])
                if group.get("status") in {"running", "failed", "stopped"}
            ],
            sort_keys=True,
        )
    )
    for worker in sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    ):
        status = _worker_status(worker)
        print(
            f"{worker.node_id}: active_run_id={status.get('active_run_id')}; "
            f"scheduled_run_id={status.get('scheduled_run_id')}"
        )


if __name__ == "__main__":
    main()
