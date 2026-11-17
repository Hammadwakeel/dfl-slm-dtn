"""Stop only the identified active MNIST NAS group through ASK_HUMAN and verify idle."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import Settings  # noqa: E402
from app.services.worker_shards import get_worker_training_status  # noqa: E402
from m5_feedback_loop import BackendClient  # noqa: E402

EXPECTED_GROUP = "dataset-nas-1793260918-14afda"


def _workers(settings: Settings):
    return sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )


def _state(client: BackendClient) -> dict:
    state = client.request("GET", "/api/autonomy/state")
    if state.get("mode") != "ASK_HUMAN" or state.get("stop_requested") is not False:
        raise RuntimeError("Refusing cleanup unless ASK_HUMAN is active and global STOP is inactive")
    if state.get("pending_decisions"):
        raise RuntimeError("Refusing cleanup while unrelated decisions are pending")
    if any(loop.get("status") == "RUNNING" for loop in state.get("loops", [])):
        raise RuntimeError("Refusing cleanup while an autonomy loop is running")
    return state


def _workers_idle(settings: Settings) -> None:
    for worker in _workers(settings):
        status = get_worker_training_status(worker)
        if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
            raise RuntimeError(f"{worker.node_id} unexpectedly has active or scheduled training")
        print(f"{worker.node_id}: authenticated and idle")


def main() -> None:
    settings = Settings.from_env()
    if settings.feedback_rule_fallback_enabled:
        raise RuntimeError("Feedback fallback must remain disabled")
    client = BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)
    _state(client)
    _workers_idle(settings)
    results = client.request("GET", "/api/slm/results")
    if results.get("active_group_id") != EXPECTED_GROUP:
        raise RuntimeError("Active group no longer matches the failed TASK K NAS group")
    group = next(
        (item for item in results.get("groups", []) if item.get("group_id") == EXPECTED_GROUP),
        None,
    )
    if (
        not isinstance(group, dict)
        or group.get("kind") != "dataset_architecture_search"
        or group.get("dataset_name") != "mnist_demo"
        or group.get("status") != "running"
    ):
        raise RuntimeError("The identified active group is not the expected running MNIST NAS")

    action = {"action": "stop_training", "parameters": {}}
    reason = "Stop the MNIST NAS group after the acceptance runner could not decode its approved result envelope."
    evidence = {"dataset_name": "mnist_demo", "group_id": EXPECTED_GROUP, "restore_idle_after_runner_error": True}
    dispatched = client.request(
        "POST",
        "/api/autonomy/dispatch",
        {"action": action, "reason": reason, "evidence": evidence},
    )
    decision = dispatched.get("decision")
    if (
        dispatched.get("status") != "pending"
        or not isinstance(decision, dict)
        or decision.get("status") != "PENDING"
        or decision.get("action") != "stop_training"
        or not isinstance(decision.get("id"), str)
    ):
        raise RuntimeError("Stop request did not create an ASK_HUMAN pending decision")
    print(
        f"stop_decision_id={decision['id']}; reason={decision.get('reason')}; "
        f"evidence={decision.get('evidence')}"
    )

    _state(client)
    results = client.request("GET", "/api/slm/results")
    if results.get("active_group_id") != EXPECTED_GROUP:
        raise RuntimeError("Active group changed before the stop decision could be approved")
    approved = client.request(
        "POST",
        f"/api/autonomy/decisions/{decision['id']}/approve",
        timeout=120,
    )
    if approved.get("status") != "approved":
        raise RuntimeError("ASK_HUMAN stop decision was not approved")

    deadline = time.monotonic() + 180
    terminal = None
    while time.monotonic() < deadline:
        results = client.request("GET", "/api/slm/results")
        if results.get("active_group_id") is None:
            terminal = next(
                (item for item in results.get("groups", []) if item.get("group_id") == EXPECTED_GROUP),
                None,
            )
            break
        time.sleep(2)
    if terminal is None:
        raise RuntimeError("MNIST NAS group did not return idle within 180 seconds")
    _workers_idle(settings)
    final_state = _state(client)
    events = [
        item.get("event")
        for item in client.request("GET", "/api/autonomy/history")
        if item.get("decision_id") == decision["id"]
    ]
    if not {"created", "approved"}.issubset(events):
        raise RuntimeError("Stop-decision history omitted creation or approval")
    print(
        f"restored_idle=True; group_status={terminal.get('status')}; "
        f"group_error={terminal.get('error')}; stop_history={events}; "
        f"mode={final_state.get('mode')}; global_STOP={final_state.get('stop_requested')}"
    )


if __name__ == "__main__":
    main()
