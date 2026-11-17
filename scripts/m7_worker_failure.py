"""Inject one controlled worker crash during a real ASK_HUMAN training run."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import Settings  # noqa: E402
from app.services.worker_shards import get_worker_training_status  # noqa: E402
from m5_feedback_loop import BackendClient, _action_data, _sample_tegrastats  # noqa: E402

WORKER1_IP = "192.168.11.123"
DATASET = "cifar_10k_demo"


class M7Error(RuntimeError):
    pass


def _ssh(command: str, timeout: int = 45) -> str:
    process = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            f"hammad@{WORKER1_IP}",
            command,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if process.returncode:
        raise M7Error(
            f"Worker1 SSH command failed (exit {process.returncode}): "
            f"{process.stderr.strip()[-300:]}"
        )
    return process.stdout.strip()


def _worker_status(worker: Any) -> dict[str, Any]:
    request = Request(
        f"{worker.training_url}/v1/status",
        headers={
            "Authorization": f"Bearer {worker.token}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise M7Error(
            f"Authenticated status request to {worker.node_id} failed: "
            f"HTTP {error.code}"
        ) from error
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise M7Error(
            f"Authenticated status request to {worker.node_id} failed: "
            f"{type(error).__name__}"
        ) from error
    if not isinstance(result, dict) or result.get("node_id") != worker.node_id:
        raise M7Error(f"{worker.node_id} returned an invalid status response")
    return result


def _check_preflight(
    backend: BackendClient,
    settings: Settings,
) -> None:
    state = backend.request("GET", "/api/autonomy/state")
    if (
        state.get("mode") != "ASK_HUMAN"
        or state.get("stop_requested") is not False
        or state.get("pending_decisions") != 0
        or any(loop.get("status") == "RUNNING" for loop in state.get("loops", []))
    ):
        raise M7Error("M7 requires ASK_HUMAN, STOP inactive, no pending decision, and no running loop")
    results = backend.request("GET", "/api/slm/results")
    if results.get("active_group_id") is not None:
        raise M7Error("Master already has an active group")
    workers = sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )
    if [node.node_id for node in workers] != ["worker1", "worker2", "worker3"]:
        raise M7Error("Expected worker1, worker2, and worker3")
    for node in workers:
        status = get_worker_training_status(node)
        if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
            raise M7Error(f"{node.node_id} is not idle before the controlled test")
        print(f"{node.node_id}: authenticated idle check passed")
    datasets = backend.request("GET", "/api/datasets")
    if not any(
        isinstance(item, dict)
        and item.get("dataset_name", item.get("name")) == DATASET
        for item in datasets
    ):
        raise M7Error(f"Dataset {DATASET} is not available in the backend")


def _create_loop(backend: BackendClient) -> str:
    loop = backend.request(
        "POST",
        "/api/autonomy/loops",
        {
            "max_iterations": 2,
            "time_limit_seconds": 900,
            "stop_rule": "max_iterations",
        },
    )
    if loop.get("status") != "RUNNING" or not isinstance(loop.get("id"), str):
        raise M7Error("Backend did not create a running autonomy loop")
    return loop["id"]


def _dispatch_training(
    backend: BackendClient,
    loop_id: str,
    *,
    reason: str,
) -> tuple[str, str]:
    action = {
        "action": "start_training",
        "parameters": {
            "dataset_name": DATASET,
            "rounds": 3,
            "epochs": 2,
        },
    }
    dispatched = backend.request(
        "POST",
        "/api/autonomy/dispatch",
        {
            "action": action,
            "reason": reason,
            "evidence": {"loop_id": loop_id, "dataset_name": DATASET},
        },
    )
    decision = dispatched.get("decision")
    if (
        dispatched.get("status") != "pending"
        or not isinstance(decision, dict)
        or decision.get("status") != "PENDING"
        or not isinstance(decision.get("id"), str)
    ):
        raise M7Error("ASK_HUMAN dispatch did not create a pending decision")
    decision_id = decision["id"]
    approved = backend.request(
        "POST",
        f"/api/autonomy/decisions/{decision_id}/approve",
        timeout=900,
    )
    if approved.get("status") != "approved":
        raise M7Error(f"Decision {decision_id} did not approve successfully")
    data = _action_data(approved)
    group_id = data.get("group_id")
    if not isinstance(group_id, str):
        raise M7Error("Approved training action did not return a group id")
    history = backend.request("GET", "/api/autonomy/history")
    events = [
        row.get("event")
        for row in history
        if row.get("decision_id") == decision_id
    ]
    if not {"created", "approved"}.issubset(events):
        raise M7Error("Training approval is missing created/approved history")
    print(
        f"decision_id={decision_id}; group_id={group_id}; "
        f"events={json.dumps(events)}"
    )
    return decision_id, group_id


def _group(backend: BackendClient, group_id: str) -> dict[str, Any]:
    results = backend.request("GET", "/api/slm/results")
    groups = results.get("groups")
    if not isinstance(groups, list):
        raise M7Error("Master results omitted its groups")
    found = next(
        (
            item
            for item in groups
            if isinstance(item, dict) and item.get("group_id") == group_id
        ),
        None,
    )
    if found is None:
        raise M7Error(f"Master results omitted group {group_id}")
    return found


def _wait_for_worker1_active(
    backend: BackendClient,
    settings: Settings,
    group_id: str,
    timeout_seconds: int = 30,
) -> str:
    worker1 = next(node for node in settings.nodes if node.node_id == "worker1")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        group = _group(backend, group_id)
        if group.get("status") != "running":
            raise M7Error(
                f"Training group ended {group.get('status')} before the failure injection"
            )
        status = _worker_status(worker1)
        active_run_id = status.get("active_run_id")
        running_ids = {
            run.get("run_id")
            for run in group.get("runs", [])
            if isinstance(run, dict) and run.get("status") == "running"
        }
        if isinstance(active_run_id, str) and active_run_id in running_ids:
            return active_run_id
        time.sleep(0.25)
    raise M7Error("Worker1 did not enter the approved training run within 30 seconds")


def _wait_terminal(
    backend: BackendClient,
    group_id: str,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        group = _group(backend, group_id)
        if group.get("status") in {"completed", "failed", "stopped"}:
            return group
        time.sleep(1)
    raise M7Error(f"Group {group_id} did not reach a terminal state in time")


def _wait_idle(worker: Any, timeout_seconds: int = 90) -> float:
    started = time.monotonic()
    deadline = started + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((worker.ip, 5004), timeout=1):
                pass
            status = get_worker_training_status(worker)
        except (OSError, RuntimeError):
            time.sleep(1)
            continue
        if status.get("active_run_id") is None and status.get("scheduled_run_id") is None:
            return time.monotonic() - started
        time.sleep(1)
    raise M7Error("Worker1 did not return to authenticated idle status within 90 seconds")


def main() -> None:
    settings = Settings.from_env()
    backend = BackendClient("http://127.0.0.1:8000", settings.api_token)
    _check_preflight(backend, settings)
    loop_id = _create_loop(backend)
    worker1 = next(node for node in settings.nodes if node.node_id == "worker1")

    with _sample_tegrastats(settings, "m7_worker1_sigkill"):
        _, group_id = _dispatch_training(
            backend,
            loop_id,
            reason="M7 controlled worker1 crash recovery acceptance",
        )
        run_id = _wait_for_worker1_active(backend, settings, group_id)
        print(f"failure injection target: group={group_id}; run={run_id}; worker=worker1")
        kill_started = time.monotonic()
        _ssh("sudo -n systemctl kill -s SIGKILL dfl-slm-worker-training.service")
        print("worker1 SIGKILL injected during active 3-round x 2-epoch run")
        auto_recovery_seconds = _wait_idle(worker1)
        print(
            "worker1 automatic systemd recovery to port 5004 + authenticated idle: "
            f"{auto_recovery_seconds:.2f} s"
        )
        group = _wait_terminal(backend, group_id)
        if group.get("status") != "failed" or not group.get("error"):
            raise M7Error(
                f"Expected failed master group with a reason; got "
                f"{group.get('status')}: {group.get('error')}"
            )
        reason = f"worker1 SIGKILL: {group['error']}"[:1000]
        paused = backend.request(
            "POST",
            f"/api/autonomy/loops/{loop_id}/iterations",
            {"worker_failure": reason},
        )
        if paused.get("status") != "PAUSED" or paused.get("failure_reason") != reason:
            raise M7Error("Autonomy loop did not pause with the worker failure reason")
        history = backend.request("GET", "/api/autonomy/history")
        failure_events = [
            row
            for row in history
            if row.get("event") == "loop_paused"
            and row.get("details", {}).get("loop_id") == loop_id
        ]
        if not failure_events:
            raise M7Error("Decision history did not record the paused worker-failure loop")
        for node in settings.nodes:
            if node.role != "worker" or node.node_id == "worker1":
                continue
            status = get_worker_training_status(node)
            if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
                raise M7Error(f"{node.node_id} was not idle after the failed group")
        print(f"failed_group_reason={group['error']}")
        print(f"loop_status={paused['status']}; failure_reason={paused['failure_reason']}")
        print("decision_history_loop_paused=" + json.dumps(failure_events, sort_keys=True))

    worker1_status = get_worker_training_status(worker1)
    if worker1_status.get("active_run_id") is not None or worker1_status.get("scheduled_run_id") is not None:
        raise M7Error("Worker1 is not idle before the explicit systemd restart")
    restart_started = time.monotonic()
    _ssh("sudo -n systemctl restart dfl-slm-worker-training.service")
    restart_seconds = _wait_idle(worker1)
    print(f"explicit worker1 systemd restart to authenticated idle: {restart_seconds:.2f} s")
    print(
        "SIGKILL-to-first-idle recovery: "
        f"{auto_recovery_seconds:.2f} s; post-restart verification: "
        f"{time.monotonic() - restart_started:.2f} s"
    )
    print("worker1 process service restored")

    recovery_loop_id = _create_loop(backend)
    _, recovery_group_id = _dispatch_training(
        backend,
        recovery_loop_id,
        reason="M7 post-recovery training verification",
    )
    recovery_group = _wait_terminal(backend, recovery_group_id)
    if recovery_group.get("status") != "completed":
        raise M7Error(
            f"Post-recovery training did not complete: "
            f"{recovery_group.get('status')}: {recovery_group.get('error')}"
        )
    for node in settings.nodes:
        if node.role == "worker":
            status = get_worker_training_status(node)
            if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
                raise M7Error(f"{node.node_id} was not idle after recovery training")
    backend.request(
        "POST",
        f"/api/autonomy/loops/{recovery_loop_id}/stop",
        {"reason": "Recovery training completed successfully"},
    )
    print(
        f"recovery_group={recovery_group_id}; status=completed; "
        f"rounds={recovery_group.get('rounds')}; epochs={recovery_group.get('epochs')}; "
        f"duration_seconds={recovery_group.get('completed_at', 0) - recovery_group.get('started_at', 0):.2f}"
    )


if __name__ == "__main__":
    main()
