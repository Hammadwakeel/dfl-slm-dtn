"""Reboot idle worker1 and verify persistent data, model, and training recovery."""

from __future__ import annotations

import argparse
import hashlib
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
from m5_feedback_loop import BackendClient, _action_data  # noqa: E402
from m7_worker_failure import _check_preflight  # noqa: E402

WORKER1_IP = "192.168.11.123"
DATASET = "cifar_10k_demo"
MODEL_DATASET = "cifar_subset_demo"
MODEL_FILES = ("architecture.json", "initial_weights.npz")


class WorkerRebootError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ssh(
    command: str,
    *,
    timeout: int = 12,
    allow_disconnect: bool = False,
) -> tuple[int, str]:
    process = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
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
    if process.returncode and not (allow_disconnect and process.returncode == 255):
        raise WorkerRebootError(
            f"Worker1 SSH command failed (exit {process.returncode}): "
            f"{process.stderr.strip()[-300:]}"
        )
    return process.returncode, process.stdout.strip()


def _pc_shard_path() -> Path:
    reports = sorted(
        (ROOT / "data" / "datasets" / DATASET / "processed").glob(
            "run-*/preprocessing_report.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not reports:
        raise WorkerRebootError("PC preprocessing report for cifar_10k_demo is missing")
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    shard = Path(report["workers"]["worker1"]["shard_path"])
    if not shard.is_file():
        raise WorkerRebootError("PC worker1 shard artifact is missing")
    return shard


def _model_hashes(directory: Path) -> dict[str, str]:
    hashes = {}
    for name in MODEL_FILES:
        path = directory / name
        if not path.is_file():
            return {}
        hashes[name] = _sha256(path)
    return hashes


def _remote_hashes(paths: dict[str, str]) -> dict[str, str]:
    command = "set -eu; "
    for label, path in paths.items():
        command += (
            f"if test -f {path!r}; then sha256sum {path!r}; "
            f"else printf 'MISSING  {label}\\n'; fi; "
        )
    _, output = _ssh(command)
    hashes: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[0] == "MISSING":
            continue
        hashes[parts[1].rsplit("/", 1)[-1]] = parts[0]
    return hashes


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
        raise WorkerRebootError(f"Authenticated worker1 status returned HTTP {error.code}") from error
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerRebootError(
            f"Authenticated worker1 status failed: {type(error).__name__}"
        ) from error
    if not isinstance(result, dict) or result.get("node_id") != "worker1":
        raise WorkerRebootError("Worker1 returned an invalid status object")
    return result


def _poll_reboot(worker: Any, old_boot_id: str) -> tuple[float, float, float]:
    started = time.monotonic()
    deadline = started + 600
    ssh_seconds: float | None = None
    service_seconds: float | None = None
    auth_seconds: float | None = None
    while time.monotonic() < deadline:
        try:
            _, boot_id = _ssh("cat /proc/sys/kernel/random/boot_id", timeout=8)
            if boot_id == old_boot_id:
                time.sleep(2)
                continue
            if ssh_seconds is None:
                ssh_seconds = time.monotonic() - started
            _, active = _ssh(
                "systemctl is-active dfl-slm-worker-training.service",
                timeout=8,
            )
            if active == "active" and service_seconds is None:
                service_seconds = time.monotonic() - started
            if active == "active":
                with socket.create_connection((worker.ip, 5004), timeout=2):
                    pass
                status = get_worker_training_status(worker)
                if status.get("active_run_id") is None and status.get("scheduled_run_id") is None:
                    auth_seconds = time.monotonic() - started
                    return ssh_seconds, service_seconds or auth_seconds, auth_seconds
        except WorkerRebootError as error:
            if "SSH command failed" not in str(error):
                raise
        except (OSError, socket.timeout):
            pass
        time.sleep(2)
    raise WorkerRebootError("Worker1 did not recover SSH, service, and authenticated status within 10 minutes")


def _approve_action(
    backend: BackendClient,
    action: dict[str, Any],
    reason: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    dispatched = backend.request(
        "POST",
        "/api/autonomy/dispatch",
        {"action": action, "reason": reason, "evidence": evidence},
    )
    decision = dispatched.get("decision")
    if (
        dispatched.get("status") != "pending"
        or not isinstance(decision, dict)
        or decision.get("status") != "PENDING"
        or not isinstance(decision.get("id"), str)
    ):
        raise WorkerRebootError("ASK_HUMAN did not create a pending M7 decision")
    approved = backend.request(
        "POST",
        f"/api/autonomy/decisions/{decision['id']}/approve",
        timeout=900,
    )
    if approved.get("status") != "approved":
        raise WorkerRebootError(f"Decision {decision['id']} was not approved")
    events = [
        row.get("event")
        for row in backend.request("GET", "/api/autonomy/history")
        if row.get("decision_id") == decision["id"]
    ]
    if not {"created", "approved"}.issubset(events):
        raise WorkerRebootError("M7 approved action is missing decision history")
    print(f"decision_id={decision['id']}; events={json.dumps(events)}")
    return approved


def _wait_group(backend: BackendClient, group_id: str, timeout: int = 300) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        results = backend.request("GET", "/api/slm/results")
        groups = results.get("groups")
        if not isinstance(groups, list):
            raise WorkerRebootError("Master results omitted its groups")
        group = next(
            (
                item
                for item in groups
                if isinstance(item, dict) and item.get("group_id") == group_id
            ),
            None,
        )
        if group is None:
            raise WorkerRebootError(f"Master results omitted group {group_id}")
        if group.get("status") in {"completed", "failed", "stopped"}:
            return group
        time.sleep(2)
    raise WorkerRebootError(f"Recovery group {group_id} did not finish within {timeout} seconds")


def _verify_short_training(backend: BackendClient, settings: Settings) -> None:
    _check_preflight(backend, settings)
    loop = backend.request(
        "POST",
        "/api/autonomy/loops",
        {"max_iterations": 2, "time_limit_seconds": 600, "stop_rule": "max_iterations"},
    )
    action = {
        "action": "start_training",
        "parameters": {"dataset_name": DATASET, "rounds": 1, "epochs": 1},
    }
    approved = _approve_action(
        backend,
        action,
        "M7 post-worker-reboot 1-round 1-epoch recovery check",
        {"loop_id": loop["id"], "dataset_name": DATASET},
    )
    data = _action_data(approved)
    group_id = data.get("group_id")
    if not isinstance(group_id, str):
        raise WorkerRebootError("Recovery training action omitted group id")
    group = _wait_group(backend, group_id)
    if group.get("status") != "completed":
        raise WorkerRebootError(
            f"Post-reboot training ended {group.get('status')}: {group.get('error')}"
        )
    backend.request(
        "POST",
        f"/api/autonomy/loops/{loop['id']}/stop",
        {"reason": "Post-reboot recovery training completed"},
    )
    print(
        f"recovery_group={group_id}; status=completed; rounds={group.get('rounds')}; "
        f"epochs={group.get('epochs')}; duration_seconds="
        f"{group.get('completed_at', 0) - group.get('started_at', 0):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    backend = BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)
    _check_preflight(backend, settings)
    worker = next(node for node in settings.nodes if node.node_id == "worker1")
    status = get_worker_training_status(worker)
    if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
        raise WorkerRebootError("Worker1 is not idle immediately before reboot")

    shard = _pc_shard_path()
    shard_hash = _sha256(shard)
    model_dir = ROOT / "data" / "datasets" / MODEL_DATASET / "models" / "initial_model"
    expected_model_hashes = _model_hashes(model_dir)
    if len(expected_model_hashes) != len(MODEL_FILES):
        raise WorkerRebootError("PC initial-model artifact for cifar_subset_demo is incomplete")
    remote_paths = {
        "train.npz": f"/var/lib/dfl-slm-dtn/datasets/{DATASET}/train.npz",
        **{
            name: f"/var/lib/dfl-slm-dtn/datasets/{MODEL_DATASET}/models/initial_model/{name}"
            for name in (*MODEL_FILES, "metadata.json")
        },
    }
    before = _remote_hashes(remote_paths)
    expected_before = {"train.npz": shard_hash, **expected_model_hashes}
    if (
        any(before.get(name) != digest for name, digest in expected_before.items())
        or "metadata.json" not in before
    ):
        raise WorkerRebootError(
            "Worker1 shard/model hash mismatch before reboot; "
            f"PC={expected_before}; worker1={before}"
        )
    print(
        f"worker1 pre-reboot: shard_sha256={shard_hash}; "
        f"initial_model_sha256={json.dumps(expected_model_hashes, sort_keys=True)}"
    )
    if args.preflight_only:
        return

    _, old_boot_id = _ssh("cat /proc/sys/kernel/random/boot_id")
    code, _ = _ssh("sudo -n systemctl reboot", timeout=15, allow_disconnect=True)
    if code not in {0, 255}:
        raise WorkerRebootError(f"Reboot command returned unexpected SSH code {code}")
    ssh_seconds, service_seconds, auth_seconds = _poll_reboot(worker, old_boot_id)
    print(
        f"worker1 reboot recovery: SSH={ssh_seconds:.2f}s; "
        f"service_active={service_seconds:.2f}s; authenticated_idle={auth_seconds:.2f}s"
    )

    after = _remote_hashes(remote_paths)
    model_survived = (
        all(after.get(name) == digest for name, digest in expected_model_hashes.items())
        and "metadata.json" in after
    )
    if after.get("train.npz") != shard_hash:
        raise WorkerRebootError("Worker1 shard hash changed across reboot")
    if not model_survived:
        workers = [
            node.node_id
            for node in settings.nodes
            if node.role == "worker"
        ]
        if workers != ["worker1", "worker2", "worker3"]:
            raise WorkerRebootError("Unexpected worker list while reinstalling the model")
        action = {
            "action": "create_initial_model",
            "parameters": {"model_name": MODEL_DATASET},
        }
        installed = _approve_action(
            backend,
            action,
            "M7 restore the missing persistent initial model through the approved action",
            {"dataset_name": MODEL_DATASET, "reason": "worker reboot persistence recovery"},
        )
        if not installed:
            raise WorkerRebootError("Approved initial model restoration returned no result")
        after = _remote_hashes(remote_paths)
        if (
            any(after.get(name) != digest for name, digest in expected_before.items())
            or "metadata.json" not in after
        ):
            raise WorkerRebootError("Approved initial model restoration did not restore PC hashes")
    print(
        f"worker1 post-reboot: shard_sha256={after.get('train.npz')}; "
        f"initial_model_survived={model_survived}; "
        f"initial_model_sha256={json.dumps({name: after.get(name) for name in MODEL_FILES}, sort_keys=True)}"
    )
    status = get_worker_training_status(worker)
    if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
        raise WorkerRebootError("Worker1 was not authenticated idle after reboot")
    _verify_short_training(backend, settings)


if __name__ == "__main__":
    main()
