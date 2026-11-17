"""Safely deploy stale master NAS code using the existing PowerShell installer."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402
from app.services.worker_shards import get_worker_training_status  # noqa: E402


BACKUP_DESTINATIONS = [
    "/opt/dfl-slm-dtn/master/app.py",
    "/opt/dfl-slm-dtn/master/__init__.py",
    "/opt/dfl-slm-dtn/master/wsgi.py",
    "/opt/dfl-slm-dtn/master/requirements.txt",
    "/opt/dfl-slm-dtn/master/training.py",
    "/opt/dfl-slm-dtn/master/cifar_evaluator.py",
    "/opt/dfl-slm-dtn/master/tests/test_app.py",
    "/opt/dfl-slm-dtn/common/__init__.py",
    "/opt/dfl-slm-dtn/common/cifar_model.py",
    "/opt/dfl-slm-dtn/common/dataset_nas.py",
    "/opt/dfl-slm-dtn/common/nas.py",
    "/opt/dfl-slm-dtn/common/slm_action_schema.json",
    "/opt/dfl-slm-dtn/topology_opt/__init__.py",
    "/opt/dfl-slm-dtn/topology_opt/reward.py",
    "/opt/dfl-slm-dtn/topology_opt/selector.py",
    "/opt/dfl-slm-dtn/topology_opt/reward_config.json",
    "/opt/dfl-slm-dtn/data/cifar10_test.npz",
    "/opt/dfl-slm-dtn/data/cifar10_train.npz",
    "/etc/dfl-slm-master.env",
    "/etc/systemd/system/dfl-slm-master.service",
]

DEPLOYED_SOURCES = {
    "master/app.py": ROOT / "master" / "app.py",
    "master/__init__.py": ROOT / "master" / "__init__.py",
    "master/wsgi.py": ROOT / "master" / "wsgi.py",
    "master/requirements.txt": ROOT / "master" / "requirements.txt",
    "master/training.py": ROOT / "master" / "training.py",
    "master/cifar_evaluator.py": ROOT / "master" / "cifar_evaluator.py",
    "common/__init__.py": ROOT / "common" / "__init__.py",
    "common/cifar_model.py": ROOT / "common" / "cifar_model.py",
    "common/dataset_nas.py": ROOT / "common" / "dataset_nas.py",
    "common/nas.py": ROOT / "common" / "nas.py",
    "common/slm_action_schema.json": ROOT / "common" / "slm_action_schema.json",
    "topology_opt/__init__.py": ROOT / "topology_opt" / "__init__.py",
    "topology_opt/reward.py": ROOT / "topology_opt" / "reward.py",
    "topology_opt/selector.py": ROOT / "topology_opt" / "selector.py",
    "topology_opt/reward_config.json": ROOT / "topology_opt" / "reward_config.json",
    "data/cifar10_test.npz": ROOT / "data" / "raw" / "cifar10_test.npz",
    "data/cifar10_train.npz": ROOT / "data" / "raw" / "cifar10_train.npz",
}


class DeploymentError(RuntimeError):
    pass


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


class Backend:
    def __init__(self, base: str, token: str) -> None:
        self.base = base.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 25,
    ) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(self.base + path, data=body, headers=self.headers, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise DeploymentError(f"Backend {method} {path} returned HTTP {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise DeploymentError(
                f"Backend {method} {path} failed: {type(error).__name__}: {error}"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DeploymentError(f"Backend {method} {path} returned invalid JSON") from error


def _ssh(user: str, host: str, command: str, timeout: int = 45) -> str:
    process = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            f"{user}@{host}",
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
        raise DeploymentError(
            f"SSH to master failed (exit {process.returncode}): {process.stderr.strip()[:500]}"
        )
    return process.stdout


def _hashes_match(host: str, user: str) -> None:
    remote_root = "/opt/dfl-slm-dtn"
    relpaths = list(DEPLOYED_SOURCES)
    args = " ".join(
        shlex.quote(f"{remote_root}/{relative}") for relative in relpaths
    )
    command = (
        "sh -c "
        + shlex.quote(
            "for f in "
            + args
            + "; do if test -f \"$f\"; then sha256sum \"$f\"; "
            + "else printf 'MISSING  %s\\n' \"$f\"; fi; done"
        )
    )
    output = _ssh(user, host, command)
    observed: dict[str, str | None] = {}
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, full_path = parts
        observed[full_path.replace(remote_root + "/", "", 1)] = (
            None if digest == "MISSING" else digest
        )
    mismatches: list[str] = []
    for relative, local_path in DEPLOYED_SOURCES.items():
        local_hash = hashlib.sha256(local_path.read_bytes()).hexdigest()
        remote_hash = observed.get(relative)
        print(f"{relative} PC={local_hash} MASTER={remote_hash or 'MISSING'}")
        if local_hash != remote_hash:
            mismatches.append(relative)
    if mismatches:
        raise DeploymentError("Post-deployment hash mismatch: " + ", ".join(mismatches))


def _preflight(backend: Backend, settings: Settings, ssh_user: str, master_ip: str) -> None:
    state = backend.request("GET", "/api/autonomy/state")
    if state.get("mode") != "ASK_HUMAN" or state.get("stop_requested") is not False:
        raise DeploymentError("Autonomy must be ASK_HUMAN with global STOP inactive")
    if state.get("pending_decisions"):
        raise DeploymentError("Review existing pending autonomy decisions before deployment")
    master_results = backend.request("GET", "/api/slm/results")
    if master_results.get("active_group_id") is not None:
        raise DeploymentError("A master training or NAS group is active")
    workers = sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )
    worker_ids = [node.node_id for node in workers]
    if len(worker_ids) < 3 or worker_ids != [
        f"worker{index}" for index in range(1, len(worker_ids) + 1)
    ]:
        raise DeploymentError("Expected at least three sequentially numbered workers")
    for worker in workers:
        status = get_worker_training_status(worker)
        print(
            f"{worker.node_id}: idle; active_run_id={status.get('active_run_id')}; "
            f"scheduled_run_id={status.get('scheduled_run_id')}"
        )
    print(f"master: no active training/NAS group; autonomy={state['mode']}; STOP inactive")
    print("--- free -m before deployment ---")
    print(_ssh(ssh_user, master_ip, "free -m"))


def _backup(host: str, user: str, timestamp: str) -> tuple[list[str], list[str]]:
    loop = " ".join(shlex.quote(path) for path in BACKUP_DESTINATIONS)
    shell = (
        "set -eu; stamp="
        + shlex.quote(timestamp)
        + "; for path in "
        + loop
        + "; do if sudo -n test -f \"$path\"; then "
        + "sudo -n cp -a -- \"$path\" \"$path.bak-$stamp\"; "
        + "printf 'BACKUP %s\\n' \"$path.bak-$stamp\"; "
        + "else printf 'ABSENT %s\\n' \"$path\"; fi; done"
    )
    output = _ssh(user, host, "sh -c " + shlex.quote(shell))
    backups = [line[7:] for line in output.splitlines() if line.startswith("BACKUP ")]
    absent = [line[7:] for line in output.splitlines() if line.startswith("ABSENT ")]
    expected = len(BACKUP_DESTINATIONS)
    if len(backups) + len(absent) != expected:
        raise DeploymentError("Remote backup did not account for every installer destination")
    print("Remote backup files:")
    for path in backups:
        print(path)
    return backups, absent


def _set_topology(backend: Backend) -> str:
    state = backend.request("GET", "/api/autonomy/state")
    if state.get("mode") != "ASK_HUMAN" or state.get("stop_requested") is not False:
        raise DeploymentError("Cannot set topology unless ASK_HUMAN is active and STOP is inactive")
    dispatched = backend.request(
        "POST",
        "/api/autonomy/dispatch",
        {
            "action": {
                "action": "set_topology",
                "parameters": {"topology": "bidirectional_ring"},
            },
            "reason": "Restore the required topology after master service restart.",
            "evidence": {"master_service_restarted": True},
        },
    )
    decision = dispatched.get("decision")
    if (
        dispatched.get("status") != "pending"
        or not isinstance(decision, dict)
        or decision.get("status") != "PENDING"
        or not isinstance(decision.get("id"), str)
    ):
        raise DeploymentError("Topology change did not create an ASK_HUMAN pending decision")
    decision_id = decision["id"]
    approved = backend.request(
        "POST",
        f"/api/autonomy/decisions/{decision_id}/approve",
        timeout=90,
    )
    if approved.get("status") != "approved":
        raise DeploymentError("Topology decision was not approved successfully")
    print(f"topology decision_id={decision_id}; approved bidirectional_ring")
    return decision_id


def _interpret(settings: Settings) -> None:
    request = Request(
        settings.master_slm_url.rstrip("/") + "/interpret",
        data=json.dumps(
            {"message": "Please report the current topology and cluster status."}
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.master_slm_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.slm_timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise DeploymentError(f"Master /interpret returned HTTP {error.code}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise DeploymentError(
            f"Master /interpret failed: {type(error).__name__}: {error}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeploymentError("Master /interpret returned invalid JSON") from error
    action = result.get("action")
    if not isinstance(action, dict) or not isinstance(action.get("action"), str):
        raise DeploymentError("Master /interpret response omitted a validated action")
    print(f"/interpret verification: HTTP 200; action={action['action']}")


def _rollback(host: str, user: str, timestamp: str, absent: list[str]) -> None:
    commands: list[str] = ["set -eu"]
    for path in BACKUP_DESTINATIONS:
        backup = f"{path}.bak-{timestamp}"
        if path in absent:
            commands.append(f"sudo -n rm -f -- {shlex.quote(path)}")
        else:
            commands.append(
                f"sudo -n cp -a -- {shlex.quote(backup)} {shlex.quote(path)}"
            )
    commands.extend(
        [
            "sudo -n systemctl daemon-reload",
            "sudo -n systemctl restart dfl-slm-master.service",
        ]
    )
    _ssh(user, host, "sh -c " + shlex.quote("; ".join(commands)), timeout=90)


def _deploy() -> None:
    settings = Settings.from_env()
    master = next(node for node in settings.nodes if node.role == "master")
    ssh_user = os.environ.get("M4_SSH_USER", "hammad")
    backend = Backend("http://127.0.0.1:8000", settings.api_token)
    _preflight(backend, settings, ssh_user, master.ip)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backups, absent = _backup(master.ip, ssh_user, timestamp)
    unexpected_absent = [
        path for path in absent
        if path != "/opt/dfl-slm-dtn/common/dataset_nas.py"
    ]
    if unexpected_absent:
        raise DeploymentError(
            "Installer destination(s) were absent during backup; refusing deployment: "
            + ", ".join(unexpected_absent)
        )
    deployment_started = False
    try:
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "deploy" / "deploy_master_slm.ps1"),
        ]
        deployment_started = True
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            check=False,
        )
        if result.stdout:
            print(result.stdout.strip())
        if result.returncode:
            detail = result.stderr.strip()[-1000:]
            raise DeploymentError(
                f"Existing master deployment method failed (exit {result.returncode}): {detail}"
            )
        deadline = time.monotonic() + 90
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                status = backend.request("GET", "/api/slm/status", timeout=15)
                ollama = status.get("ollama", {})
                if status.get("node_id") != "master" or not ollama.get("model_available"):
                    raise DeploymentError("Master status is not ready with the configured model")
                break
            except Exception as error:
                last_error = error
                time.sleep(2)
        else:
            raise DeploymentError(f"Master readiness timed out: {last_error}")

        _set_topology(backend)
        _interpret(settings)
        _hashes_match(master.ip, ssh_user)
        print("--- free -m after deployment ---")
        print(_ssh(ssh_user, master.ip, "free -m"))
        print("Master deployment verification: PASS")
        print("Jetson backup files:")
        for path in backups:
            print(path)
    except Exception as error:
        if deployment_started:
            try:
                _rollback(master.ip, ssh_user, timestamp, absent)
                _set_topology(backend)
                print("Rollback completed; topology restored through ASK_HUMAN approval.")
            except Exception as rollback_error:
                raise DeploymentError(
                    f"{error}; rollback also failed: {rollback_error}"
                ) from error
        raise


def main() -> None:
    _configure_console_encoding()
    try:
        _deploy()
    except DeploymentError as error:
        raise SystemExit(f"Master deployment FAILED: {error}") from error


if __name__ == "__main__":
    main()
