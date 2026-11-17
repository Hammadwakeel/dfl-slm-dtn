"""Back up, install, and verify dataset-aware training code on one worker."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402
from app.services.worker_shards import (  # noqa: E402
    get_worker_dataset_metadata,
    get_worker_training_status,
    WorkerShardError,
)


WORKER_HOSTS = {
    "worker1": "192.168.11.123",
    "worker2": "192.168.11.14",
    "worker3": "192.168.11.13",
    "worker4": "192.168.11.75",
    "worker5": "192.168.11.76",
}
WORKER_USERS = {
    "worker1": "hammad",
    "worker2": "hammad",
    "worker3": "hammad",
    "worker4": "jetson5",
    "worker5": "hammad",
}
REMOTE_FILES = {
    "training_service.py": (
        ROOT / "worker" / "training_service.py",
        "/opt/dfl-slm-dtn/worker/training_service.py",
    ),
    "memory_cleanup_helper.py": (
        ROOT / "worker" / "memory_cleanup_helper.py",
        "/opt/dfl-slm-dtn/worker/memory_cleanup_helper.py",
    ),
    "memory_cleanup.service": (
        ROOT / "deploy" / "dfl-slm-memory-cleanup.service",
        "/etc/systemd/system/dfl-slm-memory-cleanup.service",
    ),
    "nas_search.py": (
        ROOT / "worker" / "nas_search.py",
        "/opt/dfl-slm-dtn/worker/nas_search.py",
    ),
    "nas.py": (
        ROOT / "common" / "nas.py",
        "/opt/dfl-slm-dtn/common/nas.py",
    ),
    "dataset_nas.py": (
        ROOT / "common" / "dataset_nas.py",
        "/opt/dfl-slm-dtn/common/dataset_nas.py",
    ),
}


class WorkerDeploymentError(RuntimeError):
    pass


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _ssh(user: str, host: str, key: Path, command: str, timeout: int = 60) -> str:
    process = subprocess.run(
        [
            "ssh",
            "-i",
            str(key),
            "-o",
            "IdentitiesOnly=yes",
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
        raise WorkerDeploymentError(
            f"SSH to {host} failed (exit {process.returncode}): "
            f"{process.stderr.strip()[-500:]}"
        )
    return process.stdout


def _remote_backup(
    worker_id: str,
    host: str,
    user: str,
    key: Path,
    stamp: str,
) -> tuple[dict[str, str], list[str]]:
    backups = {
        name: f"{remote_path}.bak-{stamp}"
        for name, (_, remote_path) in REMOTE_FILES.items()
    }
    commands = ["set -eu"]
    for name, (_, remote_path) in REMOTE_FILES.items():
        backup = backups[name]
        commands.append(
            f"if sudo -n test -f {remote_path}; then "
            f"sudo -n cp -a -- {remote_path} {backup}; "
            f"sudo -n test -f {backup}; "
            f"printf 'BACKUP {name} %s\\n' {backup}; "
            f"else printf 'ABSENT {name}\\n'; fi"
        )
    command = "; ".join(commands)
    output = _ssh(user, host, key, command)
    observed_backups: set[str] = set()
    absent: list[str] = []
    for line in output.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[0] == "BACKUP":
            observed_backups.add(parts[1])
        elif len(parts) == 2 and parts[0] == "ABSENT":
            absent.append(parts[1])
    if observed_backups | set(absent) != set(REMOTE_FILES):
        raise WorkerDeploymentError(f"{worker_id} worker code backups were not verified")
    return (
        {name: backup for name, backup in backups.items() if name in observed_backups},
        absent,
    )


def _rollback(
    host: str,
    user: str,
    key: Path,
    backups: dict[str, str],
    absent: list[str],
) -> None:
    commands = ["set -eu"]
    for name, (_, remote_path) in REMOTE_FILES.items():
        if name in absent:
            commands.append(f"sudo -n rm -f -- {remote_path}")
        else:
            commands.append(
                f"sudo -n cp -a -- {backups[name]} {remote_path}"
            )
    if "memory_cleanup.service" in absent:
        commands.extend(
            [
                "sudo -n systemctl disable --now dfl-slm-memory-cleanup.service "
                ">/dev/null 2>&1 || true",
                "sudo -n rm -f -- /etc/dfl-slm-memory-cleanup.env",
            ]
        )
    commands.append("sudo -n systemctl daemon-reload")
    if "memory_cleanup.service" not in absent:
        commands.append("sudo -n systemctl restart dfl-slm-memory-cleanup.service")
    commands.append("sudo -n systemctl restart dfl-slm-worker-training.service")
    _ssh(user, host, key, "; ".join(commands), timeout=90)


def _install(
    worker_id: str,
    host: str,
    user: str,
    key: Path,
    backups: dict[str, str],
    absent: list[str],
) -> None:
    stages = {
        name: f"/tmp/dfl-worker-mnist-{uuid.uuid4().hex}-{name}"
        for name in REMOTE_FILES
    }
    try:
        for name, (source, _) in REMOTE_FILES.items():
            scp = subprocess.run(
                [
                    "scp",
                    "-i",
                    str(key),
                    "-o",
                    "IdentitiesOnly=yes",
                    "-o",
                    "BatchMode=yes",
                    str(source),
                    f"{user}@{host}:{stages[name]}",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            if scp.returncode:
                raise WorkerDeploymentError(
                    f"Could not stage {name} on {worker_id}: {scp.stderr.strip()[-500:]}"
                )
        commands = ["set -eu"]
        for name, (_, remote_path) in REMOTE_FILES.items():
            commands.append(
                f"sudo -n install -o root -g root -m 0644 {stages[name]} {remote_path}; "
                f"rm -f -- {stages[name]}"
            )
        commands.extend(
            [
                "sudo -n sh -c 'token=$(sed -n \"s/^DFL_API_TOKEN=//p\" "
                "/etc/dfl-slm-worker-training.env); "
                "test ${#token} -ge 32; umask 077; "
                "printf \"DFL_MEMORY_CLEANUP_TOKEN=%s\\n\" \"$token\" "
                "> /etc/dfl-slm-memory-cleanup.env; "
                "chown root:root /etc/dfl-slm-memory-cleanup.env; "
                "chmod 600 /etc/dfl-slm-memory-cleanup.env'",
                "sudo -n systemctl daemon-reload",
                "sudo -n systemctl restart dfl-slm-memory-cleanup.service",
                "for attempt in $(seq 1 10); do "
                "if sudo -n systemctl is-active --quiet dfl-slm-memory-cleanup.service "
                "&& ss -lnt '( sport = :5005 )' | grep -q LISTEN; then break; fi; "
                "sleep 1; done",
                "sudo -n systemctl is-active --quiet dfl-slm-memory-cleanup.service",
                "ss -lnt '( sport = :5005 )' | grep -q LISTEN",
                "sudo -n systemctl restart dfl-slm-worker-training.service",
                "for attempt in $(seq 1 30); do "
                "if sudo -n systemctl is-active --quiet dfl-slm-worker-training.service "
                "&& ss -lnt '( sport = :5004 )' | grep -q LISTEN; then break; fi; "
                "sleep 1; done",
                "sudo -n systemctl is-active --quiet dfl-slm-worker-training.service",
                "ss -lnt '( sport = :5004 )' | grep -q LISTEN",
            ]
        )
        for _, remote_path in REMOTE_FILES.values():
            commands.append(f"sha256sum {remote_path}")
        output = _ssh(user, host, key, "; ".join(commands), timeout=90)
    except (WorkerDeploymentError, subprocess.TimeoutExpired) as error:
        try:
            _rollback(host, user, key, backups, absent)
        except (WorkerDeploymentError, subprocess.TimeoutExpired) as rollback_error:
            raise WorkerDeploymentError(
                f"{error}; rollback failed: {rollback_error}"
            ) from error
        raise
    observed: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            for name, (_, remote_path) in REMOTE_FILES.items():
                if parts[1] == remote_path:
                    observed[name] = parts[0]
    mismatches = []
    for name, (source, _) in REMOTE_FILES.items():
        local_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if observed.get(name) != local_hash:
            mismatches.append(name)
        print(
            f"{worker_id} {name}: PC sha256={local_hash}; "
            f"Jetson sha256={observed.get(name, 'MISSING')}"
        )
    if mismatches:
        _rollback(host, user, key, backups, absent)
        raise WorkerDeploymentError(
            f"{worker_id} deployed worker code hash mismatch: {', '.join(mismatches)}"
        )


def main() -> None:
    _configure_console_encoding()
    parser = argparse.ArgumentParser()
    parser.add_argument("worker_id", choices=sorted(WORKER_HOSTS))
    parser.add_argument("--dataset-name", default="cifar_10k_demo")
    args = parser.parse_args()
    settings = Settings.from_env()
    worker = next(
        (node for node in settings.nodes if node.node_id == args.worker_id),
        None,
    )
    if worker is None or worker.role != "worker":
        raise SystemExit(f"{args.worker_id} is not a configured worker")
    status = get_worker_training_status(worker)
    if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
        raise SystemExit(f"{args.worker_id} is not idle; deployment refused")
    dataset = get_worker_dataset_metadata(worker, args.dataset_name)
    print(
        f"{args.worker_id}: authenticated status OK; idle; "
        f"{args.dataset_name} sample_count={dataset.get('sample_count')}"
    )
    user = os.environ.get(
        f"{args.worker_id.upper()}_SSH_USER",
        os.environ.get("M4_SSH_USER", WORKER_USERS[args.worker_id]),
    )
    key = Path(
        os.environ.get(
            "M4_SSH_KEY",
            str(Path.home() / ".ssh" / "id_ed25519"),
        )
    )
    if not key.is_file():
        raise SystemExit(f"SSH key file is unavailable: {key}")
    host = WORKER_HOSTS[args.worker_id]
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backups, absent = _remote_backup(args.worker_id, host, user, key, stamp)
    for name, backup in backups.items():
        print(f"{args.worker_id} {name}: backup={backup}")
    _install(args.worker_id, host, user, key, backups, absent)

    deadline = time.monotonic() + 60
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            ready = get_worker_training_status(worker)
            if ready.get("active_run_id") is None and ready.get("scheduled_run_id") is None:
                print(
                    f"{args.worker_id}: authenticated status OK; "
                    "service ready and idle"
                )
                return
        except WorkerShardError as error:
            last_error = error
        time.sleep(2)
    _rollback(host, user, key, backups, absent)
    raise WorkerDeploymentError(
        f"{args.worker_id} did not return authenticated status after restart: {last_error}"
    )


if __name__ == "__main__":
    main()
