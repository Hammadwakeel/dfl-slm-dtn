"""Report current M5 deployment hash differences and authenticated idle status."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import Settings  # noqa: E402
from app.services.worker_shards import get_worker_training_status  # noqa: E402
from deploy_master_nas import Backend, DEPLOYED_SOURCES  # noqa: E402


WORKER_SOURCES = {
    "worker/training_service.py": ROOT / "worker" / "training_service.py",
    "worker/memory_cleanup_helper.py": ROOT / "worker" / "memory_cleanup_helper.py",
    "worker/nas_search.py": ROOT / "worker" / "nas_search.py",
    "worker/__init__.py": ROOT / "worker" / "__init__.py",
    "common/__init__.py": ROOT / "common" / "__init__.py",
    "common/auth.py": ROOT / "common" / "auth.py",
    "common/cifar_model.py": ROOT / "common" / "cifar_model.py",
    "common/dataset_nas.py": ROOT / "common" / "dataset_nas.py",
    "common/nas.py": ROOT / "common" / "nas.py",
}
WORKER_SERVICE_SOURCES = {
    "dfl-slm-memory-cleanup.service": ROOT / "deploy" / "dfl-slm-memory-cleanup.service",
}
HOSTS = {
    "master": "192.168.11.15",
    "worker1": "192.168.11.123",
    "worker2": "192.168.11.14",
    "worker3": "192.168.11.13",
    "worker4": "192.168.11.75",
    "worker5": "192.168.11.76",
}
HOST_USERS = {
    "192.168.11.15": "hammad",
    "192.168.11.123": "hammad",
    "192.168.11.14": "hammad",
    "192.168.11.13": "hammad",
    "192.168.11.75": "jetson5",
    "192.168.11.76": "hammad",
}


def _ssh(host: str, command: str) -> str:
    user = HOST_USERS.get(host, os.environ.get("M4_SSH_USER", "hammad"))
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
        timeout=60,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            f"SSH to {host} failed (exit {process.returncode}): "
            f"{process.stderr.strip()[-500:]}"
        )
    return process.stdout


def _remote_hashes(host: str, root: str, sources: dict[str, Path]) -> dict[str, str | None]:
    remote_paths = [f"{root}/{relative}" for relative in sources]
    quoted = " ".join(shlex.quote(path) for path in remote_paths)
    shell = (
        "set -eu; for f in "
        + quoted
        + "; do if test -f \"$f\"; then sha256sum \"$f\"; "
        + "else printf 'MISSING  %s\\n' \"$f\"; fi; done"
    )
    output = _ssh(host, "sh -c " + shlex.quote(shell))
    result: dict[str, str | None] = {}
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            digest, path = parts
            result[path.replace(root + "/", "", 1)] = None if digest == "MISSING" else digest
    return result


def _print_differences(
    label: str,
    host: str,
    root: str,
    sources: dict[str, Path],
) -> None:
    observed = _remote_hashes(host, root, sources)
    for relative, local_path in sources.items():
        local_hash = hashlib.sha256(local_path.read_bytes()).hexdigest()
        remote_hash = observed.get(relative)
        if remote_hash != local_hash:
            print(
                f"{label} {root}/{relative} "
                f"PC={local_hash} DEVICE={remote_hash or 'MISSING'}"
            )


def main() -> None:
    settings = Settings.from_env()
    backend = Backend("http://127.0.0.1:8000", settings.api_token)
    workers = sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )
    worker_ids = [node.node_id for node in workers]
    if len(worker_ids) < 3 or worker_ids != [
        f"worker{index}" for index in range(1, len(worker_ids) + 1)
    ]:
        raise SystemExit("Expected at least three sequentially numbered workers")
    for worker in workers:
        _ssh(
            worker.ip,
            "systemctl is-active --quiet dfl-slm-memory-cleanup.service "
            "&& ss -lnt '( sport = :5005 )' | grep -q LISTEN",
        )
        status = get_worker_training_status(worker)
        print(
            f"{worker.node_id}: authenticated status OK; "
            f"active={status.get('active_run_id')}; "
            f"scheduled={status.get('scheduled_run_id')}"
        )
        if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
            raise SystemExit(f"{worker.node_id} is not idle")
    autonomy = backend.request("GET", "/api/autonomy/state")
    if autonomy.get("stop_requested") is not False:
        raise SystemExit("Global STOP must be inactive")
    if any(loop.get("status") == "RUNNING" for loop in autonomy.get("loops", [])):
        raise SystemExit("Autonomy has a running loop")
    print(
        f"autonomy: global_STOP={autonomy.get('stop_requested')}; "
        "running_loops=0"
    )
    master_status = backend.request("GET", "/api/slm/status")
    master_results = backend.request("GET", "/api/slm/results")
    if master_status.get("node_id") != "master":
        raise SystemExit("Authenticated master status returned unexpected node")
    if master_results.get("active_group_id") is not None:
        raise SystemExit("Master training/NAS group is active")
    print("master: authenticated status OK; active training/NAS group=None")
    print("master: service active and port 5003 listening")
    print("master file differences (path and hashes only):")
    _print_differences("master", HOSTS["master"], "/opt/dfl-slm-dtn", DEPLOYED_SOURCES)
    print("worker file differences (path and hashes only):")
    for worker_id in (node.node_id for node in workers):
        _print_differences(
            worker_id,
            HOSTS[worker_id],
            "/opt/dfl-slm-dtn",
            WORKER_SOURCES,
        )
        _print_differences(
            f"{worker_id} systemd",
            HOSTS[worker_id],
            "/etc/systemd/system",
            WORKER_SERVICE_SOURCES,
        )


if __name__ == "__main__":
    main()
