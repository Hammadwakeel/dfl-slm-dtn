"""Resolve the live worker registry and synchronize training peers."""

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import DeviceConfig, Settings
from app.db.models import Device

LOGGER = logging.getLogger(__name__)


def cluster_nodes(
    settings: Settings,
    sessions: sessionmaker[Session],
) -> tuple[DeviceConfig, ...]:
    with sessions() as session:
        managed = session.scalars(
            select(Device).where(Device.is_managed.is_(True)).order_by(Device.node_id)
        ).all()
        registered = tuple(
            DeviceConfig(
                node_id=device.node_id,
                hostname=device.hostname,
                ip=device.ip,
                role=device.role,
                port=device.port,
                token=device.api_token or "",
                training_port=device.training_port,
                is_managed=True,
            )
            for device in managed
        )
    configured_ids = {device.node_id for device in settings.nodes}
    return (*settings.nodes, *(device for device in registered if device.node_id not in configured_ids))


def sync_training_workers(
    settings: Settings,
    workers: tuple[DeviceConfig, ...],
) -> dict[str, Any]:
    training_workers = [worker for worker in workers if worker.role == "worker"]
    training_workers.sort(key=lambda worker: worker.node_id)
    expected_ids = [f"worker{index}" for index in range(1, len(training_workers) + 1)]
    if [worker.node_id for worker in training_workers] != expected_ids:
        raise ValueError("Training worker IDs must be sequential (worker1, worker2, ...)")
    if any(worker.training_port is None or len(worker.token) < 32 for worker in training_workers):
        raise ValueError("Every training worker must have a training port and API token")

    payload = {
        "workers": [
            {
                "node_id": worker.node_id,
                "base_url": worker.training_url,
                "token": worker.token,
            }
            for worker in training_workers
        ]
    }
    request = Request(
        f"{settings.master_slm_url}/cluster/workers",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {settings.master_slm_token}",
            "Content-Type": "application/json",
        },
        method="PUT",
    )
    try:
        with urlopen(request, timeout=settings.slm_timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Master training peer sync failed (HTTP {error.code}): {detail[:400]}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Master training peer sync failed: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Master training peer sync returned invalid JSON") from error
    if not isinstance(result, dict) or result.get("configured_workers") != expected_ids:
        raise RuntimeError("Master training peer sync returned an unexpected worker inventory")
    return result


def sync_training_registry(
    settings: Settings,
    sessions: sessionmaker[Session],
) -> str | None:
    workers = cluster_nodes(settings, sessions)
    active_worker_ids = {
        worker.node_id
        for worker in workers
        if worker.role == "worker"
    }
    error_message: str | None = None
    try:
        sync_training_workers(settings, workers)
    except (RuntimeError, ValueError) as error:
        error_message = str(error)
        LOGGER.warning("Dynamic training worker synchronization failed: %s", error)
    with sessions.begin() as session:
        # Configured nodes may not have a row yet (the poller creates rows later);
        # create them now so newly added workers receive the readiness result.
        for config in settings.nodes:
            if session.get(Device, config.node_id) is None:
                session.add(
                    Device(
                        node_id=config.node_id,
                        role=config.role,
                        hostname=config.hostname,
                        ip=config.ip,
                        port=config.port,
                        training_port=config.training_port,
                        online=False,
                    )
                )
        session.flush()
        devices = session.scalars(select(Device)).all()
        for device in devices:
            if device.node_id not in active_worker_ids:
                continue
            device.training_ready = error_message is None
            device.training_error = error_message
    return error_message
