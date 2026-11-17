"""Authenticated device registry and status-history routes."""

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import DeviceConfig
from app.core.security import require_api_token
from app.db.models import Device, DeviceStatus
from app.schemas.devices import (
    DeviceRegistrationRequest,
    DeviceResponse,
    StatusHistoryItem,
    DeviceUpdateRequest,
)
from app.services.worker_memory import cleanup_worker_memory_cluster
from app.services.cluster import cluster_nodes, sync_training_registry
from app.services.worker_operations import (
    WorkerOperationError,
    control_worker,
    get_worker_processes,
)

router = APIRouter(
    prefix="/api",
    tags=["devices"],
    dependencies=[Depends(require_api_token)],
)
WORKER_MEMORY_CLEANUP_TIMEOUT_SECONDS = 30.0


def get_session(request: Request):
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        yield session


@router.get("/devices", response_model=list[DeviceResponse])
def list_devices(session: Session = Depends(get_session)) -> list[Device]:
    return list(
        session.scalars(
            select(Device).order_by(Device.role, Device.node_id)
        ).all()
    )


@router.post("/devices/memory/cleanup")
def cleanup_worker_memory(request: Request) -> dict[str, Any]:
    workers = tuple(
        worker
        for worker in cluster_nodes(
            request.app.state.settings,
            request.app.state.session_factory,
        )
        if worker.role == "worker"
    )
    return cleanup_worker_memory_cluster(
        workers,
        timeout_seconds=WORKER_MEMORY_CLEANUP_TIMEOUT_SECONDS,
    )


def _configured_worker(request: Request, node_id: str) -> DeviceConfig:
    workers = tuple(
        worker
        for worker in cluster_nodes(
            request.app.state.settings,
            request.app.state.session_factory,
        )
        if worker.role == "worker"
    )
    worker = next((item for item in workers if item.node_id == node_id), None)
    if worker is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Worker not found")
    return worker


@router.get("/devices/workers/processes")
def worker_processes(request: Request) -> dict[str, Any]:
    workers = tuple(
        worker
        for worker in cluster_nodes(
            request.app.state.settings,
            request.app.state.session_factory,
        )
        if worker.role == "worker"
    )
    return get_worker_processes(workers)


@router.post("/devices/{node_id}/stop-training")
def stop_worker_training(node_id: str, request: Request) -> dict[str, Any]:
    try:
        return control_worker(_configured_worker(request, node_id), "stop_training")
    except WorkerOperationError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.post("/devices/{node_id}/stop-nas")
def stop_worker_nas(node_id: str, request: Request) -> dict[str, Any]:
    try:
        return control_worker(_configured_worker(request, node_id), "stop_nas")
    except WorkerOperationError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.post("/devices/{node_id}/reboot")
def reboot_worker(node_id: str, request: Request) -> dict[str, Any]:
    try:
        return control_worker(_configured_worker(request, node_id), "reboot")
    except WorkerOperationError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.post(
    "/devices",
    response_model=DeviceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_device(
    body: DeviceRegistrationRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> Device:
    if session.get(Device, body.node_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A device with this ID is already registered",
        )
    existing_endpoint = session.scalar(
        select(Device).where(Device.ip == body.ip, Device.port == body.port)
    )
    if existing_endpoint is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A device with this IP address and port is already registered",
        )
    device = Device(
        node_id=body.node_id,
        role="worker",
        hostname=body.hostname,
        ip=body.ip,
        port=body.port,
        training_port=body.training_port,
        is_managed=True,
        api_token=body.api_token,
        training_ready=False,
        online=False,
    )
    session.add(device)
    session.commit()
    session.refresh(device)
    poller = getattr(request.app.state, "poller", None)
    if poller is not None:
        await poller.poll_device(body.node_id)
        session.refresh(device)
    sync_training_registry(request.app.state.settings, request.app.state.session_factory)
    session.refresh(device)
    last_seen = device.last_seen
    if last_seen is not None and last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    await request.app.state.events.broadcast(
        {
            "type": "device_status",
            "device": {
                "node_id": device.node_id,
                "role": device.role,
                "hostname": device.hostname,
                "ip": device.ip,
                "port": device.port,
                "training_port": device.training_port,
                "is_managed": device.is_managed,
                "training_ready": device.training_ready,
                "training_error": device.training_error,
                "online": device.online,
                "last_seen": last_seen.isoformat() if last_seen else None,
                "last_error": device.last_error,
                "latest_status": device.latest_status,
            },
        }
    )
    return device


@router.delete("/devices/{node_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_device(
    node_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> None:
    device = session.get(Device, node_id)
    if device is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    if not device.is_managed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Configured cluster nodes cannot be removed from the dashboard",
        )
    session.delete(device)
    session.commit()
    sync_training_registry(request.app.state.settings, request.app.state.session_factory)
    await request.app.state.events.broadcast(
        {"type": "device_removed", "node_id": node_id}
    )


@router.patch("/devices/{node_id}", response_model=DeviceResponse)
async def update_device(
    node_id: str,
    body: DeviceUpdateRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> Device:
    device = session.get(Device, node_id)
    if device is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    if not device.is_managed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Configured cluster nodes can only be edited in the cluster configuration",
        )

    updates = body.model_dump(exclude_unset=True)
    ip = updates.get("ip", device.ip)
    port = updates.get("port", device.port)
    existing_endpoint = session.scalar(
        select(Device).where(
            Device.ip == ip,
            Device.port == port,
            Device.node_id != node_id,
        )
    )
    if existing_endpoint is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A device with this IP address and port is already registered",
        )

    for field, value in updates.items():
        setattr(device, field, value)
    device.online = False
    device.last_error = None
    session.commit()
    session.refresh(device)
    poller = getattr(request.app.state, "poller", None)
    if poller is not None:
        await poller.poll_device(node_id)
        session.refresh(device)
    sync_training_registry(request.app.state.settings, request.app.state.session_factory)
    session.refresh(device)

    last_seen = device.last_seen
    if last_seen is not None and last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    await request.app.state.events.broadcast(
        {
            "type": "device_status",
            "device": {
                "node_id": device.node_id,
                "role": device.role,
                "hostname": device.hostname,
                "ip": device.ip,
                "port": device.port,
                "training_port": device.training_port,
                "is_managed": device.is_managed,
                "training_ready": device.training_ready,
                "training_error": device.training_error,
                "online": device.online,
                "last_seen": last_seen.isoformat() if last_seen else None,
                "last_error": device.last_error,
                "latest_status": device.latest_status,
            },
        }
    )
    return device


@router.get("/devices/{node_id}", response_model=DeviceResponse)
def get_device(node_id: str, session: Session = Depends(get_session)) -> Device:
    device = session.get(Device, node_id)
    if device is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    return device


@router.get(
    "/devices/{node_id}/history",
    response_model=list[StatusHistoryItem],
)
def get_status_history(
    node_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    since_minutes: int = Query(default=60, ge=1, le=10080),
    session: Session = Depends(get_session),
) -> list[DeviceStatus]:
    if session.get(Device, node_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    statement = (
        select(DeviceStatus)
        .where(DeviceStatus.node_id == node_id, DeviceStatus.sampled_at >= cutoff)
        .order_by(DeviceStatus.sampled_at.desc(), DeviceStatus.id.desc())
        .limit(limit)
    )
    return list(session.scalars(statement).all())
