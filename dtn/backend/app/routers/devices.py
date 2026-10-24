"""Authenticated device registry and status-history routes."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import require_api_token
from app.db.models import Device, DeviceStatus
from app.schemas.devices import DeviceResponse, StatusHistoryItem

router = APIRouter(
    prefix="/api",
    tags=["devices"],
    dependencies=[Depends(require_api_token)],
)


def get_session(request: Request):
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        yield session


@router.get("/devices", response_model=list[DeviceResponse])
def list_devices(session: Session = Depends(get_session)) -> list[Device]:
    return list(session.scalars(select(Device).order_by(Device.role, Device.node_id)).all())


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
