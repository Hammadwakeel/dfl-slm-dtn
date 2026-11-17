"""Authenticated WebSocket endpoint for device status events."""

import json
import logging
from datetime import timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.core.security import verify_websocket_token
from app.db.models import Device
from app.services.events import EventManager

LOGGER = logging.getLogger(__name__)
router = APIRouter(tags=["live"])


@router.websocket("/ws/devices")
async def device_events(websocket: WebSocket) -> None:
    events: EventManager = websocket.app.state.events
    expected_token = websocket.app.state.settings.api_token
    await websocket.accept()
    try:
        try:
            auth_message = json.loads(await websocket.receive_text())
        except (json.JSONDecodeError, TypeError):
            auth_message = None
        supplied_token = (
            auth_message.get("token")
            if isinstance(auth_message, dict) and isinstance(auth_message.get("token"), str)
            else None
        )
        if not verify_websocket_token(supplied_token, expected_token):
            await websocket.close(code=1008, reason="Unauthorized")
            return

        await events.connect(websocket)
        with websocket.app.state.session_factory() as session:
            devices = list(
                session.scalars(select(Device).order_by(Device.role, Device.node_id)).all()
            )
            await websocket.send_json(
                {
                    "type": "device_snapshot",
                    "devices": [
                        {
                            "node_id": device.node_id,
                            "role": device.role,
                            "hostname": device.hostname,
                            "ip": device.ip,
                            "port": device.port,
                            "is_managed": device.is_managed,
                            "online": device.online,
                            "last_seen": (
                                (
                                    device.last_seen.replace(tzinfo=timezone.utc)
                                    if device.last_seen.tzinfo is None
                                    else device.last_seen
                                ).isoformat()
                                if device.last_seen
                                else None
                            ),
                            "last_error": device.last_error,
                            "latest_status": device.latest_status,
                        }
                        for device in devices
                    ],
                }
            )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await events.disconnect(websocket)
