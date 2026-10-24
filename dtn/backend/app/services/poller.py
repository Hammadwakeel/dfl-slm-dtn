"""Background polling and persistence for real Jetson status."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from common.api_client import ApiClient, ApiClientError, NodeEndpoint
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import DeviceConfig, Settings
from app.db.models import Device, DeviceStatus
from app.schemas.devices import StatusPayload
from app.services.events import EventManager

LOGGER = logging.getLogger(__name__)


class DevicePoller:
    def __init__(
        self,
        settings: Settings,
        sessions: sessionmaker[Session],
        events: EventManager,
    ) -> None:
        self._settings = settings
        self._sessions = sessions
        self._events = events
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._ensure_registry()
        self._task = asyncio.create_task(self._run(), name="jetson-device-poller")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def poll_once(self) -> None:
        await asyncio.gather(*(self._poll_device(device) for device in self._settings.nodes))

    async def _run(self) -> None:
        while not self._stop.is_set():
            await self.poll_once()
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self._settings.poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

    async def _poll_device(self, config: DeviceConfig) -> None:
        endpoint = NodeEndpoint(
            node_id=config.node_id,
            base_url=config.base_url,
            token=config.token,
        )
        try:
            payload = await asyncio.to_thread(
                ApiClient(endpoint, self._settings.request_timeout_seconds).get_status
            )
            status = StatusPayload.model_validate(payload)
            if status.node_id != config.node_id or status.role != config.role:
                raise ValueError("Jetson status identity did not match the device registry")
        except (ApiClientError, ValueError) as error:
            LOGGER.warning("Status poll failed for %s: %s", config.node_id, error)
            was_online, previous_status, previous_last_seen = await asyncio.to_thread(
                self._mark_offline,
                config,
                str(error),
            )
            if was_online:
                await self._events.broadcast(
                    {
                        "type": "device_status",
                        "device": self._device_event(
                            config,
                            False,
                            previous_last_seen,
                            str(error)[:512],
                            previous_status,
                        ),
                    }
                )
            return

        now = datetime.now(timezone.utc)
        normalized_payload = status.model_dump(mode="json")
        await asyncio.to_thread(self._store_status, config, status, normalized_payload, now)
        await self._events.broadcast(
            {
                "type": "device_status",
                "device": self._device_event(config, True, now, None, normalized_payload),
            }
        )

    def _ensure_registry(self) -> None:
        with self._sessions() as session:
            for config in self._settings.nodes:
                device = session.get(Device, config.node_id)
                if device is None:
                    device = Device(
                        node_id=config.node_id,
                        role=config.role,
                        hostname=config.hostname,
                        ip=config.ip,
                        port=config.port,
                        online=False,
                    )
                    session.add(device)
                else:
                    device.role = config.role
                    device.hostname = config.hostname
                    device.ip = config.ip
                    device.port = config.port
            session.commit()

    def _store_status(
        self,
        config: DeviceConfig,
        status: StatusPayload,
        payload: dict[str, Any],
        now: datetime,
    ) -> None:
        with self._sessions() as session:
            device = session.get(Device, config.node_id)
            if device is None:
                raise RuntimeError(f"Device registry entry disappeared: {config.node_id}")
            device.online = True
            device.last_seen = now
            device.last_error = None
            device.latest_status = payload
            session.add(
                DeviceStatus(
                    node_id=config.node_id,
                    sampled_at=status.sampled_at,
                    cpu_utilization_percent=status.cpu_utilization_percent,
                    gpu_utilization_percent=status.gpu_utilization_percent,
                    memory=status.memory.model_dump(mode="json"),
                    temperatures_celsius=status.temperatures_celsius,
                    payload=payload,
                )
            )
            self._trim_history(session, config.node_id)
            session.commit()

    def _mark_offline(
        self,
        config: DeviceConfig,
        error: str,
    ) -> tuple[bool, dict[str, Any] | None, datetime | None]:
        with self._sessions() as session:
            device = session.get(Device, config.node_id)
            if device is None:
                return False, None, None
            was_online = device.online
            previous_status = device.latest_status
            previous_last_seen = device.last_seen
            device.online = False
            device.last_error = error[:512]
            session.commit()
        return was_online, previous_status, previous_last_seen

    def _trim_history(self, session: Session, node_id: str) -> None:
        excess_ids = session.scalars(
            select(DeviceStatus.id)
            .where(DeviceStatus.node_id == node_id)
            .order_by(DeviceStatus.sampled_at.desc(), DeviceStatus.id.desc())
            .offset(self._settings.history_retention_count)
        ).all()
        if excess_ids:
            session.execute(delete(DeviceStatus).where(DeviceStatus.id.in_(excess_ids)))

    @staticmethod
    def _device_event(
        config: DeviceConfig,
        online: bool,
        last_seen: datetime | None,
        error: str | None,
        latest_status: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if last_seen is not None and last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=timezone.utc)
        return {
            "node_id": config.node_id,
            "role": config.role,
            "hostname": config.hostname,
            "ip": config.ip,
            "port": config.port,
            "online": online,
            "last_seen": last_seen.isoformat() if last_seen else None,
            "last_error": error,
            "latest_status": latest_status,
        }
