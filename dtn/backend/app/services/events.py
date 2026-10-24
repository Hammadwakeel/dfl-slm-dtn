"""In-process WebSocket fan-out for live device status changes."""

import asyncio
import logging

from fastapi import WebSocket

LOGGER = logging.getLogger(__name__)


class EventManager:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast(self, payload: dict[str, object]) -> None:
        async with self._lock:
            clients = tuple(self._clients)
        stale: list[WebSocket] = []
        for client in clients:
            try:
                await client.send_json(payload)
            except (RuntimeError, OSError) as error:
                LOGGER.debug("Removing disconnected WebSocket client: %s", error)
                stale.append(client)
        if stale:
            async with self._lock:
                self._clients.difference_update(stale)
