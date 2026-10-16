"""WebSocket client for a realtime voice provider. JSON audio only."""

from __future__ import annotations

import json
import logging

import websockets

logger = logging.getLogger("tack45.realtime")


class WebSocketProvider:
    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self.url = url
        self.headers = headers
        self._ws = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            self.url,
            additional_headers=self.headers,
            max_size=8 * 1024 * 1024,
            open_timeout=15,
        )
        logger.info("Connected to realtime provider.")

    async def send(self, payload: dict) -> None:
        if self._ws is None:
            raise RuntimeError("Provider socket is not connected.")
        await self._ws.send(json.dumps(payload))

    async def recv(self) -> dict | None:
        if self._ws is None:
            return None
        message = await self._ws.recv()
        if isinstance(message, bytes):
            # Binary audio would mean the session left JSON μ-law passthrough.
            # Drop the frame rather than interpret it as another codec.
            logger.warning("Ignoring unexpected binary frame from the voice provider.")
            return {"type": "ignored_binary"}
        return json.loads(message)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


def default_provider_factory(settings, _call: dict) -> WebSocketProvider:
    from app.providers import provider_connection_target

    url, headers = provider_connection_target(settings)
    return WebSocketProvider(url, headers)
