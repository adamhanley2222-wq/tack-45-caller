"""WebSocket client for a realtime voice provider.

Grok, OpenAI, and GPT-Live send JSON text frames. Gemini Live sends the same
JSON (setupComplete, audio, tool calls) in binary frames. Those are decoded
only when ``binary_json`` is set, which is ``VOICE_PROVIDER=geminilive``.
"""

from __future__ import annotations

import json
import logging

import websockets

logger = logging.getLogger("tack45.realtime")


class WebSocketProvider:
    def __init__(self, url: str, headers: dict[str, str], *, binary_json: bool = False) -> None:
        self.url = url
        self.headers = headers
        self.binary_json = binary_json
        self._ws = None
        self._logged_binary_json = False

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
        if isinstance(message, (bytes, bytearray, memoryview)):
            if self.binary_json:
                return self._binary_json(bytes(message))
            # Binary audio would mean the session left JSON text. Drop the
            # frame rather than play another codec on the Twilio leg.
            logger.warning("Ignoring unexpected binary frame from the voice provider.")
            return {"type": "ignored_binary"}
        return json.loads(message)

    def _binary_json(self, message: bytes) -> dict:
        """Gemini Live puts JSON in binary WebSocket frames."""
        try:
            text = message.decode("utf-8")
            payload = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.error(
                "Gemini Live binary frame was not UTF-8 JSON (%s bytes): %s",
                len(message),
                exc,
            )
            return {"error": {"message": "binary frame was not UTF-8 JSON"}}
        if not isinstance(payload, dict):
            logger.error("Gemini Live binary frame JSON was not an object.")
            return {"error": {"message": "binary frame JSON was not an object"}}
        if not self._logged_binary_json:
            self._logged_binary_json = True
            logger.info("Gemini Live binary JSON frame keys=%s", sorted(payload))
        return payload

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


def default_provider_factory(settings, _call: dict) -> WebSocketProvider:
    from app.providers import provider_connection_target

    url, headers = provider_connection_target(settings)
    provider = settings.voice_provider.strip().lower()
    return WebSocketProvider(url, headers, binary_json=provider == "geminilive")
