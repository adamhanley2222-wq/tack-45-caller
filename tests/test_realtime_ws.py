"""Provider socket framing.

Gemini Live sends JSON in binary frames. The other providers stay on text.
"""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.realtime_ws import WebSocketProvider, default_provider_factory


class _Frames:
    def __init__(self, frames: list) -> None:
        self.frames = list(frames)

    async def recv(self):
        return self.frames.pop(0)


@pytest.mark.asyncio
async def test_geminilive_decodes_binary_utf8_json(caplog):
    caplog.set_level("INFO", logger="tack45.realtime")
    provider = WebSocketProvider("wss://example.test/live", {}, binary_json=True)
    setup = {"setupComplete": {"sessionId": "sess-live-1"}}
    audio = {
        "serverContent": {
            "modelTurn": {
                "parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": "AAAA"}}]
            }
        }
    }
    provider._ws = _Frames(
        [
            json.dumps(setup).encode("utf-8"),
            bytearray(json.dumps(audio).encode("utf-8")),
        ]
    )
    assert await provider.recv() == setup
    assert await provider.recv() == audio
    assert "setupComplete" in caplog.text
    assert "Ignoring unexpected binary frame" not in caplog.text


@pytest.mark.asyncio
async def test_binary_frames_stay_ignored_for_other_providers(caplog):
    caplog.set_level("WARNING", logger="tack45.realtime")
    provider = WebSocketProvider("wss://example.test/realtime", {})
    provider._ws = _Frames([b'{"setupComplete": {"sessionId": "nope"}}'])
    assert await provider.recv() == {"type": "ignored_binary"}
    assert "Ignoring unexpected binary frame" in caplog.text


@pytest.mark.asyncio
async def test_geminilive_binary_that_is_not_json_is_an_error(caplog):
    caplog.set_level("ERROR", logger="tack45.realtime")
    provider = WebSocketProvider("wss://example.test/live", {}, binary_json=True)
    provider._ws = _Frames([b"\xff\xfe not json", b"[1, 2]"])
    first = await provider.recv()
    second = await provider.recv()
    assert first["error"]["message"] == "binary frame was not UTF-8 JSON"
    assert second["error"]["message"] == "binary frame JSON was not an object"
    assert "not UTF-8 JSON" in caplog.text


def test_factory_enables_binary_json_only_for_geminilive():
    live = default_provider_factory(
        Settings(_env_file=None, voice_provider="geminilive", gemini_api_key="gem-test"),
        {},
    )
    assert live.binary_json is True
    grok = default_provider_factory(
        Settings(_env_file=None, voice_provider="grok", xai_api_key="xai-test"),
        {},
    )
    assert grok.binary_json is False
