"""Outbound opener waits for speech, then a short silence. No live calls."""

from __future__ import annotations

import base64
import json
import logging

import pytest

from app.config import Settings
from app.providers import build_session_plan
from app.relay import RelayController
from tests.test_api import _allowed_time, _auth

SPEECH = bytes([0xC8, 0x48, 0xB8, 0x38]) * 200
SILENCE = b"\xff" * 160
HISS = bytes([0xD8, 0xE0, 0x58, 0x60]) * 80
CLICK = b"\x00" * 160


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _settings(provider: str) -> Settings:
    return Settings(
        voice_provider=provider,
        xai_api_key="x",
        openai_api_key="y",
        gemini_api_key="g",
        _env_file=None,
    )


def _relay(provider: str, *, wait: bool, clock: Clock | None = None) -> RelayController:
    plan = build_session_plan(_settings(provider), "be brief", "Hi, is this Sam?")
    return RelayController(
        call_id="call-1",
        call_sid="CA1",
        phone="+61400000000",
        opener="Hi, is this Sam?",
        plan=plan,
        clock=clock or Clock(),
        opener_wait_for_speech=wait,
        opener_silence_ms=500,
        opener_max_wait_ms=2500,
    )


def _start(relay: RelayController):
    return relay.on_twilio(
        {
            "event": "start",
            "streamSid": "MZ1",
            "start": {"streamSid": "MZ1", "callSid": "CA1"},
        }
    )


def _media(payload: bytes) -> dict:
    return {
        "event": "media",
        "media": {"track": "inbound", "payload": base64.b64encode(payload).decode("ascii")},
    }


def _ready(relay: RelayController, provider: str):
    if provider == "gptlive":
        return relay.on_provider(
            {
                "type": "session.started",
                "session": {
                    "id": "sess_1",
                    "model": "gpt-live-1",
                    "audio": {"output": {"voice": "ripple"}},
                },
            }
        )
    if provider == "geminilive":
        return relay.on_provider({"setupComplete": {"sessionId": "sess-live"}})
    return None


def _opener_sent(messages: list[dict], provider: str) -> bool:
    if provider == "grok":
        return any(
            isinstance(message.get("item"), dict) and message["item"].get("type") == "force_message"
            for message in messages
        )
    if provider == "openai":
        return any(message.get("type") == "response.create" for message in messages)
    if provider == "gptlive":
        return any(message.get("type") == "session.commentary.append" for message in messages)
    return any("clientContent" in message for message in messages)


def _audio_forwarded(messages: list[dict]) -> bool:
    for message in messages:
        if message.get("type") in {"input_audio_buffer.append", "session.input_audio.append"}:
            return True
        if "realtimeInput" in message:
            return True
    return False


def _speak_then_pause(relay: RelayController, clock: Clock):
    clock.now = 0.30
    during = relay.on_twilio(_media(SPEECH))
    clock.now = 0.79
    still = relay.on_twilio(_media(SILENCE))
    clock.now = 0.80
    released = relay.on_twilio(_media(SILENCE))
    return during, still, released


@pytest.mark.parametrize("provider", ["grok", "openai", "gptlive", "geminilive"])
def test_speech_then_silence_plays_the_opener(provider, caplog):
    caplog.set_level(logging.INFO, logger="tack45.relay")
    clock = Clock()
    relay = _relay(provider, wait=True, clock=clock)
    started = _start(relay)
    assert _opener_sent(started.provider_messages, provider) is False
    assert started.transcripts == []
    if provider == "gptlive":
        assert "Hi, is this Sam?" not in json.dumps(started.provider_messages)
        assert "input" in relay.plan.setup_messages[0]["session"]
    ready = _ready(relay, provider)
    if provider == "geminilive":
        assert ready is not None
        assert _opener_sent(ready.provider_messages, provider) is False
        assert ready.provider_voice == "Charon"

    early = relay.on_provider(
        {"type": "session.output_audio.delta", "delta": "AAAA"}
        if provider == "gptlive"
        else {"serverContent": {"modelTurn": {"parts": [{"text": "hello"}]}}}
        if provider == "geminilive"
        else {"type": "response.output_audio.delta", "delta": "AAAA"}
    )
    assert early.twilio_messages == []

    during, still, released = _speak_then_pause(relay, clock)
    assert _audio_forwarded(during.provider_messages) is False
    assert _audio_forwarded(still.provider_messages) is False
    assert _opener_sent(still.provider_messages, provider) is False
    assert _opener_sent(released.provider_messages, provider) is True
    assert released.transcripts[0]["text"] == "Hi, is this Sam?"
    assert relay.transcript[0]["role"] == "assistant"
    assert relay.protecting_opener is True
    assert "Opener triggered by speech after 800 ms" in caplog.text
    assert "timeout" not in caplog.text

    if provider == "gptlive":
        commentary = released.provider_messages[0]
        assert commentary["delegation_id"] is None
        assert "Hi, is this Sam?" in commentary["content"]
        assert "pause and listen" in commentary["content"]
    if provider == "geminilive":
        spoken = released.provider_messages[0]["clientContent"]["turns"][0]["parts"][0]["text"]
        assert spoken.endswith("Hi, is this Sam?")

    later = relay.on_twilio(_media(SILENCE))
    if provider == "geminilive":
        assert later.provider_messages == []
    else:
        assert _audio_forwarded(later.provider_messages) is True

    if provider == "grok":
        played = relay.on_provider({"type": "response.output_audio.delta", "delta": "BBBB"})
        assert played.twilio_messages[0]["media"]["payload"] == "BBBB"
        blocked = relay.on_provider({"type": "input_audio_buffer.speech_started"})
        assert blocked.twilio_messages == []
        assert blocked.drop_queued_audio is False


def test_timeout_plays_the_opener_when_line_noise_is_all_that_arrives(caplog):
    caplog.set_level(logging.INFO, logger="tack45.relay")
    clock = Clock()
    relay = _relay("grok", wait=True, clock=clock)
    _start(relay)
    relay.on_twilio(_media(CLICK))
    clock.now = 1.0
    relay.on_twilio(_media(HISS))
    clock.now = 2.4
    assert relay.poll_opener() is None
    clock.now = 2.5
    released = relay.poll_opener()
    assert released is not None
    assert _opener_sent(released.provider_messages, "grok") is True
    assert released.transcripts[0]["text"] == "Hi, is this Sam?"
    assert "Opener triggered by the timeout after 2500 ms" in caplog.text
    assert "triggered by speech" not in caplog.text
    assert relay.poll_opener() is None


def test_feature_off_sends_the_opener_with_the_session(caplog):
    caplog.set_level(logging.INFO, logger="tack45.relay")
    relay = _relay("grok", wait=False)
    started = _start(relay)
    assert started.provider_messages[0]["type"] == "session.update"
    assert started.provider_messages[1]["item"]["type"] == "force_message"
    assert started.provider_messages[1]["item"]["content"][0]["text"] == "Hi, is this Sam?"
    assert started.transcripts[0]["text"] == "Hi, is this Sam?"
    forwarded = relay.on_twilio(_media(SPEECH))
    assert forwarded.provider_messages == [
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(SPEECH).decode("ascii")}
    ]
    assert "Opener triggered" not in caplog.text

    live = _relay("gptlive", wait=False)
    opened = _start(live)
    greeting = opened.provider_messages[0]["session"]["input"][0]
    assert greeting["role"] == "developer"
    assert "Hi, is this Sam?" in greeting["content"][0]["text"]


def test_gptlive_hello_is_dropped_until_the_session_can_take_the_opener(caplog):
    caplog.set_level(logging.INFO, logger="tack45.relay")
    clock = Clock()
    relay = _relay("gptlive", wait=True, clock=clock)
    _start(relay)
    during, _still, released = _speak_then_pause(relay, clock)
    assert during.provider_messages == []
    assert released.provider_messages == []
    assert "Opener triggered by speech after 800 ms" in caplog.text
    ready = _ready(relay, "gptlive")
    assert ready is not None
    assert [message["type"] for message in ready.provider_messages] == ["session.commentary.append"]
    assert relay.transcript[0]["text"] == "Hi, is this Sam?"


def test_wait_off_prewarms_the_opener(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, settings = client
    settings.opener_wait_for_speech = False
    created = http.post(
        "/calls",
        json={"to": "+61280001111", "name": "Pat", "business": "Quay Books"},
        headers=_auth(),
    ).json()
    voice = http.post(
        f"/twilio/voice?call_id={created['id']}",
        data={"CallSid": created["call_sid"]},
    )
    assert voice.status_code == 200
    provider = seen["providers"][0]
    assert provider.sent[0]["type"] == "session.update"
    assert provider.sent[1]["item"]["type"] == "force_message"
    assert provider.sent[1]["item"]["content"][0]["text"] == "Hi, is this Pat?"


def test_opener_wait_defaults_do_not_block_dialling():
    settings = Settings(_env_file=None)
    assert settings.opener_wait_for_speech is True
    assert settings.opener_silence_ms == 500
    assert settings.opener_max_wait_ms == 2500
