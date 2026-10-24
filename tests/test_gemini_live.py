"""Gemini Live setup, resampling on the relay, and tool calls.

No socket is opened and nothing is dialled.
"""

from __future__ import annotations

import base64
import json
import struct

from app.config import Settings, missing_dial_configuration
from app.gemini_live import (
    gemini_live_events,
    gemini_live_opener,
    gemini_live_ws_url,
    realtime_audio_message,
)
from app.pcm_bridge import PcmBridge
from app.providers import build_session_plan, provider_connection_target
from app.relay import RelayController


def _settings(**overrides) -> Settings:
    base = dict(
        voice_provider="grok",
        xai_api_key="xai-test",
        openai_api_key="sk-test",
        _env_file=None,
    )
    base.update(overrides)
    return Settings(**base)


def _dial_settings(**overrides) -> Settings:
    base = dict(
        api_bearer_token="token",
        public_base_url="https://caller.example.com",
        twilio_account_sid="AC123",
        twilio_api_key_sid="SK123",
        twilio_api_key_secret="secret",
        twilio_from_number="+61290904464",
        xai_api_key="x",
        openai_api_key="y",
        _env_file=None,
    )
    base.update(overrides)
    return Settings(**base)


def _live_settings(**overrides) -> Settings:
    values = dict(voice_provider="geminilive", gemini_api_key="gem-test")
    values.update(overrides)
    return _settings(**values)


def _controller() -> RelayController:
    plan = build_session_plan(_live_settings(), "be brief", "Hi, is this Sam?")
    return RelayController(
        call_id="call-live",
        call_sid="CA-live",
        phone="+61400000000",
        opener="Hi, is this Sam?",
        plan=plan,
        hangup_grace_seconds=5,
    )


def _start(relay: RelayController):
    return relay.on_twilio(
        {
            "event": "start",
            "start": {"streamSid": "MZ-live", "callSid": "CA-live"},
            "streamSid": "MZ-live",
        }
    )


def _pcm24_b64(samples: list[int]) -> str:
    raw = struct.pack("<" + "h" * len(samples), *samples)
    return base64.b64encode(raw).decode("ascii")


def _release_opener(relay: RelayController) -> None:
    """A turn with no audio ends opener protection without touching the resampler."""
    _start(relay)
    relay.on_provider({"setupComplete": {}})
    relay.on_provider({"serverContent": {"turnComplete": True}})
    assert relay.protecting_opener is False


def test_setup_uses_gemini_3_8_live_charon_and_keeps_the_accent_out_of_the_greeting():
    plan = build_session_plan(_live_settings(), "be brief", "Hi, is this Sam?")
    assert plan.provider == "geminilive"
    assert len(plan.setup_messages) == 1
    setup = plan.setup_messages[0]["setup"]
    assert setup["model"] == "models/gemini-3.8-live"
    speech = setup["generationConfig"]["speechConfig"]
    assert speech["languageCode"] == "en-AU"
    assert speech["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Charon"
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert "thinkingConfig" not in setup["generationConfig"]
    instruction = setup["systemInstruction"]["parts"][0]["text"]
    assert instruction.startswith("be brief")
    assert "Australian English accent, relaxed and friendly" in instruction
    assert "en-AU" in instruction
    names = {item["name"] for item in setup["tools"][0]["functionDeclarations"]}
    assert names == {"end_call", "record_opt_out"}
    assert {item["behavior"] for item in setup["tools"][0]["functionDeclarations"]} == {"BLOCKING"}
    assert setup["inputAudioTranscription"]["languageCodes"] == ["en-AU"]
    assert setup["outputAudioTranscription"]["languageCodes"] == ["en-AU"]
    assert setup["realtimeInputConfig"]["activityHandling"] == "START_OF_ACTIVITY_INTERRUPTS"
    dumped = json.dumps(plan.setup_messages)
    assert "Hi, is this Sam?" not in dumped

    opener = gemini_live_opener("Hi, is this Sam?")
    spoken = opener["clientContent"]["turns"][0]["parts"][0]["text"]
    assert "Hi, is this Sam?" in spoken
    assert "Australian" not in spoken
    assert opener["clientContent"]["turnComplete"] is True


def test_connection_puts_the_key_on_the_live_url_and_not_on_grok():
    url, headers = provider_connection_target(
        _live_settings(gemini_api_key="gem-live-key", geminilive_model="gemini-3.8-live")
    )
    assert url.startswith(
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?"
    )
    assert "key=gem-live-key" in url
    assert "model=" not in url.split("?", 1)[-1]
    assert headers == {}
    assert gemini_live_ws_url("wss://example.test/live?x=1", "a b") == (
        "wss://example.test/live?x=1&key=a+b"
    )
    grok_url, _grok_headers = provider_connection_target(_settings())
    assert "generativelanguage" not in grok_url


def test_geminilive_needs_the_gemini_key_and_not_deepgram():
    assert missing_dial_configuration(_dial_settings(voice_provider="geminilive", gemini_api_key="")) == [
        "GEMINI_API_KEY"
    ]
    ready = _dial_settings(voice_provider="geminilive", gemini_api_key="gem-test", deepgram_api_key="")
    assert missing_dial_configuration(ready) == []
    assert "DEEPGRAM_API_KEY" not in missing_dial_configuration(ready)
    cascade = _dial_settings(voice_provider="gemini", gemini_api_key="gem-test", deepgram_api_key="")
    assert missing_dial_configuration(cascade) == ["DEEPGRAM_API_KEY"]


def test_caller_audio_is_16k_pcm_only_after_the_opener_and_grok_stays_mulaw():
    relay = _controller()
    _start(relay)
    early = relay.on_twilio(
        {"event": "media", "media": {"track": "inbound", "payload": base64.b64encode(b"\xff" * 160).decode()}}
    )
    assert early.provider_messages == []
    ready = relay.on_provider({"setupComplete": {}})
    assert ready.provider_messages[0]["clientContent"]["turns"][0]["parts"][0]["text"].endswith(
        "Hi, is this Sam?"
    )
    assert ready.provider_voice == "Charon"
    assert ready.provider_session_id == "models/gemini-3.8-live"
    during = relay.on_twilio(
        {"event": "media", "media": {"track": "inbound", "payload": base64.b64encode(b"\xff" * 160).decode()}}
    )
    assert during.provider_messages == []

    relay = _controller()
    _release_opener(relay)
    payload = base64.b64encode(b"\xff" * 160).decode("ascii")
    sent = relay.on_twilio({"event": "media", "media": {"track": "inbound", "payload": payload}})
    message = sent.provider_messages[0]
    assert message["realtimeInput"]["audio"]["mimeType"] == "audio/pcm;rate=16000"
    pcm = base64.b64decode(message["realtimeInput"]["audio"]["data"])
    assert abs(len(pcm) // 2 - 320) <= 2
    direct = realtime_audio_message(PcmBridge(), payload)
    assert direct is not None
    assert base64.b64decode(direct["realtimeInput"]["audio"]["data"]) == pcm


def test_model_audio_is_resampled_to_mulaw_and_barge_in_clears_twilio():
    relay = _controller()
    _release_opener(relay)
    samples = [4000, -4000, 0] * 80
    played = relay.on_provider(
        {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {
                            "thought": True,
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=24000",
                                "data": _pcm24_b64([30000] * 9),
                            },
                        },
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=24000",
                                "data": _pcm24_b64(samples),
                            },
                        },
                    ]
                }
            }
        }
    )
    assert len(played.twilio_messages) == 1
    mulaw = base64.b64decode(played.twilio_messages[0]["media"]["payload"])
    assert played.twilio_messages[0]["event"] == "media"
    assert len(mulaw) == len(samples) // 3

    barge = relay.on_provider({"serverContent": {"interrupted": True}})
    assert barge.drop_queued_audio is True
    assert barge.twilio_messages == [{"event": "clear", "streamSid": "MZ-live"}]
    dropped = relay.on_provider(
        {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=24000",
                                "data": _pcm24_b64([1000] * 24),
                            }
                        }
                    ]
                }
            }
        }
    )
    assert dropped.twilio_messages == []


def test_opener_barge_in_does_not_clear_twilio():
    relay = _controller()
    _start(relay)
    relay.on_provider({"setupComplete": {}})
    relay.on_provider(
        {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=24000",
                                "data": _pcm24_b64([0] * 48),
                            }
                        }
                    ]
                }
            }
        }
    )
    during = relay.on_provider({"serverContent": {"interrupted": True}})
    assert during.twilio_messages == []
    assert during.drop_queued_audio is False


def test_transcript_fragments_are_committed_on_the_turn_boundary():
    relay = _controller()
    _release_opener(relay)
    relay.on_provider({"serverContent": {"inputTranscription": {"text": "Hello"}}})
    relay.on_provider({"serverContent": {"inputTranscription": {"text": " there"}}})
    done = relay.on_provider(
        {
            "serverContent": {
                "outputTranscription": {"text": "Good"},
            }
        }
    )
    # The caller's line is flushed when the model starts speaking.
    assert any(line["role"] == "caller" and line["text"] == "Hello there" for line in done.transcripts)
    finished = relay.on_provider(
        {"serverContent": {"outputTranscription": {"text": " day."}, "turnComplete": True}}
    )
    assert any(line["role"] == "assistant" and line["text"] == "Good day." for line in finished.transcripts)


def test_opt_out_is_immediate_and_a_cancellation_does_not_undo_it():
    relay = _controller()
    _release_opener(relay)
    result = relay.on_provider(
        {
            "toolCall": {
                "functionCalls": [
                    {"id": "opt-1", "name": "record_opt_out", "args": {"reason": "stop calling"}}
                ]
            }
        }
    )
    assert result.opt_out_reason == "stop calling"
    assert relay.opted_out is True
    response = result.provider_messages[0]["toolResponse"]["functionResponses"][0]
    assert response["id"] == "opt-1"
    assert response["name"] == "record_opt_out"
    assert response["response"]["ok"] is True
    assert "response.create" not in json.dumps(result.provider_messages)
    relay.on_provider({"toolCallCancellation": {"ids": ["opt-1"]}})
    assert relay.opted_out is True
    assert relay.opt_out_reason == "stop calling"
    assert relay.end_reason == "opt_out"


def test_end_call_waits_for_playback_and_can_be_cancelled_before_hangup():
    relay = _controller()
    _release_opener(relay)
    relay.on_provider(
        {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=24000",
                                "data": _pcm24_b64([0] * 2400),
                            }
                        }
                    ]
                }
            }
        }
    )
    result = relay.on_provider(
        {
            "toolCall": {
                "functionCalls": [
                    {
                        "id": "end-1",
                        "name": "end_call",
                        "args": json.dumps({"reason": "finished"}),
                    }
                ]
            }
        }
    )
    assert result.hangup is False
    assert result.provider_messages[0]["toolResponse"]["functionResponses"][0]["name"] == "end_call"
    assert result.twilio_messages[0]["event"] == "mark"
    assert result.twilio_messages[0]["mark"]["name"].startswith("hangup-")
    assert "response.create" not in json.dumps(result.provider_messages)
    relay.on_provider({"toolCallCancellation": {"ids": ["end-1"]}})
    assert relay.hangup_requested is False
    assert relay.awaiting_mark is None
    assert relay.end_reason == "completed"


def test_events_skip_thoughts_and_ignore_unrelated_messages():
    bridge = PcmBridge()
    events = gemini_live_events({"usageMetadata": {"totalTokenCount": 3}}, bridge)
    assert events == []
    cancelled = gemini_live_events(
        {"toolCallCancellation": {"ids": ["a", "b"]}},
        bridge,
    )
    assert [event.call_id for event in cancelled] == ["a", "b"]
    assert [event.kind for event in cancelled] == ["tool_cancelled", "tool_cancelled"]
