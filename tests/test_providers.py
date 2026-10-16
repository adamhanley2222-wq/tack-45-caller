import json

from app.config import Settings
from app.prompts import render_instructions
from app.providers import build_session_plan, normalize_event, realtime_ws_url


def _settings(**overrides) -> Settings:
    base = dict(
        voice_provider="grok",
        xai_voice="eve",
        xai_api_key="x",
        openai_api_key="y",
        openai_voice="marin",
        use_force_message=True,
        _env_file=None,
    )
    base.update(overrides)
    return Settings(**base)


def test_grok_session_is_pcmu_passthrough_with_force_message_opener():
    plan = build_session_plan(_settings(), "be brief", "Hello from Tack45")
    assert plan.provider == "grok"
    session = plan.setup_messages[0]["session"]
    assert session["audio"]["input"]["format"] == {"type": "audio/pcmu"}
    assert session["audio"]["output"]["format"] == {"type": "audio/pcmu"}
    assert "rate" not in session["audio"]["input"]["format"]
    assert "rate" not in session["audio"]["output"]["format"]
    assert session["turn_detection"] == {"type": "server_vad"}
    assert session["audio"]["input"]["transcription"]["model"] == "grok-transcribe"
    assert session["voice"] == "eve"
    names = {tool["name"] for tool in session["tools"]}
    assert names == {"end_call", "record_opt_out"}
    opener = plan.setup_messages[1]
    assert opener["item"]["type"] == "force_message"
    assert opener["item"]["interruptible"] is False
    assert opener["item"]["content"][0]["text"] == "Hello from Tack45"
    assert all(message["type"] != "response.create" for message in plan.setup_messages)
    encoded = json.dumps(plan.setup_messages)
    assert "g711_ulaw" not in encoded
    assert "24000" not in encoded


def test_openai_session_uses_audio_pcmu_and_a_spoken_opener():
    plan = build_session_plan(
        _settings(voice_provider="openai"),
        "be brief",
        "Hello from Tack45",
    )
    session = plan.setup_messages[0]["session"]
    assert session["type"] == "realtime"
    assert session["audio"]["input"]["format"] == {"type": "audio/pcmu"}
    assert session["audio"]["output"]["format"] == {"type": "audio/pcmu"}
    assert session["audio"]["input"]["turn_detection"] == {"type": "server_vad"}
    assert session["audio"]["output"]["voice"] == "marin"
    opener = plan.setup_messages[1]
    assert opener["type"] == "response.create"
    assert "Hello from Tack45" in opener["response"]["instructions"]
    assert "force_message" not in json.dumps(plan.setup_messages)


def test_grok_can_skip_force_message():
    plan = build_session_plan(_settings(use_force_message=False), "be brief", "Hello")
    assert plan.setup_messages[1]["type"] == "response.create"


def test_event_normalization_keeps_audio_payload():
    audio = normalize_event({"type": "response.output_audio.delta", "delta": "QQ=="})
    assert audio.kind == "audio"
    assert audio.audio_b64 == "QQ=="
    legacy = normalize_event({"type": "response.audio.delta", "delta": "QQ=="})
    assert legacy.audio_b64 == "QQ=="
    assert normalize_event({"type": "input_audio_buffer.speech_started"}).kind == "speech_started"
    updated = normalize_event(
        {
            "type": "conversation.item.input_audio_transcription.updated",
            "transcript": "hello there",
            "item_id": "item_1",
        }
    )
    assert updated.cumulative is True
    assert updated.final is False


def test_realtime_url_includes_the_model():
    assert realtime_ws_url("wss://api.x.ai/v1/realtime", "grok-voice-latest").endswith(
        "model=grok-voice-latest"
    )


def test_prompt_mentions_accent_opt_out_and_placeholder():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    system = (root / "prompts" / "system.txt").read_text(encoding="utf-8")
    opener = (root / "prompts" / "compliance_opener.txt").read_text(encoding="utf-8")
    assert "PLACEHOLDER" in system
    assert "Australian accent" in system
    assert "record_opt_out" in system
    assert "end_call" in system
    assert "AI assistant" in opener
    assert "Adam Hanley" in opener
    assert "Tack45" in opener
    rendered = render_instructions(
        system,
        opener=opener,
        name="Sam",
        business="Harbour {notes}",
        notes="left a message",
        number="+61200000000",
    )
    assert "Harbour {notes}" in rendered
    assert "left a message" in rendered
    assert "{opener}" not in rendered
    assert "Sam" in rendered
