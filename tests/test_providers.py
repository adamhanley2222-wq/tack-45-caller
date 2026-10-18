import json

from app.config import Settings, missing_dial_configuration
from app.prompts import NO_NAME_OPENER, first_name_from, render_instructions, render_opener
from app.providers import (
    build_session_plan,
    normalize_event,
    provider_connection_target,
    realtime_ws_url,
)


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


def test_gptlive_session_is_pcmu_8000_with_responses_tools():
    plan = build_session_plan(
        _settings(voice_provider="gptlive"),
        "be brief",
        "Hi, is this Sam?",
    )
    assert plan.provider == "gptlive"
    assert len(plan.setup_messages) == 1
    start = plan.setup_messages[0]
    assert start["type"] == "session.start"
    session = start["session"]
    assert session["model"] == "gpt-live-1"
    assert session["audio"]["format"] == {"type": "audio/pcmu", "rate": 8000}
    assert session["audio"]["output"]["voice"] == "ripple"
    assert "tools" not in session
    assert "be brief" in session["instructions"]
    assert "Delegation policy" in session["instructions"]
    assert "record_opt_out" in session["instructions"]
    assert "end_call" in session["instructions"]
    delegation = session["delegation"]
    assert delegation["type"] == "responses"
    backend = delegation["responses"]
    assert backend["model"] == "gpt-6-luna"
    assert backend["tool_choice"] == "auto"
    assert backend["parallel_tool_calls"] is False
    assert "be brief" in backend["instructions"]
    assert {tool["name"] for tool in backend["tools"]} == {"end_call", "record_opt_out"}
    greeting = session["input"][0]
    assert greeting["role"] == "developer"
    assert greeting["content"][0]["type"] == "input_text"
    assert "Hi, is this Sam?" in greeting["content"][0]["text"]
    assert "pause and listen" in greeting["content"][0]["text"]
    encoded = json.dumps(plan.setup_messages)
    assert "force_message" not in encoded
    assert "input_audio_buffer.append" not in encoded
    assert "g711_ulaw" not in encoded


def test_gptlive_settings_can_override_model_voice_and_backend():
    plan = build_session_plan(
        _settings(
            voice_provider="gptlive",
            gptlive_model="gpt-live-1",
            gptlive_voice="marin",
            gptlive_responses_model="gpt-5.6-luna",
        ),
        "be brief",
        "Hello",
    )
    session = plan.setup_messages[0]["session"]
    assert session["audio"]["output"]["voice"] == "marin"
    assert session["delegation"]["responses"]["model"] == "gpt-5.6-luna"


def test_gptlive_connection_has_no_model_query():
    url, headers = provider_connection_target(
        _settings(voice_provider="gptlive", openai_api_key="sk-live", gptlive_url="wss://api.openai.com/v1/live/sessions")
    )
    assert url == "wss://api.openai.com/v1/live/sessions"
    assert "model" not in url
    assert headers == {"Authorization": "Bearer sk-live"}


def test_gptlive_events_normalize_audio_transcripts_tools_and_errors():
    audio = normalize_event({"type": "session.output_audio.delta", "delta": "QQ=="})
    assert audio.kind == "audio"
    assert audio.audio_b64 == "QQ=="
    assert audio.start_ms is None

    caller = normalize_event(
        {
            "type": "session.input_transcript.delta",
            "delta": "What is",
            "start_ms": 1000,
            "end_ms": 1200,
        }
    )
    assert caller.kind == "user_transcript"
    assert caller.text == "What is"
    assert caller.final is False
    assert caller.cumulative is False
    assert caller.start_ms == 1000
    assert caller.end_ms == 1200
    assert normalize_event({"type": "session.input_transcript.delta", "delta": " hi"}).kind == (
        "user_transcript"
    )

    spoken = normalize_event(
        {"type": "session.output_transcript.delta", "delta": "Hello", "start_ms": 0, "end_ms": 400}
    )
    assert spoken.kind == "assistant_transcript"
    assert spoken.text == "Hello"
    assert spoken.final is False

    assert normalize_event({"type": "session.started", "session": {"id": "sess_1"}}).kind == (
        "session_started"
    )
    assert normalize_event({"type": "session.closed"}).kind == "session_closed"

    tool = normalize_event(
        {
            "type": "response.event",
            "delegation_id": "item_1",
            "event": {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "name": "record_opt_out",
                    "call_id": "call_9",
                    "arguments": {"reason": "stop calling"},
                },
            },
        }
    )
    assert tool.kind == "function_call"
    assert tool.name == "record_opt_out"
    assert tool.call_id == "call_9"
    assert tool.arguments == {"reason": "stop calling"}

    done = normalize_event(
        {
            "type": "response.event",
            "event": {"type": "response.completed", "response": {"output": []}},
        }
    )
    assert done.kind == "response_done"
    created = normalize_event(
        {"type": "response.event", "event": {"type": "response.created", "response": {"id": "resp_1"}}}
    )
    assert created.kind == "ignore"

    err = normalize_event(
        {
            "type": "error",
            "error": {"message": "unknown model", "code": "responses_validation_failed"},
        }
    )
    assert err.kind == "error"
    assert err.text == "unknown model"
    assert normalize_event({"type": "error", "error": "socket closed"}).text == "socket closed"


def test_gptlive_requires_openai_api_key_and_grok_stays_the_default():
    assert _settings().voice_provider == "grok"
    assert _settings().gptlive_url == "wss://api.openai.com/v1/live/sessions"
    assert _settings().gptlive_model == "gpt-live-1"
    assert _settings().gptlive_voice == "ripple"
    assert missing_dial_configuration(_dial_settings(voice_provider="gptlive", openai_api_key="")) == [
        "OPENAI_API_KEY"
    ]
    assert missing_dial_configuration(_dial_settings(voice_provider="gptlive")) == []
    assert missing_dial_configuration(_dial_settings(voice_provider="grok", xai_api_key="")) == [
        "XAI_API_KEY"
    ]
    assert missing_dial_configuration(_dial_settings(voice_provider="openai", openai_api_key="")) == [
        "OPENAI_API_KEY"
    ]
    assert "VOICE_PROVIDER" in missing_dial_configuration(_dial_settings(voice_provider="other"))


def test_realtime_url_includes_the_model():
    assert realtime_ws_url("wss://api.x.ai/v1/realtime", "grok-voice-latest").endswith(
        "model=grok-voice-latest"
    )


def test_prompt_mentions_accent_opt_out_and_placeholder():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    system = (root / "prompts" / "system.txt").read_text(encoding="utf-8")
    opener = (root / "prompts" / "compliance_opener.txt").read_text(encoding="utf-8").strip()
    assert opener == "Hi, is this {first_name}?"
    assert "PLACEHOLDER" not in system
    assert "KNOWLEDGE" in system
    assert "Adam Hanley, who runs Tack45" in system
    assert "Northern Beaches" in system
    assert "Forestville" not in system
    assert "North Shore" not in system
    assert "University of Sydney" in system
    assert "AI for Owners" in system
    assert "UiPath integration and audits" in system
    assert "Australian Privacy Principles" in system
    assert "adam@tack45.com.au" in system
    assert "tack45-website.pages.dev" in system
    assert "ring back on this number" in system
    assert "Recruit Vector" in system
    assert "male Australian voice" in system
    assert "ABC-newsreader-style General Australian accent" in system
    assert "record_opt_out" in system
    assert "end_call" in system
    assert "I'm an AI assistant calling for Tack45, a Sydney-based AI consultancy" in system
    assert "free 60-minute chat with Adam" not in system
    assert "Would that be worth an hour?" not in system
    assert (
        "Would you be open to a chat with Adam, just to talk it through? "
        "No cost, and you'll come away with at least one useful idea."
    ) in system
    assert "Up to an hour, and there's no cost." in system
    assert "Would it help to talk that through with Adam?" in system
    assert "Ask at most twice" in system
    assert "Do not mention prices" in system
    assert "Do not claim client results" in system
    assert "removed from the call list" in system
    assert "Never claim to be human" in system
    assert "compliance opener has already been spoken" not in system
    assert "AI assistant" not in opener
    assert "Tack45" not in opener
    spoken = render_opener(opener, "Sam Lee")
    rendered = render_instructions(
        system,
        opener=spoken,
        name="Sam Lee",
        business="Harbour {notes}",
        notes="left a message",
        number="+61200000000",
        first_name=first_name_from("Sam Lee"),
    )
    assert "Harbour {notes}" in rendered
    assert "left a message" in rendered
    assert "{opener}" not in rendered
    assert "{named_thanks}" not in rendered
    assert "Hi, is this Sam?" in rendered
    assert (
        "Thanks Sam. I'm an AI assistant calling for Tack45, a Sydney-based AI consultancy. "
        "Is now a good time for a quick chat?"
    ) in rendered
    assert "Ha, I'll leave the cooking to you." in rendered
    assert "Thanks unknown" not in rendered


def test_prepare_script_speaks_the_owner_check_when_there_is_no_name(tmp_path):
    from app.api import prepare_script
    from tests.conftest import make_settings

    settings = make_settings(tmp_path)
    opener, plan = prepare_script(
        settings,
        {"name": "", "business": "Northside", "notes": "", "to_number": "+61280001111"},
    )
    assert opener == "Hi, is that the business owner?"
    assert plan.setup_messages[1]["item"]["content"][0]["text"] == opener
    instructions = plan.setup_messages[0]["session"]["instructions"]
    assert "Thanks. I'm an AI assistant calling for Tack45, a Sydney-based AI consultancy" in instructions
    assert "Thanks unknown" not in instructions
    assert "Do not pitch" in instructions


def test_opener_uses_the_first_name_or_asks_for_the_owner():
    template = "Hi, is this {first_name}?"
    assert first_name_from("Sam Lee") == "Sam"
    assert first_name_from("  Mary-Jane Watson ") == "Mary-Jane"
    assert first_name_from("") == ""
    assert first_name_from("...") == ""
    assert render_opener(template, "Sam Lee") == "Hi, is this Sam?"
    assert render_opener(template, "") == NO_NAME_OPENER
    assert render_opener(template, "   ") == "Hi, is that the business owner?"
    unnamed = render_instructions(
        "Say {named_thanks} First name: {first_name}. {opener}",
        opener=render_opener(template, ""),
        name="",
        business="",
        notes="",
        number="+611",
        first_name="",
    )
    assert "Thanks. " in unnamed or unnamed.startswith("Say Thanks.")
    assert "Thanks unknown" not in unnamed
    assert "Hi, is that the business owner?" in unnamed
