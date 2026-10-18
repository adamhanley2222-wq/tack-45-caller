import json

from app.config import Settings
from app.prompts import NO_NAME_OPENER, first_name_from, render_instructions, render_opener
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
