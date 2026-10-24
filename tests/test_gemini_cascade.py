"""Gemini cascade wiring. The simulation places no call and contacts no provider."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.api import prepare_script
from app.config import missing_dial_configuration
from app.gemini_sim import run_scenarios
from app.main import create_app
from app.providers import build_session_plan, provider_connection_target
from tests.conftest import make_settings
from tests.test_api import _allowed_time, _auth


def test_gemini_plan_reuses_prompt_assembly_and_keeps_accent_out_of_the_opener(tmp_path):
    settings = make_settings(
        tmp_path,
        voice_provider="gemini",
        gemini_api_key="gem-test",
        deepgram_api_key="dg-test",
    )
    opener, plan = prepare_script(
        settings,
        {
            "name": "Sam Lee",
            "business": "Harbour Roasters",
            "notes": "Asked about invoices",
            "to_number": "+61400000000",
            "brief": "Mention the workshop.",
        },
    )
    assert opener == "Hi, is this Sam?"
    message = plan.setup_messages[0]
    assert plan.provider == "gemini"
    assert message["type"] == "gemini.cascade"
    assert message["opener"] == opener
    assert "accent" not in message["opener"].lower()
    assert "Sam Lee" in message["instructions"]
    assert "Harbour Roasters" in message["instructions"]
    assert "Asked about invoices" in message["instructions"]
    assert "Mention the workshop." in message["instructions"]
    assert "record_opt_out" in message["instructions"]
    assert "end_call" in message["instructions"]
    assert "twelve words" in message["instructions"]
    assert message["tools"] == ["end_call", "record_opt_out"]
    assert message["tts"] == {
        "model": "gemini-3.8-flash-lite-tts",
        "voice": "Charon",
        "style": "Australian English accent, relaxed and friendly",
        "language": "en-AU",
    }
    assert message["stt"]["provider"] == "deepgram"
    encoded = str(plan.setup_messages)
    assert "session.update" not in encoded
    assert "session.start" not in encoded


def test_gemini_is_not_a_realtime_socket_and_needs_both_keys(tmp_path):
    grok = make_settings(tmp_path)
    assert "GEMINI_API_KEY" not in missing_dial_configuration(grok)
    assert "DEEPGRAM_API_KEY" not in missing_dial_configuration(grok)

    missing_model = make_settings(
        tmp_path,
        voice_provider="gemini",
        gemini_api_key="",
        deepgram_api_key="dg-test",
    )
    assert missing_dial_configuration(missing_model) == ["GEMINI_API_KEY"]
    missing_stt = make_settings(
        tmp_path,
        voice_provider="gemini",
        gemini_api_key="gem-test",
        deepgram_api_key="",
    )
    assert missing_dial_configuration(missing_stt) == ["DEEPGRAM_API_KEY"]
    ready = make_settings(
        tmp_path,
        voice_provider="gemini",
        gemini_api_key="gem-test",
        deepgram_api_key="dg-test",
    )
    assert missing_dial_configuration(ready) == []
    with pytest.raises(ValueError, match="cascaded pipeline"):
        provider_connection_target(ready)
    plan = build_session_plan(ready, "be brief", "Hi, is this Sam?")
    assert plan.provider == "gemini"


def test_gemini_answer_does_not_open_a_realtime_socket(tmp_path, twilio_capture, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    warmed: dict = {}

    def schedule(_app, _settings, opener):
        warmed["opener"] = opener

    monkeypatch.setattr("app.gemini_cascade.schedule_gemini_prewarm", schedule)

    def factory(*_args):
        raise AssertionError("gemini must not open a realtime socket")

    settings = make_settings(
        tmp_path,
        voice_provider="gemini",
        gemini_api_key="gem-test",
        deepgram_api_key="dg-test",
    )
    application = create_app(
        settings,
        http_transport=twilio_capture["transport"],
        provider_factory=factory,
    )
    with TestClient(application) as http:
        created = http.post(
            "/calls",
            json={"to": "+61280001111", "name": "Pat", "business": "Quay Books"},
            headers=_auth(),
        )
        assert created.status_code == 201
        assert created.json()["provider"] == "gemini"
        voice = http.post(
            f"/twilio/voice?call_id={created.json()['id']}",
            data={"CallSid": created.json()["call_sid"]},
        )
        assert voice.status_code == 200
        assert "<Connect>" in voice.text
        assert "wss://caller.example.com/streams/media/" in voice.text
    assert warmed["opener"] == "Hi, is this Pat?"
    assert twilio_capture["requests"]


def test_gemini_still_refuses_outside_calling_hours(tmp_path, twilio_capture, monkeypatch):
    sunday = datetime(2026, 1, 18, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("app.guards.utcnow", lambda: sunday)
    scheduled = []
    monkeypatch.setattr(
        "app.gemini_cascade.schedule_gemini_prewarm",
        lambda *_args, **_kwargs: scheduled.append(True),
    )
    settings = make_settings(
        tmp_path,
        voice_provider="gemini",
        gemini_api_key="gem-test",
        deepgram_api_key="dg-test",
    )
    application = create_app(settings, http_transport=twilio_capture["transport"])
    with TestClient(application) as http:
        refused = http.post(
            "/calls",
            json={"to": "+61280001111", "name": "Pat"},
            headers=_auth(),
        )
    assert refused.status_code == 403
    assert refused.json()["code"] == "outside_calling_hours"
    assert scheduled == []
    assert twilio_capture["requests"] == []


@pytest.mark.asyncio
async def test_local_simulation_covers_barge_in_overlap_opt_out_and_rate_limit():
    summary = await run_scenarios()
    assert summary["ok"] is True
    assert len(summary["lines"]) == 4
