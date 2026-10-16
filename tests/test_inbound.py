"""Inbound callbacks: same relay, no AMD, no calling-hours gate."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.db import Database
from app.prompts import inbound_opener
from app.signatures import compute_twilio_signature
from tests.conftest import make_settings, recording_factory
from tests.test_api import _allowed_time, _auth

KNOWN = "Hello, Tack45, is this Sam?"
UNKNOWN = "Hello, Tack45. Just so you know, I'm an AI assistant. How can I help?"
NO_NAME = "Hello, Tack45, is that the business owner?"
CONFIRM = "Thanks for calling back. Just so you know, I'm an AI assistant for Tack45."


def _spoken(provider) -> str:
    opener = provider.sent[1]
    assert opener["item"]["type"] == "force_message"
    return opener["item"]["content"][0]["text"]


def _instructions(provider) -> str:
    return provider.sent[0]["session"]["instructions"]


def _post_inbound(http, **fields):
    data = {
        "CallSid": "CAINBOUND0000000000000000000001",
        "From": "+61400999888",
        "To": "+61290904464",
        "CallStatus": "ringing",
    }
    data.update(fields)
    return http.post("/twilio/inbound", data=data)


def test_inbound_opener_known_unknown_and_unnamed():
    assert inbound_opener("Sam Lee", known=True) == KNOWN
    assert inbound_opener("", known=True) == NO_NAME
    assert inbound_opener("Sam Lee", known=False) == UNKNOWN
    assert "unknown" not in UNKNOWN.lower()


def test_inbound_prompt_keeps_disclosure_opt_out_and_the_adam_cta():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "prompts" / "inbound_system.txt").read_text(
        encoding="utf-8"
    )
    assert "{opener}" in text
    assert "{callback}" in text
    assert "male Australian voice" in text
    assert "ABC-newsreader-style General Australian accent" in text
    assert "Never claim to be human" in text
    assert "record_opt_out" in text
    assert "end_call" in text
    assert "free 60-minute chat with Adam" not in text
    assert "Would you be open to a chat with Adam, just to talk it through?" in text
    assert "Up to an hour, and there's no cost." in text
    assert "Ask at most twice" in text
    assert "Do not mention prices" in text
    assert "Do not claim client results" in text
    assert "PLACEHOLDER" in text
    assert "Do not use the word \"unknown\" aloud" in text
    assert "Thanks for calling back. Just so you know, I'm an AI assistant for Tack45." in text
    assert "The AI disclosure must be in the first or second turn." in text
    assert "Hello, Tack45, is this" not in text


def test_unknown_caller_is_answered_without_amd_or_a_twilio_update(client):
    http, seen, _settings = client
    response = _post_inbound(http)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/xml")
    assert "<Connect>" in response.text
    assert "<Stream" in response.text
    assert "<Pause" not in response.text
    assert "<Hangup" not in response.text
    assert seen["requests"] == []

    assert len(seen["providers"]) == 1
    provider = seen["providers"][0]
    assert provider.connected is True
    assert _spoken(provider) == UNKNOWN
    instructions = _instructions(provider)
    assert "Callback: no" in instructions
    assert "record_opt_out" in instructions
    assert "end_call" in instructions
    assert "Would you be open to a chat with Adam, just to talk it through?" in instructions
    assert "60-minute" not in instructions
    assert "Do not mention prices" in instructions
    assert UNKNOWN in instructions
    assert "That disclosure is the first turn." in instructions
    assert "I'm an AI" in UNKNOWN
    assert "AI" not in KNOWN

    listed = http.get("/calls", headers=_auth()).json()
    assert len(listed) == 1
    row = listed[0]
    assert row["direction"] == "inbound"
    assert row["from"] == "+61400999888"
    assert row["to"] == "+61290904464"
    assert row["status"] == "streaming"
    assert row["amd_result"] is None
    assert row["name"] == ""


def test_known_caller_gets_the_callback_opener_and_prior_context(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, _settings = client
    created = http.post(
        "/calls",
        json={
            "to": "02 8000 1111",
            "name": "Sam Lee",
            "business": "Harbour Roasters",
            "notes": "Asked about invoices",
        },
        headers=_auth(),
    )
    assert created.status_code == 201, created.text
    before = len(seen["requests"])
    assert seen["providers"] == []

    response = _post_inbound(
        http,
        CallSid="CAINBOUND0000000000000000000002",
        From="+61280001111",
    )
    assert response.status_code == 200
    assert "<Connect>" in response.text
    assert "<Pause" not in response.text
    assert len(seen["requests"]) == before

    assert len(seen["providers"]) == 1
    provider = seen["providers"][0]
    assert _spoken(provider) == KNOWN
    assert "AI" not in _spoken(provider)
    instructions = _instructions(provider)
    assert CONFIRM in instructions
    assert "second turn" in instructions
    assert "Callback: yes" in instructions
    assert "Harbour Roasters" in instructions
    assert "Asked about invoices" in instructions
    assert "Sam Lee" in instructions

    listed = http.get("/calls", headers=_auth()).json()
    inbound = next(row for row in listed if row["direction"] == "inbound")
    assert inbound["from"] == "+61280001111"
    assert inbound["to"] == "+61290904464"
    assert inbound["name"] == "Sam Lee"
    assert inbound["business"] == "Harbour Roasters"
    assert inbound["notes"] == "Asked about invoices"
    assert inbound["amd_result"] is None


def test_refused_outbound_and_sunday_do_not_block_or_count_as_a_callback(client, monkeypatch):
    http, seen, _settings = client
    sunday = datetime(2026, 1, 18, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("app.guards.utcnow", lambda: sunday)
    refused = http.post("/calls", json={"to": "+61280001111", "name": "Pat"}, headers=_auth())
    assert refused.status_code == 403
    assert refused.json()["code"] == "outside_calling_hours"

    response = _post_inbound(http, From="+61280001111", CallSid="CAINBOUND0000000000000000000003")
    assert response.status_code == 200
    assert "<Connect>" in response.text
    assert seen["requests"] == []
    assert _spoken(seen["providers"][0]) == UNKNOWN
    assert "Callback: no" in _instructions(seen["providers"][0])


def test_do_not_call_does_not_block_an_inbound_call(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, settings = client
    with open(settings.do_not_call_path, "w", encoding="utf-8") as handle:
        handle.write("0412 000 111\n")
    blocked = http.post("/calls", json={"to": "+61412000111", "name": "Pat"}, headers=_auth())
    assert blocked.status_code == 403
    assert blocked.json()["code"] == "do_not_call"

    response = _post_inbound(http, From="+61412000111", CallSid="CAINBOUND0000000000000000000004")
    assert response.status_code == 200
    assert "<Connect>" in response.text
    assert seen["requests"] == []
    assert _spoken(seen["providers"][0]) == UNKNOWN


def test_inbound_retry_reuses_the_stream_and_anonymous_callers_are_unknown(client):
    http, seen, _settings = client
    missing = http.post("/twilio/inbound", data={"From": "+61400999888"})
    assert missing.status_code == 200
    assert "<Hangup/>" in missing.text
    assert seen["providers"] == []

    first = _post_inbound(http, From="anonymous", CallSid="CAINBOUND0000000000000000000005")
    assert first.status_code == 200
    assert "<Connect>" in first.text
    again = _post_inbound(http, From="anonymous", CallSid="CAINBOUND0000000000000000000005")
    assert again.status_code == 200
    assert "<Connect>" in again.text
    assert first.text == again.text
    assert len(seen["providers"]) == 1
    assert _spoken(seen["providers"][0]) == UNKNOWN

    listed = http.get("/calls", headers=_auth()).json()
    assert len(listed) == 1
    assert listed[0]["from"] == "anonymous"
    assert listed[0]["direction"] == "inbound"

    from app.api import relay_phone

    assert relay_phone(listed[0] | {"from_number": listed[0]["from"], "to_number": listed[0]["to"]}) == "anonymous"
    assert relay_phone({"direction": "outbound", "to_number": "+614", "from_number": "+612"}) == "+614"


def test_inbound_signature_is_enforced_when_configured(tmp_path, twilio_capture):
    from app.main import create_app

    settings = make_settings(tmp_path, twilio_auth_token="auth-token")
    twilio_capture["providers"] = []
    application = create_app(
        settings,
        http_transport=twilio_capture["transport"],
        provider_factory=recording_factory(twilio_capture["providers"]),
    )
    form = {
        "CallSid": "CAINBOUND0000000000000000000006",
        "From": "+61400999888",
        "To": "+61290904464",
    }
    url = "https://caller.example.com/twilio/inbound"
    with TestClient(application) as http:
        bad = http.post("/twilio/inbound", data=form)
        assert bad.status_code == 403
        signature = compute_twilio_signature(settings.twilio_auth_token, url, form)
        good = http.post(
            "/twilio/inbound",
            data=form,
            headers={"X-Twilio-Signature": signature},
        )
        assert good.status_code == 200
        assert "<Connect>" in good.text
        assert "<Pause" not in good.text
    assert twilio_capture["requests"] == []


def test_prepare_inbound_script_for_a_known_caller_with_no_first_name(tmp_path):
    from app.api import prepare_inbound_script

    settings = make_settings(tmp_path)
    opener, plan = prepare_inbound_script(
        settings,
        {"from_number": "+61280001111"},
        {"name": "", "business": "Northside", "notes": "owner was out"},
    )
    assert opener == NO_NAME
    assert "AI" not in opener
    assert plan.setup_messages[1]["item"]["content"][0]["text"] == opener
    instructions = plan.setup_messages[0]["session"]["instructions"]
    assert CONFIRM in instructions
    assert "Callback: yes" in instructions
    assert "Northside" in instructions
    assert "owner was out" in instructions
    assert "+61280001111" in instructions


async def test_direction_column_is_added_to_an_existing_calls_table(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE calls (
            id VARCHAR(36) PRIMARY KEY,
            call_sid VARCHAR(64),
            stream_token VARCHAR(128) NOT NULL,
            to_number VARCHAR(20) NOT NULL,
            from_number VARCHAR(20),
            name VARCHAR(200),
            business VARCHAR(200),
            notes TEXT,
            timezone VARCHAR(64),
            provider VARCHAR(16),
            status VARCHAR(32),
            amd_result VARCHAR(64),
            end_reason VARCHAR(64),
            opted_out INTEGER,
            opt_out_reason TEXT,
            transcript_json TEXT,
            duration_seconds INTEGER,
            created_at DATETIME,
            answered_at DATETIME,
            ended_at DATETIME
        )
        """
    )
    conn.execute(
        "INSERT INTO calls (id, stream_token, to_number, status, created_at) "
        "VALUES ('old', 'tok-old', '+61280001111', 'completed', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    settings = make_settings(tmp_path, sqlite_path=str(path))
    db = Database(settings)
    await db.connect()
    try:
        legacy = await db.get("old")
        assert legacy is not None
        assert legacy["direction"] == "outbound"
        created = await db.create_call(
            {
                "id": "new",
                "stream_token": "tok-new",
                "to_number": "+61280002222",
                "status": "dialing",
            }
        )
        assert created["direction"] == "outbound"
    finally:
        await db.dispose()


async def test_latest_outbound_ignores_refused_inbound_and_old_calls(tmp_path):
    settings = make_settings(tmp_path)
    db = Database(settings)
    await db.connect()
    now = datetime.now(timezone.utc)
    try:
        await db.create_call(
            {
                "id": "old",
                "stream_token": "t-old",
                "call_sid": "CA_OLD",
                "to_number": "+61280001111",
                "name": "Old",
                "status": "completed",
                "direction": "outbound",
                "created_at": now - timedelta(days=120),
            }
        )
        await db.create_call(
            {
                "id": "refused",
                "stream_token": "t-refused",
                "call_sid": "CA_REFUSED",
                "to_number": "+61280001111",
                "name": "Refused",
                "status": "refused",
                "direction": "outbound",
                "created_at": now,
            }
        )
        await db.create_call(
            {
                "id": "no-sid",
                "stream_token": "t-nosid",
                "to_number": "+61280001111",
                "name": "Never placed",
                "status": "failed",
                "direction": "outbound",
                "created_at": now,
            }
        )
        await db.create_call(
            {
                "id": "inbound",
                "stream_token": "t-in",
                "call_sid": "CA_IN",
                "to_number": "+61280001111",
                "from_number": "+61290904464",
                "name": "Wrong direction",
                "status": "completed",
                "direction": "inbound",
                "created_at": now,
            }
        )
        await db.create_call(
            {
                "id": "recent",
                "stream_token": "t-recent",
                "call_sid": "CA_RECENT",
                "to_number": "+61280001111",
                "name": "Sam Lee",
                "business": "Harbour",
                "notes": "left a message",
                "status": "completed",
                "direction": "outbound",
                "created_at": now - timedelta(days=2),
            }
        )
        match = await db.latest_outbound_to("+61280001111", within_days=90)
        assert match is not None
        assert match["id"] == "recent"
        assert match["name"] == "Sam Lee"
        assert await db.latest_outbound_to("+61280001111", within_days=1) is None
        aged = await db.latest_outbound_to("+61280009999", within_days=90)
        assert aged is None
        await db.create_call(
            {
                "id": "ancient-only",
                "stream_token": "t-ancient",
                "call_sid": "CA_ANCIENT",
                "to_number": "+61280009999",
                "name": "Ancient",
                "status": "completed",
                "direction": "outbound",
                "created_at": now - timedelta(days=200),
            }
        )
        assert await db.latest_outbound_to("+61280009999", within_days=90) is None
        kept = await db.latest_outbound_to("+61280009999", within_days=0)
        assert kept is not None
        assert kept["id"] == "ancient-only"
    finally:
        await db.dispose()
