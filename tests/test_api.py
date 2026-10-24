import asyncio
import base64
from datetime import datetime, timezone
from urllib.parse import parse_qs

import pytest

from app.signatures import compute_twilio_signature


def _auth(token="test-token"):
    return {"Authorization": f"Bearer {token}"}


def _allowed_time():
    # Thursday 15 Jan 2026 00:00 UTC is 11:00 in Sydney (AEDT).
    return datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)


def _body(request):
    return parse_qs(request.content.decode())


def test_health(client):
    http, _seen, _settings = client
    response = http.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "tack45-caller"


def test_bearer_required(client):
    http, _seen, _settings = client
    missing = http.post("/calls", json={"to": "+61412345678"})
    assert missing.status_code == 401
    assert missing.json()["code"] == "unauthorized"
    wrong = http.post("/calls", json={"to": "+61412345678"}, headers=_auth("nope"))
    assert wrong.status_code == 401


def test_create_call_uses_async_amd_and_api_key(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, settings = client
    response = http.post(
        "/calls",
        json={
            "to": "0412 345 678",
            "name": "Sam Lee",
            "business": "Harbour Roasters",
            "notes": "Asked about invoices",
        },
        headers=_auth(),
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["call_sid"].startswith("CA")
    assert body["to"] == "+61412345678"
    assert body["from"] == "+61290904464"
    assert body["direction"] == "outbound"
    assert body["provider"] == "grok"
    assert body["timezone"] == "Australia/Sydney"
    assert body["timezone_warning"]
    assert body["stream_url"].startswith("wss://caller.example.com/streams/media/")

    request = seen["requests"][0]
    encoded = base64.b64encode(
        f"{settings.twilio_api_key_sid}:{settings.twilio_api_key_secret}".encode()
    ).decode()
    assert request.headers["authorization"] == f"Basic {encoded}"
    form = _body(request)
    assert form["To"] == ["+61412345678"]
    assert form["From"] == ["+61290904464"]
    assert form["MachineDetection"] == ["DetectMessageEnd"]
    assert form["AsyncAmd"] == ["true"]
    assert "DetectMessageEnd" in form["MachineDetection"]
    assert "/twilio/voice?call_id=" in form["Url"][0]
    assert "/twilio/amd?call_id=" in form["AsyncAmdStatusCallback"][0]
    assert "/twilio/status?call_id=" in form["StatusCallback"][0]
    assert set(form["StatusCallbackEvent"]) == {"initiated", "ringing", "answered", "completed"}

    stored = http.get(f"/calls/{body['id']}", headers=_auth())
    assert stored.status_code == 200
    assert stored.json()["status"] == "initiated"


def test_test_number_is_allowed_on_sunday_unless_do_not_call(client, monkeypatch):
    http, seen, settings = client
    sunday = datetime(2026, 1, 18, 0, 0, tzinfo=timezone.utc)  # 11:00 Sunday Sydney
    monkeypatch.setattr("app.guards.utcnow", lambda: sunday)

    allowed = http.post(
        "/calls",
        json={"to": "+61425229662", "name": "Adam"},
        headers=_auth(),
    )
    assert allowed.status_code == 201, allowed.text
    assert allowed.json()["to"] == "+61425229662"
    assert len(seen["requests"]) == 1

    refused = http.post(
        "/calls",
        json={"to": "+61280001111", "name": "Pat"},
        headers=_auth(),
    )
    assert refused.status_code == 403
    assert refused.json()["code"] == "outside_calling_hours"
    assert len(seen["requests"]) == 1

    with open(settings.do_not_call_path, "w", encoding="utf-8") as handle:
        handle.write("+61425229662\n")
    blocked = http.post(
        "/calls",
        json={"to": "+61425229662", "name": "Adam"},
        headers=_auth(),
    )
    assert blocked.status_code == 403
    assert blocked.json()["code"] == "do_not_call"
    assert len(seen["requests"]) == 1


def test_refuses_sunday_and_do_not_call(client, monkeypatch, tmp_path):
    http, seen, settings = client
    sunday = datetime(2026, 1, 18, 0, 0, tzinfo=timezone.utc)  # 11:00 Sunday Sydney
    monkeypatch.setattr("app.guards.utcnow", lambda: sunday)
    refused = http.post("/calls", json={"to": "+61280001111", "name": "Pat"}, headers=_auth())
    assert refused.status_code == 403
    assert refused.json()["code"] == "outside_calling_hours"
    assert seen["requests"] == []

    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    dnc = settings.do_not_call_path
    with open(dnc, "w", encoding="utf-8") as handle:
        handle.write("# blocked\n0412 000 111\n")
    blocked = http.post(
        "/calls",
        json={"to": "+61412000111", "business": "Northside Bakery"},
        headers=_auth(),
    )
    assert blocked.status_code == 403
    assert blocked.json()["code"] == "do_not_call"
    assert seen["requests"] == []


def test_answer_connects_immediately_and_prewarms_before_amd(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, _settings = client
    created = http.post(
        "/calls",
        json={"to": "+61280001111", "name": "Pat", "business": "Quay Books"},
        headers=_auth(),
    ).json()
    call_id = created["id"]
    assert seen["providers"] == []

    voice = http.post(
        f"/twilio/voice?call_id={call_id}",
        data={"CallSid": created["call_sid"]},
    )
    assert voice.status_code == 200
    assert "<Pause" not in voice.text
    assert "<Connect>" in voice.text
    assert "<Stream" in voice.text
    assert "wss://caller.example.com/streams/media/" in voice.text
    assert "Quay Books" in voice.text
    assert 'track="inbound_track"' in voice.text
    assert voice.headers["content-type"].startswith("text/xml")
    logged = http.get(f"/calls/{call_id}", headers=_auth()).json()
    assert logged["status"] == "streaming"

    assert len(seen["providers"]) == 1
    provider = seen["providers"][0]
    assert provider.connected is True
    assert [item.get("type") for item in provider.sent] == ["session.update"]
    instructions = provider.sent[0]["session"]["instructions"]
    assert "Thanks Pat. I'm an AI assistant calling for Tack45" in instructions
    assert "has already been spoken" not in instructions

    before = len(seen["requests"])
    human = http.post(
        f"/twilio/amd?call_id={call_id}",
        data={"CallSid": created["call_sid"], "AnsweredBy": "human"},
    )
    assert human.status_code == 204
    assert len(seen["requests"]) == before
    assert provider.closed is False

    logged = http.get(f"/calls/{call_id}", headers=_auth()).json()
    assert logged["amd_result"] == "human"
    assert logged["status"] == "streaming"

    fax = http.post(
        "/calls",
        json={"to": "+61280002222"},
        headers=_auth(),
    )
    fax_id = fax.json()["id"]
    unknown = http.post(
        f"/twilio/amd?call_id={fax_id}",
        data={"CallSid": fax.json()["call_sid"], "AnsweredBy": "fax"},
    )
    assert unknown.status_code == 204
    hangup = _body(seen["requests"][-1])
    assert hangup["Status"] == ["completed"]
    logged_fax = http.get(f"/calls/{fax_id}", headers=_auth()).json()
    assert logged_fax["amd_result"] == "fax"
    assert logged_fax["end_reason"] == "amd_fax"


def test_machine_after_answer_hangs_up_without_a_new_session(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, _settings = client
    created = http.post(
        "/calls",
        json={"to": "+61280008888", "name": "Sam Lee"},
        headers=_auth(),
    ).json()
    voice = http.post(
        f"/twilio/voice?call_id={created['id']}",
        data={"CallSid": created["call_sid"]},
    )
    assert "<Connect>" in voice.text
    assert len(seen["providers"]) == 1
    http.post(
        f"/twilio/amd?call_id={created['id']}",
        data={"CallSid": created["call_sid"], "AnsweredBy": "machine_end_beep"},
    )
    hangup = _body(seen["requests"][-1])
    assert hangup["Status"] == ["completed"]
    assert "Twiml" not in hangup
    logged = http.get(f"/calls/{created['id']}", headers=_auth()).json()
    assert logged["amd_result"] == "machine_end_beep"
    assert logged["end_reason"] == "amd_machine"
    assert seen["providers"][0].closed is True
    assert len(seen["providers"]) == 1


def test_late_answered_status_does_not_restart_the_stream(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, _settings = client
    created = http.post("/calls", json={"to": "+61280009999", "name": "Jo"}, headers=_auth()).json()
    http.post(
        f"/twilio/voice?call_id={created['id']}",
        data={"CallSid": created["call_sid"]},
    )
    http.post(
        "/twilio/status",
        data={"CallSid": created["call_sid"], "CallStatus": "answered"},
    )
    http.post(
        "/twilio/status",
        data={"CallSid": created["call_sid"], "CallStatus": "ringing"},
    )
    logged = http.get(f"/calls/{created['id']}", headers=_auth()).json()
    assert logged["status"] == "streaming"
    before = len(seen["requests"])
    http.post(
        f"/twilio/amd?call_id={created['id']}",
        data={"CallSid": created["call_sid"], "AnsweredBy": "human"},
    )
    assert len(seen["requests"]) == before


def test_machine_message_and_unknown(monkeypatch, tmp_path, twilio_capture):
    from app.main import create_app
    from tests.conftest import make_settings, recording_factory

    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    settings = make_settings(tmp_path, machine_message="Sorry we missed you.")
    application = create_app(
        settings,
        http_transport=twilio_capture["transport"],
        provider_factory=recording_factory([]),
    )
    from fastapi.testclient import TestClient

    with TestClient(application) as http:
        created = http.post("/calls", json={"to": "+61280003333"}, headers=_auth()).json()
        voice = http.post(
            f"/twilio/voice?call_id={created['id']}",
            data={"CallSid": created["call_sid"]},
        )
        assert "<Connect>" in voice.text
        http.post(
            f"/twilio/amd?call_id={created['id']}",
            data={"CallSid": created["call_sid"], "AnsweredBy": "machine_end_silence"},
        )
        twiml = _body(twilio_capture["requests"][-1])["Twiml"][0]
        assert "Sorry we missed you." in twiml
        assert "<Hangup/>" in twiml
        logged = http.get(f"/calls/{created['id']}", headers=_auth()).json()
        assert logged["end_reason"] == "amd_machine"

        other = http.post("/calls", json={"to": "+61280004444"}, headers=_auth()).json()
        http.post(
            f"/twilio/amd?call_id={other['id']}",
            data={"CallSid": other["call_sid"], "AnsweredBy": "unknown"},
        )
        continued = _body(twilio_capture["requests"][-1])
        assert continued.get("Status") != ["completed"]
        assert "<Connect>" in continued["Twiml"][0]
        logged_unknown = http.get(f"/calls/{other['id']}", headers=_auth()).json()
        assert logged_unknown["amd_result"] == "unknown"
        assert logged_unknown["end_reason"] is None
        assert logged_unknown["status"] != "completed"


def test_late_unknown_amd_does_not_hang_up_a_live_call(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, _settings = client
    created = http.post(
        "/calls",
        json={"to": "+61280001234", "name": "Marc Hanley"},
        headers=_auth(),
    ).json()
    voice = http.post(
        f"/twilio/voice?call_id={created['id']}",
        data={"CallSid": created["call_sid"]},
    )
    assert "<Connect>" in voice.text
    assert [item.get("type") for item in seen["providers"][0].sent] == ["session.update"]
    before = len(seen["requests"])
    late = http.post(
        f"/twilio/amd?call_id={created['id']}",
        data={"CallSid": created["call_sid"], "AnsweredBy": "unknown"},
    )
    assert late.status_code == 204
    assert len(seen["requests"]) == before
    assert seen["providers"][0].closed is False
    logged = http.get(f"/calls/{created['id']}", headers=_auth()).json()
    assert logged["status"] == "streaming"
    assert logged["amd_result"] == "unknown"
    assert logged["end_reason"] is None

    odd = http.post(
        f"/twilio/amd?call_id={created['id']}",
        data={"CallSid": created["call_sid"], "AnsweredBy": "not_a_real_result"},
    )
    assert odd.status_code == 204
    assert len(seen["requests"]) == before
    assert seen["providers"][0].closed is False


def test_sync_amd_on_the_answer_webhook(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, _seen, _settings = client
    created = http.post("/calls", json={"to": "+61280005555"}, headers=_auth()).json()
    voice = http.post(
        f"/twilio/voice?call_id={created['id']}",
        data={"CallSid": created["call_sid"], "AnsweredBy": "human"},
    )
    assert "<Connect>" in voice.text
    assert "wss://" in voice.text


def test_status_callback_records_duration(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, _seen, _settings = client
    created = http.post("/calls", json={"to": "+61280006666"}, headers=_auth()).json()
    response = http.post(
        "/twilio/status",
        data={"CallSid": created["call_sid"], "CallStatus": "completed", "CallDuration": "42"},
    )
    assert response.status_code == 204
    logged = http.get(f"/calls/{created['id']}", headers=_auth()).json()
    assert logged["status"] == "completed"
    assert logged["duration_seconds"] == 42


def test_twilio_signature_is_enforced_when_configured(monkeypatch, tmp_path, twilio_capture):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import make_settings, recording_factory

    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    settings = make_settings(tmp_path, twilio_auth_token="auth-token")
    application = create_app(
        settings,
        http_transport=twilio_capture["transport"],
        provider_factory=recording_factory([]),
    )
    with TestClient(application) as http:
        created = http.post("/calls", json={"to": "+61280007777"}, headers=_auth()).json()
        url = f"https://caller.example.com/twilio/voice?call_id={created['id']}"
        form = {"CallSid": created["call_sid"], "AnsweredBy": "human"}
        bad = http.post(f"/twilio/voice?call_id={created['id']}", data=form)
        assert bad.status_code == 403
        signature = compute_twilio_signature(settings.twilio_auth_token, url, form)
        good = http.post(
            f"/twilio/voice?call_id={created['id']}",
            data=form,
            headers={"X-Twilio-Signature": signature},
        )
        assert good.status_code == 200
        assert "<Connect>" in good.text


def _expect_socket_closed(http):
    from starlette.websockets import WebSocketDisconnect

    try:
        with http.websocket_connect("/streams/media/not-a-real-token") as socket:
            socket.receive_text()
    except WebSocketDisconnect:
        return
    raise AssertionError("expected the socket to close")


def test_bad_stream_token_closes(client):
    http, _seen, _settings = client
    _expect_socket_closed(http)


@pytest.mark.asyncio
async def test_amd_reason_survives_the_relay_finishing(tmp_path):
    from app.db import Database
    from tests.conftest import make_settings

    db = Database(make_settings(tmp_path))
    await db.connect()
    try:
        await db.create_call(
            {
                "id": "c1",
                "stream_token": "tok",
                "to_number": "+61280001111",
                "from_number": "+61290904464",
                "name": "Pat",
                "business": "",
                "notes": "",
                "timezone": "Australia/Sydney",
                "provider": "grok",
                "status": "initiated",
            }
        )
        await db.attach_sid("c1", "CA1", "initiated")
        await db.mark_streaming("c1")
        await db.apply_twilio_status("CA1", "ringing", None)
        await db.apply_twilio_status("CA1", "answered", None)
        row = await db.get("c1")
        assert row["status"] == "streaming"
        assert row["answered_at"]
        await db.record_amd(
            "c1",
            amd_result="machine_end_silence",
            end_reason="amd_machine",
            status="machine",
        )
        await db.mark_in_progress("c1")
        await db.finish_call(
            "c1",
            end_reason="caller_hangup",
            transcript=[{"role": "assistant", "text": "Hi, is this Pat?"}],
            status="completed",
        )
        row = await db.get("c1")
        assert row["status"] == "machine"
        assert row["end_reason"] == "amd_machine"
        assert row["transcript"][0]["text"] == "Hi, is this Pat?"

        await db.create_call(
            {
                "id": "c2",
                "stream_token": "tok-2",
                "to_number": "+61280002222",
                "from_number": "+61290904464",
                "name": "Sam",
                "business": "",
                "notes": "",
                "timezone": "Australia/Sydney",
                "provider": "grok",
                "status": "streaming",
            }
        )
        await asyncio.gather(
            db.mark_in_progress("c2"),
            db.record_amd("c2", amd_result="human", end_reason=None, status=None),
            db.append_transcript("c2", [{"role": "assistant", "text": "Hi, is this Sam?"}]),
        )
        live = await db.get("c2")
        assert live["amd_result"] == "human"
        assert live["status"] == "in-progress"
        assert live["transcript"][0]["text"] == "Hi, is this Sam?"
    finally:
        await db.dispose()


def test_no_transcode_imports():
    """Grok, OpenAI, GPT-Live, and the cascade do not resample.

    Gemini Live is the exception: ``app/pcm_bridge.py`` uses the stdlib
    ``audioop`` codec. No other resampler is imported anywhere.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "app"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for banned in ("soxr", "samplerate", "librosa", "numpy", "pydub"):
            assert f"import {banned}" not in text
            assert f"from {banned}" not in text
        if path.name == "pcm_bridge.py":
            assert "import audioop" in text
            continue
        assert "import audioop" not in text
        assert "from audioop" not in text
