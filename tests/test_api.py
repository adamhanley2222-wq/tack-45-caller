import base64
from datetime import datetime, timezone
from urllib.parse import parse_qs

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


def test_voice_hold_then_async_amd_branches(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, _settings = client
    created = http.post(
        "/calls",
        json={"to": "+61280001111", "name": "Pat", "business": "Quay Books"},
        headers=_auth(),
    ).json()
    call_id = created["id"]

    voice = http.post(
        f"/twilio/voice?call_id={call_id}",
        data={"CallSid": created["call_sid"]},
    )
    assert voice.status_code == 200
    assert "<Pause" in voice.text
    assert voice.headers["content-type"].startswith("text/xml")

    human = http.post(
        f"/twilio/amd?call_id={call_id}",
        data={"CallSid": created["call_sid"], "AnsweredBy": "human"},
    )
    assert human.status_code == 204
    update = _body(seen["requests"][-1])
    twiml = update["Twiml"][0]
    assert "<Connect>" in twiml
    assert "<Stream" in twiml
    assert "wss://caller.example.com/streams/media/" in twiml
    assert "Quay Books" in twiml
    assert 'track="inbound_track"' in twiml

    logged = http.get(f"/calls/{call_id}", headers=_auth()).json()
    assert logged["amd_result"] == "human"

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


def test_machine_message_and_unknown(monkeypatch, tmp_path, twilio_capture):
    from app.main import create_app
    from tests.conftest import make_settings

    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    settings = make_settings(tmp_path, machine_message="Sorry we missed you.")
    application = create_app(settings, http_transport=twilio_capture["transport"])
    from fastapi.testclient import TestClient

    with TestClient(application) as http:
        created = http.post("/calls", json={"to": "+61280003333"}, headers=_auth()).json()
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
        assert _body(twilio_capture["requests"][-1])["Status"] == ["completed"]
        logged_unknown = http.get(f"/calls/{other['id']}", headers=_auth()).json()
        assert logged_unknown["amd_result"] == "unknown"
        assert logged_unknown["end_reason"] == "amd_unknown"


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
    from tests.conftest import make_settings

    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    settings = make_settings(tmp_path, twilio_auth_token="auth-token")
    application = create_app(settings, http_transport=twilio_capture["transport"])
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


def test_no_transcode_imports():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "app"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    for banned in ("audioop", "soxr", "samplerate", "librosa", "numpy", "pydub"):
        assert f"import {banned}" not in text
        assert f"from {banned}" not in text
