"""Finished calls are posted once, after the transcript is complete."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import Database, public_call
from app.transcript_webhook import (
    RETRY_DELAYS_SECONDS,
    WEBHOOK_TIMEOUT_SECONDS,
    deliver_transcript,
    post_with_retries,
    webhook_headers,
)
from tests.conftest import make_settings, recording_factory
from tests.test_api import _allowed_time, _auth

PAYLOAD_KEYS = {
    "id",
    "call_sid",
    "direction",
    "to",
    "from",
    "name",
    "business",
    "notes",
    "provider",
    "status",
    "amd_result",
    "end_reason",
    "opted_out",
    "opt_out_reason",
    "duration_seconds",
    "created_at",
    "answered_at",
    "ended_at",
    "transcript",
    "turn_latencies",
}


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class FakeClient:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.calls: list[dict] = []

    async def post(self, url, *, json, headers, timeout):
        self.calls.append(
            {"url": url, "json": json, "headers": dict(headers), "timeout": timeout}
        )
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _app(db, client, settings):
    application = SimpleNamespace()
    application.state = SimpleNamespace(
        db=db,
        http=client,
        settings=settings,
        webhook_tasks=set(),
    )
    return application


async def _seed(db: Database, **fields) -> dict:
    base = {
        "id": "call-1",
        "stream_token": "token-1",
        "call_sid": "CA123",
        "to_number": "+61280001111",
        "from_number": "+61290904464",
        "name": "Sam Lee",
        "business": "Harbour Roasters",
        "notes": "Asked about invoices",
        "provider": "grok",
        "status": "completed",
        "direction": "outbound",
    }
    base.update(fields)
    return await db.create_call(base)


def test_auth_header_is_bearer_only_for_authorization():
    bearer = Settings(
        _env_file=None,
        transcript_webhook_url="https://hooks.example/calls",
        transcript_webhook_secret="topsecret",
    )
    assert webhook_headers(bearer) == {"Authorization": "Bearer topsecret"}
    custom = Settings(
        _env_file=None,
        transcript_webhook_url="https://hooks.example/calls",
        transcript_webhook_secret="topsecret",
        transcript_webhook_auth_header="X-Webhook-Token",
    )
    assert webhook_headers(custom) == {"X-Webhook-Token": "topsecret"}
    unnamed = Settings(
        _env_file=None,
        transcript_webhook_url="https://hooks.example/calls",
        transcript_webhook_secret="",
        transcript_webhook_auth_header="X-Webhook-Token",
    )
    assert webhook_headers(unnamed) == {}
    pasted = Settings(
        _env_file=None,
        transcript_webhook_url="https://hooks.example/calls",
        transcript_webhook_secret="ignored-because-the-line-has-a-value",
        transcript_webhook_auth_header="Authorization: Bearer pasted-token",
    )
    assert webhook_headers(pasted) == {"Authorization": "Bearer pasted-token"}
    # The Railway failure used the whole line as the header name.
    assert ":" not in next(iter(webhook_headers(pasted)))
    scheme_only = Settings(
        _env_file=None,
        transcript_webhook_secret="topsecret",
        transcript_webhook_auth_header="Authorization: Bearer",
    )
    assert webhook_headers(scheme_only) == {"Authorization": "Bearer topsecret"}
    custom_line = Settings(
        _env_file=None,
        transcript_webhook_secret="topsecret",
        transcript_webhook_auth_header="X-Webhook-Token: line-secret",
    )
    assert webhook_headers(custom_line) == {"X-Webhook-Token": "line-secret"}
    assert len(RETRY_DELAYS_SECONDS) == 3
    assert WEBHOOK_TIMEOUT_SECONDS == 10


async def test_post_retries_then_succeeds_and_redacts_the_secret(caplog):
    settings = Settings(
        _env_file=None,
        transcript_webhook_url="https://hooks.example/calls",
        transcript_webhook_secret="topsecret",
        transcript_webhook_auth_header="X-Webhook-Token",
    )
    client = FakeClient(
        [
            RuntimeError("connect failed topsecret"),
            FakeResponse(503),
            FakeResponse(204),
        ]
    )
    payload = {"id": "call-1", "transcript": [{"speaker": "assistant", "text": "Hello"}]}
    caplog.set_level(logging.WARNING, logger="tack45.transcript")
    ok = await post_with_retries(
        client,
        settings,
        payload,
        retry_delays=(0, 0, 0),
        sleep=_no_sleep,
    )
    assert ok is True
    assert len(client.calls) == 3
    assert client.calls[0]["headers"] == {"X-Webhook-Token": "topsecret"}
    assert client.calls[0]["timeout"] == 10
    assert client.calls[0]["json"] == payload
    logged = caplog.text
    assert "topsecret" not in logged
    assert "[redacted]" in logged
    assert "503" in logged


async def test_post_gives_up_after_three_retries(caplog):
    settings = Settings(
        _env_file=None,
        transcript_webhook_url="https://hooks.example/calls",
        transcript_webhook_secret="topsecret",
    )
    client = FakeClient([FakeResponse(500), FakeResponse(500), FakeResponse(500), FakeResponse(500)])
    caplog.set_level(logging.WARNING, logger="tack45.transcript")
    ok = await post_with_retries(
        client,
        settings,
        {"id": "call-1"},
        retry_delays=(0, 0, 0),
        sleep=_no_sleep,
    )
    assert ok is False
    assert len(client.calls) == 4
    assert all(call["headers"]["Authorization"] == "Bearer topsecret" for call in client.calls)
    assert "topsecret" not in caplog.text
    assert "Gave up posting the transcript" in caplog.text


async def test_unset_url_posts_nothing_and_an_open_stream_waits(tmp_path):
    settings = make_settings(tmp_path, transcript_webhook_url="", transcript_webhook_secret="topsecret")
    db = Database(settings)
    await db.connect()
    client = FakeClient([])
    try:
        await _seed(db)
        await deliver_transcript(_app(db, client, settings), "call-1")
        assert client.calls == []

        settings.transcript_webhook_url = "https://hooks.example/calls"
        await db.finish_call(
            "call-1",
            end_reason="caller_hangup",
            transcript=[{"role": "assistant", "text": "Hello, Tack45"}],
            status="completed",
        )
        await db.mark_stream_open("call-1")
        with pytest.raises(asyncio.CancelledError):
            await deliver_transcript(
                _app(db, client, settings),
                "call-1",
                close_wait=30,
                sleep=_cancel_sleep,
            )
        assert client.calls == []
        assert (await db.get("call-1"))["transcript_posted"] is False

        await db.mark_stream_closed("call-1")
        client.results.append(FakeResponse(204))
        await deliver_transcript(
            _app(db, client, settings),
            "call-1",
            attach_grace=0,
            retry_delays=(),
            sleep=_no_sleep,
        )
        assert len(client.calls) == 1
        body = client.calls[0]["json"]
        assert set(body) == PAYLOAD_KEYS
        assert body["direction"] == "outbound"
        assert body["to"] == "+61280001111"
        assert body["from"] == "+61290904464"
        assert body["name"] == "Sam Lee"
        assert body["business"] == "Harbour Roasters"
        assert body["notes"] == "Asked about invoices"
        assert body["transcript"] == [{"role": "assistant", "text": "Hello, Tack45"}]
        assert body["opted_out"] is False
        await deliver_transcript(
            _app(db, client, settings),
            "call-1",
            attach_grace=0,
            retry_delays=(),
            sleep=_no_sleep,
        )
        assert len(client.calls) == 1
    finally:
        await db.dispose()


async def test_new_columns_are_added_to_an_existing_calls_table(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE calls (
            id VARCHAR(36) PRIMARY KEY,
            call_sid VARCHAR(64),
            stream_token VARCHAR(128) NOT NULL,
            direction VARCHAR(16) NOT NULL DEFAULT 'outbound',
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
    conn.commit()
    conn.close()
    settings = make_settings(tmp_path, sqlite_path=str(path))
    db = Database(settings)
    await db.connect()
    try:
        async with db.engine.connect() as connection:
            rows = (await connection.execute(_pragma())).all()
        names = {row[1] for row in rows}
        assert {
            "stream_pending",
            "stream_open",
            "transcript_posted",
            "provider_voice",
            "provider_session_id",
            "turn_latency_json",
        } <= names
        created = await _seed(db, id="legacy", stream_token="token-legacy", call_sid="CALEGACY")
        assert created["transcript_posted"] is False
        claimed = await db.claim_transcript_webhook("legacy")
        assert claimed is not None
        assert claimed["status"] == "completed"
        assert await db.claim_transcript_webhook("legacy") is None
    finally:
        await db.dispose()


async def test_turn_latencies_stay_on_the_call_and_the_webhook(tmp_path):
    settings = make_settings(tmp_path, transcript_webhook_url="https://hooks.example/calls")
    db = Database(settings)
    await db.connect()
    client = FakeClient([FakeResponse(204)])
    turns = [{"speech_to_audio_ms": 800, "stt_ms": 0, "llm_ms": 500, "tts_ms": 800}]
    try:
        await _seed(db, status="in-progress")
        await db.finish_call(
            "call-1",
            end_reason="caller_hangup",
            transcript=[{"role": "caller", "text": "Yes"}],
            status="completed",
            turn_latencies=turns,
        )
        await db.finish_call(
            "call-1",
            end_reason="caller_hangup",
            transcript=[{"role": "caller", "text": "Yes"}],
            status="completed",
        )
        stored = await db.get("call-1")
        assert stored["turn_latencies"] == turns
        shown = public_call(stored)
        assert shown["turn_latencies"] == turns
        await db.mark_stream_closed("call-1")
        await deliver_transcript(
            _app(db, client, settings),
            "call-1",
            attach_grace=0,
            retry_delays=(),
            sleep=_no_sleep,
        )
        assert client.calls[0]["json"]["turn_latencies"] == turns
    finally:
        await db.dispose()


def test_status_callback_posts_each_finished_call_once(monkeypatch, tmp_path):
    from app.main import create_app

    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    seen: dict = {"requests": []}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["requests"].append(request)
        if request.url.host == "hooks.example":
            return httpx.Response(204)
        if str(request.url).endswith("/Calls.json"):
            return httpx.Response(201, json={"sid": "CAOUTBOUND000000000000000000001", "status": "queued"})
        return httpx.Response(200, json={"sid": "CA", "status": "completed"})

    settings = make_settings(
        tmp_path,
        transcript_webhook_url="https://hooks.example/calls",
        transcript_webhook_secret="topsecret",
    )
    application = create_app(
        settings,
        http_transport=_Transport(handler),
        provider_factory=recording_factory([]),
    )
    with TestClient(application) as http:
        created = http.post(
            "/calls",
            json={"to": "+61280001111", "name": "Sam Lee", "business": "Harbour Roasters"},
            headers=_auth(),
        ).json()
        ringing = http.post(
            "/twilio/status",
            data={"CallSid": created["call_sid"], "CallStatus": "ringing"},
        )
        assert ringing.status_code == 204
        _flush(http)
        assert _hooks(seen) == []

        async def finish():
            await http.app.state.db.finish_call(
                created["id"],
                end_reason="caller_hangup",
                transcript=[{"role": "caller", "text": "Yes"}],
                status="completed",
            )

        http.portal.call(finish)
        completed = http.post(
            "/twilio/status",
            data={"CallSid": created["call_sid"], "CallStatus": "completed", "CallDuration": "12"},
        )
        assert completed.status_code == 204
        again = http.post(
            "/twilio/status",
            data={"CallSid": created["call_sid"], "CallStatus": "completed", "CallDuration": "12"},
        )
        assert again.status_code == 204
        _flush(http)

        hooks = _hooks(seen)
        assert len(hooks) == 1
        assert hooks[0].headers["authorization"] == "Bearer topsecret"
        body = json.loads(hooks[0].content)
        assert set(body) == PAYLOAD_KEYS
        assert body["id"] == created["id"]
        assert body["direction"] == "outbound"
        assert body["status"] == "completed"
        assert body["duration_seconds"] == 12
        assert body["transcript"] == [{"role": "caller", "text": "Yes"}]
        assert "topsecret" not in hooks[0].content.decode()

        inbound = http.post(
            "/twilio/inbound",
            data={"CallSid": "CAINBOUND0000000000000000000099", "From": "+61400111222", "To": "+61290904464"},
        )
        assert inbound.status_code == 200

        async def close_stream():
            call = await http.app.state.db.get_by_sid("CAINBOUND0000000000000000000099")
            await http.app.state.db.mark_stream_closed(call["id"])

        http.portal.call(close_stream)
        no_answer = http.post(
            "/twilio/status",
            data={"CallSid": "CAINBOUND0000000000000000000099", "CallStatus": "completed", "CallDuration": "3"},
        )
        assert no_answer.status_code == 204
        _flush(http)
        inbound_hooks = _hooks(seen)[1:]
        assert len(inbound_hooks) == 1
        inbound_body = json.loads(inbound_hooks[0].content)
        assert inbound_body["direction"] == "inbound"
        assert inbound_body["from"] == "+61400111222"
        assert inbound_body["to"] == "+61290904464"
        assert isinstance(inbound_body["transcript"], list)


class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, handler) -> None:
        self.handler = handler

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)


def _hooks(seen) -> list:
    return [request for request in seen["requests"] if request.url.host == "hooks.example"]


def _flush(http) -> None:
    async def _run() -> None:
        tasks = list(http.app.state.webhook_tasks)
        if tasks:
            await asyncio.gather(*tasks)

    http.portal.call(_run)


async def _no_sleep(_seconds: float) -> None:
    return None


async def _cancel_sleep(_seconds: float) -> None:
    raise asyncio.CancelledError()


def _pragma():
    from sqlalchemy import text

    return text("PRAGMA table_info(calls)")


@pytest.mark.asyncio
async def test_resolved_gptlive_voice_is_on_the_call_record(tmp_path):
    settings = make_settings(tmp_path)
    db = Database(settings)
    await db.connect()
    try:
        created = await _seed(db, provider="gptlive", status="in-progress")
        assert created["provider_voice"] is None
        assert created["provider_session_id"] is None
        await db.record_provider_session(
            "call-1",
            voice="ripple",
            session_id="live_abc123",
        )
        stored = await db.get("call-1")
        assert stored["provider_voice"] == "ripple"
        assert stored["provider_session_id"] == "live_abc123"
        shown = public_call(stored)
        assert shown["provider_voice"] == "ripple"
        assert shown["provider_session_id"] == "live_abc123"
    finally:
        await db.dispose()
