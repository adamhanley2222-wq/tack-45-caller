from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

ROOT = Path(__file__).resolve().parents[1]


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        api_bearer_token="test-token",
        public_base_url="https://caller.example.com",
        twilio_account_sid="ACaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        twilio_api_key_sid="SKbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        twilio_api_key_secret="api-key-secret",
        twilio_from_number="+61290904464",
        voice_provider="grok",
        xai_api_key="xai-test-key",
        openai_api_key="openai-test-key",
        database_url="",
        sqlite_path=str(tmp_path / "calls.db"),
        do_not_call_path=str(tmp_path / "do_not_call.txt"),
        prompt_path=str(ROOT / "prompts" / "system.txt"),
        opener_path=str(ROOT / "prompts" / "compliance_opener.txt"),
        inbound_prompt_path=str(ROOT / "prompts" / "inbound_system.txt"),
        machine_message="",
        default_timezone="Australia/Sydney",
    )
    values.update(overrides)
    (tmp_path / "do_not_call.txt").write_text("# none\n", encoding="utf-8")
    return Settings(_env_file=None, **values)


class AsyncHandlerTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler) -> None:
        self.handler = handler

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)


@pytest.fixture
def twilio_capture():
    seen: dict = {"requests": [], "n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["requests"].append(request)
        path = request.url.path
        if path.endswith("/Calls.json"):
            seen["n"] += 1
            sid = f"CA{seen['n']:032d}"
            return httpx.Response(201, json={"sid": sid, "status": "queued"})
        if "/Calls/" in path:
            return httpx.Response(200, json={"sid": "CA_UPDATED", "status": "in-progress"})
        return httpx.Response(500, json={"message": f"unexpected {path}"})

    seen["transport"] = AsyncHandlerTransport(handler)
    return seen


class RecordingProvider:
    """In-memory realtime socket. Tests must not open a billed provider session."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.connected = False
        self.closed = False
        self.inbound: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        self.connected = True

    async def send(self, payload: dict) -> None:
        self.sent.append(payload)

    async def recv(self):
        return await self.inbound.get()

    async def close(self) -> None:
        self.closed = True


def recording_factory(bucket: list):
    def factory(_settings, _call):
        provider = RecordingProvider()
        bucket.append(provider)
        return provider

    return factory


@pytest.fixture
def client(tmp_path, twilio_capture):
    settings = make_settings(tmp_path)
    twilio_capture["providers"] = []
    application = create_app(
        settings,
        http_transport=twilio_capture["transport"],
        provider_factory=recording_factory(twilio_capture["providers"]),
    )
    with TestClient(application) as test_client:
        yield test_client, twilio_capture, settings
