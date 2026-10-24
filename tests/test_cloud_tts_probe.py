"""Cloud TTS probe. HTTP is mocked. Nothing is sent to Google."""

from __future__ import annotations

import base64
import json
import logging
import struct

import httpx
import pytest

from google.auth.credentials import TokenState

from app.config import Settings, missing_dial_configuration
from app.tts_probe import (
    CLOUD_PLATFORM_SCOPE,
    CLOUD_TTS_URL,
    mulaw_from_cloud_audio,
    mulaw_wav,
    reset_cloudtts_auth_cache,
    synthesize_cloudtts_streaming,
)
from tests.conftest import make_settings
from tests.test_api import _auth

SECRET = "google-tts-test-key"


class Clock:
    def __init__(self, times: list[float]) -> None:
        self.times = list(times)

    def __call__(self) -> float:
        if not self.times:
            raise AssertionError("clock ran out of readings")
        return self.times.pop(0)


def _pcm_wav(pcm: bytes, rate: int = 8000) -> bytes:
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        rate,
        rate * 2,
        2,
        16,
        b"data",
        len(pcm),
    )
    return header + pcm


def test_mulaw_wav_is_unwrapped_without_touching_the_clock():
    raw = b"\x7f\x00\xff"
    audio, conversion_ms, error = mulaw_from_cloud_audio(mulaw_wav(raw), _raise_clock)
    assert error is None
    assert audio == raw
    assert conversion_ms == 0.0


def test_linear16_is_converted_and_the_conversion_is_timed():
    import audioop

    pcm = struct.pack("<hh", 0, 16000)
    clock = Clock([1.0, 1.2])
    audio, conversion_ms, error = mulaw_from_cloud_audio(_pcm_wav(pcm), clock)
    assert error is None
    assert audio == audioop.lin2ulaw(pcm, 2)
    assert conversion_ms == pytest.approx(200.0)
    assert clock.times == []


def _raise_clock() -> float:
    raise AssertionError("μ-law audio must not be timed as a conversion")


class _ScriptedCloud:
    def __init__(self, responses: list[httpx.Response], seen: list[dict]) -> None:
        self.responses = responses
        self.seen = seen

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, url, headers=None, json=None):
        self.seen.append({"url": str(url), "headers": dict(headers or {}), "json": json})
        return self.responses.pop(0)


def _install(monkeypatch, responses: list[httpx.Response], seen: list[dict]) -> None:
    def factory(*_args, **_kwargs):
        return _ScriptedCloud(responses, seen)

    monkeypatch.setattr("app.tts_probe.httpx.AsyncClient", factory)


def _post(http, body: dict):
    return http.post("/probe/tts", json=body, headers=_auth())


def test_cloudtts_requires_bearer_and_credentials(client, caplog):
    caplog.set_level(logging.DEBUG)
    http, _seen, settings = client
    settings.gemini_api_key = ""
    settings.google_tts_api_key = ""
    settings.google_tts_credentials_json = ""
    hidden = http.post("/probe/tts", json={"provider": "cloudtts"})
    assert hidden.status_code == 401
    refused = _post(http, {"provider": "cloudtts", "sentences": ["Thanks Ian."]})
    assert refused.status_code == 400
    assert refused.json()["detail"] == "GOOGLE_TTS_CREDENTIALS_JSON not set"
    assert SECRET not in refused.text
    assert "GOOGLE_TTS_CREDENTIALS_JSON not set" in refused.text


def test_google_tts_key_is_not_required_to_dial(tmp_path):
    settings = make_settings(tmp_path, google_tts_api_key="", google_tts_credentials_json="")
    missing = missing_dial_configuration(settings)
    assert "GOOGLE_TTS_API_KEY" not in missing
    assert "GOOGLE_TTS_CREDENTIALS_JSON" not in missing
    plain = Settings(_env_file=None)
    assert plain.google_tts_api_key == ""
    assert plain.google_tts_credentials_json == ""
    assert "GOOGLE_TTS_API_KEY" not in missing_dial_configuration(plain)
    assert "GOOGLE_TTS_CREDENTIALS_JSON" not in missing_dial_configuration(plain)
    loaded = Settings(
        _env_file=None,
        google_tts_credentials_json='{"private_key": "-----BEGIN PRIVATE KEY-----\\nSECRET\\n-----END PRIVATE KEY-----\\n"}',
    )
    assert "SECRET" not in repr(loaded)
    assert "BEGIN PRIVATE KEY" not in repr(loaded)


def test_cloudtts_mulaw_timings_and_saved_wav(client, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    http, _seen, settings = client
    settings.gemini_api_key = ""
    settings.google_tts_api_key = SECRET
    raw = b"\x7f" * 160
    seen: list[dict] = []
    _install(
        monkeypatch,
        [httpx.Response(200, json={"audioContent": base64.b64encode(mulaw_wav(raw)).decode("ascii")})],
        seen,
    )
    monkeypatch.setattr("app.tts_probe._now", Clock([0.0, 0.25, 0.40]))
    response = _post(
        http,
        {"provider": "cloudtts", "sentences": ["Thanks Ian."], "runs_per_sentence": 1, "save_samples": 1},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provider"] == "cloudtts"
    assert body["method"] == "text:synthesize"
    assert body["voice"] == "en-AU-Chirp3-HD-Charon"
    assert body["language_code"] == "en-AU"
    assert body["sample_rate"] == 8000
    assert "model" not in body
    request = seen[0]
    assert request["url"] == CLOUD_TTS_URL
    assert SECRET not in request["url"]
    assert request["headers"]["X-Goog-Api-Key"] == SECRET
    assert "Authorization" not in request["headers"]
    assert request["json"] == {
        "input": {"text": "Thanks Ian."},
        "voice": {"languageCode": "en-AU", "name": "en-AU-Chirp3-HD-Charon"},
        "audioConfig": {"audioEncoding": "MULAW", "sampleRateHertz": 8000},
    }
    run = body["sentences"][0]["runs"][0]
    assert run["ttfa_ms"] == pytest.approx(250.0)
    assert run["total_ms"] == pytest.approx(400.0)
    assert run["conversion_ms"] == 0.0
    assert run["audio_bytes"] == 160
    assert run["audio_seconds"] == pytest.approx(0.02)
    assert run["error"] is None
    assert base64.b64decode(body["samples"][0]["audio_base64"]) == raw
    assert SECRET not in response.text
    assert SECRET not in caplog.text

    wav = http.get("/probe/tts/sample/0.wav", headers=_auth())
    assert wav.status_code == 200
    assert wav.headers["content-type"].startswith("audio/wav")
    assert struct.unpack_from("<H", wav.content, 20)[0] == 7
    assert struct.unpack_from("<I", wav.content, 24)[0] == 8000
    assert wav.content[44:] == raw


def test_cloudtts_linear16_conversion_is_separate(client, monkeypatch):
    import audioop

    http, _seen, settings = client
    settings.google_tts_api_key = SECRET
    pcm = struct.pack("<hhhh", 0, 8000, -8000, 16000)
    seen: list[dict] = []
    _install(
        monkeypatch,
        [httpx.Response(200, json={"audioContent": base64.b64encode(_pcm_wav(pcm)).decode("ascii")})],
        seen,
    )
    monkeypatch.setattr("app.tts_probe._now", Clock([0.0, 0.25, 1.0, 1.2, 1.5]))
    response = _post(
        http,
        {
            "provider": "cloudtts",
            "sentences": ["Is now a good time for a quick chat?"],
            "voice": "en-AU-Chirp3-HD-Achird",
            "save_samples": 1,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["voice"] == "en-AU-Chirp3-HD-Achird"
    assert seen[0]["json"]["voice"]["name"] == "en-AU-Chirp3-HD-Achird"
    assert seen[0]["json"]["voice"]["languageCode"] == "en-AU"
    run = body["sentences"][0]["runs"][0]
    stored = audioop.lin2ulaw(pcm, 2)
    assert run["ttfa_ms"] == pytest.approx(250.0)
    assert run["conversion_ms"] == pytest.approx(200.0)
    assert run["total_ms"] == pytest.approx(1500.0)
    assert run["audio_bytes"] == len(stored)
    assert run["error"] is None
    assert base64.b64decode(body["samples"][0]["audio_base64"]) == stored


def test_cloudtts_error_does_not_echo_the_key(client, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    http, _seen, settings = client
    settings.google_tts_api_key = SECRET
    seen: list[dict] = []
    _install(
        monkeypatch,
        [httpx.Response(403, json={"error": {"message": f"denied for {SECRET}"}})],
        seen,
    )
    monkeypatch.setattr("app.tts_probe._now", Clock([0.0, 0.3]))
    response = _post(http, {"provider": "cloudtts", "sentences": ["Thanks Ian."]})
    assert response.status_code == 200
    detail = response.json()["sentences"][0]["runs"][0]["error"]
    assert detail is not None
    assert "HTTP 403" in detail
    assert SECRET not in detail
    assert "[redacted]" in detail
    assert SECRET not in response.text
    assert SECRET not in caplog.text


def test_unknown_probe_provider_is_rejected(client):
    http, _seen, settings = client
    settings.gemini_api_key = "gem-test"
    response = _post(http, {"provider": "twilio"})
    assert response.status_code == 422
    assert "cloudtts" in response.json()["detail"]


PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----\nFAKE-CLOUD-TTS-PRIVATE-KEY\n-----END PRIVATE KEY-----\n"
SERVICE_ACCOUNT = {
    "type": "service_account",
    "project_id": "tack45-probe",
    "private_key_id": "probe-key-id",
    "private_key": PRIVATE_KEY,
    "client_email": "tts-probe@tack45-probe.iam.gserviceaccount.com",
    "client_id": "1234567890",
    "token_uri": "https://oauth2.googleapis.com/token",
}


class _FakeCredentials:
    def __init__(self) -> None:
        self.token: str | None = None
        self.expiry = None
        self.refresh_count = 0
        self.state = TokenState.INVALID
        self.requests: list[object] = []

    @property
    def token_state(self):
        return self.state

    def refresh(self, request) -> None:
        self.requests.append(request)
        self.refresh_count += 1
        self.token = f"ya29.probe-token-{self.refresh_count}"
        self.state = TokenState.FRESH


class _Chunk:
    def __init__(self, audio: bytes) -> None:
        self.audio_content = audio


class _StreamClient:
    def __init__(self, chunks: list[_Chunk], seen: list[list]) -> None:
        self.chunks = chunks
        self.seen = seen
        self.credentials = None

    def streaming_synthesize(self, requests):
        self.seen.append(list(requests))
        yield from self.chunks


def _patch_service_account(monkeypatch, credentials: _FakeCredentials, built: list[dict]) -> None:
    def from_info(info, **kwargs):
        built.append({"info": info, "kwargs": kwargs})
        return credentials

    monkeypatch.setattr(
        "google.oauth2.service_account.Credentials.from_service_account_info",
        staticmethod(from_info),
    )


def _service_account_json() -> str:
    return json.dumps(SERVICE_ACCOUNT)


def test_service_account_sends_bearer_and_caches_the_token(client, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    reset_cloudtts_auth_cache()
    http, _seen, settings = client
    settings.google_tts_api_key = SECRET
    settings.google_tts_credentials_json = _service_account_json()
    credentials = _FakeCredentials()
    built: list[dict] = []
    _patch_service_account(monkeypatch, credentials, built)
    raw = b"\x7f" * 80
    seen: list[dict] = []
    _install(
        monkeypatch,
        [
            httpx.Response(200, json={"audioContent": base64.b64encode(mulaw_wav(raw)).decode("ascii")}),
            httpx.Response(200, json={"audioContent": base64.b64encode(mulaw_wav(raw)).decode("ascii")}),
            httpx.Response(200, json={"audioContent": base64.b64encode(mulaw_wav(raw)).decode("ascii")}),
        ],
        seen,
    )
    monkeypatch.setattr("app.tts_probe._now", _CountingClock())
    try:
        first = _post(http, {"provider": "cloudtts", "sentences": ["Thanks Ian."], "save_samples": 0})
        second = _post(http, {"provider": "cloudtts", "sentences": ["Thanks Ian."], "save_samples": 0})
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert credentials.refresh_count == 1
        assert len(built) == 1
        assert built[0]["kwargs"]["scopes"] == [CLOUD_PLATFORM_SCOPE]
        assert built[0]["info"]["private_key"] == PRIVATE_KEY
        assert seen[0]["headers"]["Authorization"] == "Bearer ya29.probe-token-1"
        assert seen[1]["headers"]["Authorization"] == "Bearer ya29.probe-token-1"
        for request in seen[:2]:
            assert "X-Goog-Api-Key" not in request["headers"]
            assert SECRET not in request["url"]
            assert PRIVATE_KEY not in json.dumps(request["json"])
        credentials.state = TokenState.STALE
        third = _post(http, {"provider": "cloudtts", "sentences": ["Thanks Ian."], "save_samples": 0})
        assert third.status_code == 200, third.text
        assert credentials.refresh_count == 2
        assert len(built) == 1
        assert seen[2]["headers"]["Authorization"] == "Bearer ya29.probe-token-2"
        combined = first.text + second.text + third.text + caplog.text
        assert SECRET not in combined
        assert PRIVATE_KEY not in combined
        assert "ya29.probe-token" not in combined
        assert "FAKE-CLOUD-TTS-PRIVATE-KEY" not in combined
    finally:
        reset_cloudtts_auth_cache()


def test_invalid_credentials_json_is_not_echoed(client, caplog):
    caplog.set_level(logging.DEBUG)
    reset_cloudtts_auth_cache()
    http, _seen, settings = client
    settings.google_tts_api_key = SECRET
    settings.google_tts_credentials_json = "not-json FAKE-CLOUD-TTS-PRIVATE-KEY"
    try:
        response = _post(http, {"provider": "cloudtts", "sentences": ["Thanks Ian."]})
        assert response.status_code == 400
        assert response.json()["detail"] == "GOOGLE_TTS_CREDENTIALS_JSON is not valid JSON."
        assert "FAKE-CLOUD-TTS-PRIVATE-KEY" not in response.text
        assert SECRET not in response.text
        assert "FAKE-CLOUD-TTS-PRIVATE-KEY" not in caplog.text
    finally:
        reset_cloudtts_auth_cache()


def test_malformed_service_account_does_not_echo_the_key(client, caplog):
    caplog.set_level(logging.DEBUG)
    reset_cloudtts_auth_cache()
    http, _seen, settings = client
    settings.google_tts_credentials_json = json.dumps({"private_key": PRIVATE_KEY})
    try:
        response = _post(http, {"provider": "cloudtts", "sentences": ["Thanks Ian."]})
        assert response.status_code == 400
        assert response.json()["detail"] == "GOOGLE_TTS_CREDENTIALS_JSON is not a service account key."
        assert "FAKE-CLOUD-TTS-PRIVATE-KEY" not in response.text
        assert PRIVATE_KEY not in response.text
        assert "FAKE-CLOUD-TTS-PRIVATE-KEY" not in caplog.text
    finally:
        reset_cloudtts_auth_cache()


def test_token_refresh_failure_is_redacted(client, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    reset_cloudtts_auth_cache()
    http, _seen, settings = client
    settings.google_tts_credentials_json = _service_account_json()
    credentials = _FakeCredentials()

    def explode(request) -> None:
        raise RuntimeError(f"invalid_grant for {PRIVATE_KEY} token ya29.super-secret-token")

    credentials.refresh = explode  # type: ignore[method-assign]
    built: list[dict] = []
    _patch_service_account(monkeypatch, credentials, built)
    try:
        response = _post(http, {"provider": "cloudtts", "sentences": ["Thanks Ian."]})
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "token refresh failed" in detail
        assert "invalid_grant" in detail
        assert "FAKE-CLOUD-TTS-PRIVATE-KEY" not in detail
        assert "ya29.super-secret-token" not in detail
        assert PRIVATE_KEY not in response.text
        assert "ya29.super-secret-token" not in caplog.text
        assert "FAKE-CLOUD-TTS-PRIVATE-KEY" not in caplog.text
    finally:
        reset_cloudtts_auth_cache()


def test_streaming_reports_ttfa_for_both_methods(client, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    reset_cloudtts_auth_cache()
    http, _seen, settings = client
    settings.google_tts_api_key = SECRET
    settings.google_tts_credentials_json = _service_account_json()
    credentials = _FakeCredentials()
    _patch_service_account(monkeypatch, credentials, [])
    unary = b"\x7f" * 160
    first_chunk = b"\x00" * 40
    second_chunk = b"\xff" * 40
    seen_http: list[dict] = []
    seen_rpc: list[list] = []
    _install(
        monkeypatch,
        [httpx.Response(200, json={"audioContent": base64.b64encode(mulaw_wav(unary)).decode("ascii")})],
        seen_http,
    )
    stream = _StreamClient([_Chunk(b""), _Chunk(first_chunk), _Chunk(second_chunk)], seen_rpc)

    def client_factory(passed):
        stream.credentials = passed
        return stream

    monkeypatch.setattr("app.tts_probe._text_to_speech_client", client_factory)
    monkeypatch.setattr(
        "app.tts_probe._now",
        Clock([0.0, 0.25, 0.40, 1.0, 1.08, 1.30]),
    )
    try:
        response = _post(
            http,
            {
                "provider": "cloudtts",
                "sentences": ["Thanks Ian."],
                "runs_per_sentence": 1,
                "save_samples": 1,
                "streaming": True,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["method"] == "text:synthesize"
        assert body["voice"] == "en-AU-Chirp3-HD-Charon"
        unary_run = body["sentences"][0]["runs"][0]
        assert unary_run["ttfa_ms"] == pytest.approx(250.0)
        assert unary_run["total_ms"] == pytest.approx(400.0)
        assert unary_run["error"] is None
        assert seen_http[0]["headers"]["Authorization"] == "Bearer ya29.probe-token-1"
        assert "X-Goog-Api-Key" not in seen_http[0]["headers"]
        streamed = body["streaming"]
        assert streamed["method"] == "streaming_synthesize"
        stream_run = streamed["sentences"][0]["runs"][0]
        assert stream_run["ttfa_ms"] == pytest.approx(80.0)
        assert stream_run["total_ms"] == pytest.approx(300.0)
        assert stream_run["conversion_ms"] == 0.0
        assert stream_run["audio_bytes"] == 80
        assert stream_run["error"] is None
        assert body["ttfa_ms"]["min"] == pytest.approx(250.0)
        assert streamed["ttfa_ms"]["min"] == pytest.approx(80.0)
        assert base64.b64decode(body["samples"][0]["audio_base64"]) == unary
        assert base64.b64decode(streamed["samples"][0]["audio_base64"]) == first_chunk + second_chunk
        assert stream.credentials is credentials
        config_request, input_request = seen_rpc[0]
        from google.cloud import texttospeech

        voice = config_request.streaming_config.voice
        audio_config = config_request.streaming_config.streaming_audio_config
        assert voice.name == "en-AU-Chirp3-HD-Charon"
        assert voice.language_code == "en-AU"
        assert audio_config.audio_encoding == texttospeech.AudioEncoding.MULAW
        assert audio_config.sample_rate_hertz == 8000
        assert input_request.input.text == "Thanks Ian."
        wav = http.get("/probe/tts/sample/0.wav", headers=_auth())
        assert wav.status_code == 200
        assert wav.content[44:] == unary
        assert SECRET not in response.text
        assert PRIVATE_KEY not in response.text
        assert "ya29.probe-token" not in response.text
        assert "FAKE-CLOUD-TTS-PRIVATE-KEY" not in caplog.text
    finally:
        reset_cloudtts_auth_cache()


def test_streaming_requires_the_service_account(client, monkeypatch):
    http, _seen, settings = client
    settings.google_tts_api_key = SECRET
    settings.google_tts_credentials_json = ""
    called = {"grpc": 0}

    def forbid(_credentials):
        called["grpc"] += 1
        raise AssertionError("streaming client should not be built")

    monkeypatch.setattr("app.tts_probe._text_to_speech_client", forbid)
    response = _post(http, {"provider": "cloudtts", "streaming": True, "sentences": ["Thanks Ian."]})
    assert response.status_code == 400
    assert response.json()["detail"] == "GOOGLE_TTS_CREDENTIALS_JSON not set"
    assert SECRET not in response.text
    assert called["grpc"] == 0
    settings.gemini_api_key = "gem-test"
    gemini = _post(http, {"provider": "gemini", "streaming": True, "sentences": ["Thanks Ian."]})
    assert gemini.status_code == 422
    assert "cloudtts" in gemini.json()["detail"]
    bad = _post(http, {"provider": "cloudtts", "streaming": "yes", "sentences": ["Thanks Ian."]})
    assert bad.status_code == 422
    assert "boolean" in bad.json()["detail"]


def test_streaming_conversion_is_outside_ttfa(monkeypatch):
    import audioop

    pcm = struct.pack("<hh", 0, 16000)
    seen: list[list] = []
    stream = _StreamClient([_Chunk(_pcm_wav(pcm))], seen)
    monkeypatch.setattr("app.tts_probe._text_to_speech_client", lambda _credentials: stream)
    monkeypatch.setattr("app.tts_probe._now", Clock([0.0, 0.10, 0.50, 0.70, 0.90]))
    result = synthesize_cloudtts_streaming(object(), "Thanks Ian.", "en-AU-Chirp3-HD-Achird")
    assert result["error"] is None
    assert result["ttfa_ms"] == pytest.approx(100.0)
    assert result["conversion_ms"] == pytest.approx(200.0)
    assert result["total_ms"] == pytest.approx(900.0)
    assert result["audio"] == audioop.lin2ulaw(pcm, 2)
    voice = seen[0][0].streaming_config.voice
    assert voice.name == "en-AU-Chirp3-HD-Achird"
    assert voice.language_code == "en-AU"


def test_streaming_failure_redacts_the_token(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    class _Boom:
        def streaming_synthesize(self, requests):
            list(requests)
            raise RuntimeError("rpc failed ya29.super-secret-token and FAKE-CLOUD-TTS-PRIVATE-KEY")

    monkeypatch.setattr("app.tts_probe._text_to_speech_client", lambda _credentials: _Boom())
    monkeypatch.setattr("app.tts_probe._now", Clock([5.0, 5.2]))
    credentials = _FakeCredentials()
    credentials.token = "ya29.super-secret-token"
    credentials.state = TokenState.FRESH
    result = synthesize_cloudtts_streaming(
        credentials,
        "Thanks Ian.",
        "en-AU-Chirp3-HD-Charon",
        [PRIVATE_KEY, "ya29.super-secret-token"],
    )
    assert result["ttfa_ms"] is None
    assert result["audio"] == b""
    assert "Cloud TTS stream failed" in result["error"]
    assert "ya29.super-secret-token" not in result["error"]
    assert "FAKE-CLOUD-TTS-PRIVATE-KEY" not in result["error"]
    assert "[redacted]" in result["error"]
    assert "ya29.super-secret-token" not in caplog.text


def test_speech_client_is_reused_for_the_same_credentials(monkeypatch):
    reset_cloudtts_auth_cache()
    created: list[object] = []

    class _Marker:
        def __init__(self, credentials) -> None:
            self.credentials = credentials
            created.append(self)

    monkeypatch.setattr("google.cloud.texttospeech.TextToSpeechClient", _Marker)
    from app.tts_probe import _text_to_speech_client

    credentials = object()
    try:
        first = _text_to_speech_client(credentials)
        second = _text_to_speech_client(credentials)
        assert first is second
        assert created == [first]
        assert first.credentials is credentials
    finally:
        reset_cloudtts_auth_cache()


class _CountingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += 0.05
        return current
