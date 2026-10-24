"""Cloud TTS probe. HTTP is mocked. Nothing is sent to Google."""

from __future__ import annotations

import base64
import json
import logging
import struct

import httpx
import pytest

from app.config import Settings, missing_dial_configuration
from app.tts_probe import CLOUD_TTS_URL, mulaw_from_cloud_audio, mulaw_wav
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


def test_cloudtts_requires_bearer_and_a_key(client, caplog):
    caplog.set_level(logging.DEBUG)
    http, _seen, settings = client
    settings.gemini_api_key = ""
    settings.google_tts_api_key = ""
    hidden = http.post("/probe/tts", json={"provider": "cloudtts"})
    assert hidden.status_code == 401
    refused = _post(http, {"provider": "cloudtts", "sentences": ["Thanks Ian."]})
    assert refused.status_code == 400
    assert refused.json()["detail"] == "GOOGLE_TTS_API_KEY not set"
    assert SECRET not in refused.text
    assert "GOOGLE_TTS_API_KEY not set" in refused.text


def test_google_tts_key_is_not_required_to_dial(tmp_path):
    settings = make_settings(tmp_path, google_tts_api_key="")
    missing = missing_dial_configuration(settings)
    assert "GOOGLE_TTS_API_KEY" not in missing
    plain = Settings(_env_file=None)
    assert plain.google_tts_api_key == ""
    assert "GOOGLE_TTS_API_KEY" not in missing_dial_configuration(plain)


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
