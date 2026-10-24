import base64
import json
import struct

import pytest

from app.config import missing_dial_configuration
from app.tts_probe import (
    DEFAULT_SENTENCES,
    DEFAULT_STYLE,
    DEFAULT_VOICE,
    audio_chunks_from_line,
    interaction_body,
    mulaw_wav,
    timed_audio,
    timing_summary,
)
from tests.conftest import make_settings
from tests.test_api import _auth


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_default_sentences_cover_short_and_medium_lines():
    assert len(DEFAULT_SENTENCES) == 20
    assert "Thanks Ian." in DEFAULT_SENTENCES
    assert "Is now a good time for a quick chat?" in DEFAULT_SENTENCES
    assert (
        "Adam usually starts with just one small job, sees if it helps, and goes from there."
        in DEFAULT_SENTENCES
    )
    assert any("Goodbye" in line for line in DEFAULT_SENTENCES)


def test_interaction_requests_mulaw_8k_and_a_voice_string():
    body = interaction_body(
        "Thanks Ian.",
        model="gemini-3.8-flash-lite-tts",
        style=DEFAULT_STYLE,
        voice=DEFAULT_VOICE,
    )
    assert body["model"] == "gemini-3.8-flash-lite-tts"
    assert body["stream"] is True
    assert body["response_format"] == {
        "type": "audio",
        "mime_type": "audio/mulaw",
        "sample_rate": 8000,
    }
    assert body["generation_config"]["speech_config"] == [{"voice": "Kore"}]
    text = body["input"][0]["content"][0]
    assert text["type"] == "text"
    assert text["text"] == "Thanks Ian."
    assert text["annotations"] == [{"type": "speech_metadata", "style": DEFAULT_STYLE}]


@pytest.mark.asyncio
async def test_stream_timing_and_percentiles():
    clock = Clock()

    async def lines():
        clock.now = 0.10
        yield 'data: {"event_type":"step.start"}'
        clock.now = 0.25
        yield "data: " + json.dumps(
            {
                "event_type": "step.delta",
                "delta": {"type": "audio", "data": base64.b64encode(b"\xff\xff").decode()},
            }
        )
        clock.now = 0.80
        yield "data: " + json.dumps(
            {
                "event_type": "step.delta",
                "delta": {"type": "audio", "data": base64.b64encode(b"\x00\x00").decode()},
            }
        )

    audio, first, end, error = await timed_audio(lines(), clock=clock, start=0.0)
    assert error is None
    assert audio == b"\xff\xff\x00\x00"
    assert first == 0.25
    assert end == 0.80
    ttfa_ms = (first - 0.0) * 1000
    total_ms = (end - 0.0) * 1000
    assert ttfa_ms == 250
    assert total_ms == 800
    assert len(audio) / 8000 == 0.0005
    summary = timing_summary([10, 20, 30, 40])
    assert summary["min"] == 10
    assert summary["max"] == 40
    assert summary["p50"] == 25
    assert summary["p90"] == 37
    assert timing_summary([]) == {"p50": None, "p90": None, "min": None, "max": None}
    assert audio_chunks_from_line("data: [DONE]") == []


def test_mulaw_wav_header_is_8khz_format_7():
    wav = mulaw_wav(b"\xff\x00")
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert struct.unpack_from("<H", wav, 20)[0] == 7
    assert struct.unpack_from("<H", wav, 22)[0] == 1
    assert struct.unpack_from("<I", wav, 24)[0] == 8000
    assert struct.unpack_from("<H", wav, 34)[0] == 8
    assert wav[36:40] == b"data"
    assert wav[44:] == b"\xff\x00"


def test_gemini_key_is_not_required_to_dial(tmp_path):
    settings = make_settings(tmp_path, gemini_api_key="")
    missing = missing_dial_configuration(settings)
    assert "GEMINI_API_KEY" not in missing
    assert settings.gemini_api_key == ""


def test_probe_requires_bearer_and_gemini_key(client):
    http, _seen, settings = client
    settings.gemini_api_key = ""
    missing = http.post("/probe/tts", json={})
    assert missing.status_code == 401
    assert missing.json()["code"] == "unauthorized"
    refused = http.post("/probe/tts", json={}, headers=_auth())
    assert refused.status_code == 503
    assert refused.json()["detail"] == "GEMINI_API_KEY not set"
    hidden = http.get("/probe/tts/sample/0.wav")
    assert hidden.status_code == 401


def test_probe_returns_timings_and_wav(client, monkeypatch):
    http, _seen, settings = client
    settings.gemini_api_key = "gem-test"
    seen: list[dict] = []

    async def fake(api_key, body, *, clock=None):
        seen.append(body)
        assert api_key == "gem-test"
        return {
            "ttfa_ms": 125.0,
            "total_ms": 400.0,
            "audio": b"\x7f" * 1600,
            "error": None,
        }

    monkeypatch.setattr("app.tts_probe.synthesize", fake)
    response = http.post(
        "/probe/tts",
        json={"sentences": ["Thanks Ian."], "runs_per_sentence": 2, "save_samples": 1, "voice": "Achird"},
        headers=_auth(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "gemini-3.8-flash-lite-tts"
    assert body["voice"] == "Achird"
    assert body["style"] == DEFAULT_STYLE
    assert body["sample_rate"] == 8000
    assert len(seen) == 2
    assert seen[0]["generation_config"]["speech_config"] == [{"voice": "Achird"}]
    assert seen[0]["response_format"]["mime_type"] == "audio/mulaw"
    assert seen[0]["stream"] is True
    row = body["sentences"][0]
    assert row["sentence"] == "Thanks Ian."
    assert len(row["runs"]) == 2
    assert row["runs"][0]["ttfa_ms"] == 125.0
    assert row["runs"][0]["total_ms"] == 400.0
    assert row["runs"][0]["audio_bytes"] == 1600
    assert row["runs"][0]["audio_seconds"] == 0.2
    assert body["ttfa_ms"]["min"] == 125.0
    assert body["ttfa_ms"]["max"] == 125.0
    assert body["total_ms"]["p50"] == 400.0
    assert len(body["samples"]) == 1
    assert base64.b64decode(body["samples"][0]["audio_base64"]) == b"\x7f" * 1600

    wav = http.get("/probe/tts/sample/0.wav", headers=_auth())
    assert wav.status_code == 200
    assert wav.headers["content-type"].startswith("audio/wav")
    assert wav.content[8:12] == b"WAVE"
    assert struct.unpack_from("<H", wav.content, 20)[0] == 7
    assert struct.unpack_from("<I", wav.content, 24)[0] == 8000
    assert wav.content[44:] == b"\x7f" * 1600
    missing = http.get("/probe/tts/sample/1.wav", headers=_auth())
    assert missing.status_code == 404
