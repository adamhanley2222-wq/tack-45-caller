"""Admin-only Gemini TTS latency probe.

This does not dial, open a voice session, or change call configuration.
Audio is requested as 8 kHz G.711 μ-law, the same codec Twilio media uses.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import struct
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.api import APIError, require_bearer
from app.config import Settings

logger = logging.getLogger("tack45.tts_probe")

router = APIRouter()

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_MODEL = "gemini-3.8-flash-lite-tts"
DEFAULT_STYLE = "Australian English accent, relaxed and friendly"
DEFAULT_VOICE = "Kore"
SAMPLE_RATE = 8000
MAX_SENTENCES = 40
MAX_RUNS = 5
MAX_SAMPLES = 10

DEFAULT_SENTENCES = [
    "Thanks Ian.",
    "Is now a good time for a quick chat?",
    "Adam usually starts with just one small job, sees if it helps, and goes from there.",
    "Thanks. I'll leave it there. Goodbye.",
    "Thanks Sam. I'm an AI assistant calling for Tack45, a Sydney-based AI consultancy.",
    "Hi, is this Sam?",
    "Would a short chat about practical AI for the business be useful?",
    "When would suit for a call back?",
    "I'll remove this number and won't call again.",
    "No worries. Thanks for your time. Goodbye.",
    "Ha, I'll leave the cooking to you. I'm just here to talk about AI for your business.",
    "Adam Hanley, who runs Tack45, is based on the Northern Beaches.",
    "The chat with Adam is free, and there's no cost.",
    "Would you be open to a chat with Adam, just to talk it through?",
    "If you'd like a colleague to hear this, they're welcome to ring back on this number anytime.",
    "Find is a look at the day-to-day work, then a short list of what to build and what to leave.",
    "You don't need technical staff for this.",
    "The readiness check is ten questions and takes about two minutes.",
    "I can send the details to adam@tack45.com.au.",
    "Sorry, I didn't catch that. Is now a good time for a quick chat?",
]


def server_region() -> str | None:
    for name in (
        "RAILWAY_REPLICA_REGION",
        "RAILWAY_REGION",
        "RAILWAY_DEPLOYMENT_REGION",
    ):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None


def interaction_body(text: str, *, model: str, style: str, voice: str) -> dict[str, Any]:
    """One Interactions API speech request. Voice is the documented speech_config array."""
    return {
        "model": model,
        "store": False,
        "input": [
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "annotations": [
                            {
                                "type": "speech_metadata",
                                "style": style,
                            }
                        ],
                    }
                ],
            }
        ],
        "response_format": {
            "type": "audio",
            "mime_type": "audio/mulaw",
            "sample_rate": SAMPLE_RATE,
        },
        "generation_config": {
            "speech_config": [
                {"voice": voice},
            ]
        },
        "stream": True,
    }


def audio_chunks_from_line(line: str) -> list[bytes]:
    """Pull raw audio bytes out of one SSE or JSON line. Non-audio lines yield nothing."""
    text = line.strip()
    if not text or text.startswith(":"):
        return []
    if text.startswith("data:"):
        text = text[5:].strip()
    if not text or text == "[DONE]":
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    return _audio_from_payload(payload)


def _audio_from_payload(payload: dict) -> list[bytes]:
    found: list[bytes] = []
    delta = payload.get("delta")
    if isinstance(delta, dict) and delta.get("type") == "audio":
        chunk = _decode_audio(delta.get("data"))
        if chunk:
            found.append(chunk)
    content = payload.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "audio":
                chunk = _decode_audio(item.get("data"))
                if chunk:
                    found.append(chunk)
    steps = payload.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                found.extend(_audio_from_payload(step))
    return found


def _decode_audio(data: Any) -> bytes:
    if not isinstance(data, str) or not data:
        return b""
    try:
        return base64.b64decode(data, validate=True)
    except (ValueError, TypeError):
        return b""


def stream_error_message(line: str) -> str | None:
    text = line.strip()
    if text.startswith("data:"):
        text = text[5:].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("event_type") not in {"error", "interaction.error"} and payload.get("type") != "error":
        if "error" not in payload:
            return None
    detail = payload.get("error")
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)[:500]
    if isinstance(detail, str) and detail:
        return detail[:500]
    if payload.get("event_type") in {"error", "interaction.error"}:
        return str(payload)[:500]
    return None


async def timed_audio(
    lines: AsyncIterator[str],
    *,
    clock: Callable[[], float],
    start: float,
) -> tuple[bytes, float | None, float, str | None]:
    """Read a Gemini stream. The clock starts at ``start``, before the request is sent."""
    first: float | None = None
    audio = bytearray()
    error: str | None = None
    async for line in lines:
        message = stream_error_message(line)
        if message and error is None:
            error = message
        for chunk in audio_chunks_from_line(line):
            if first is None:
                first = clock()
            audio.extend(chunk)
    return bytes(audio), first, clock(), error


def percentile(sorted_values: list[float], p: float) -> float | None:
    """Linear percentile. ``p`` is 0-100. Empty input is None."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (p / 100)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * weight


def timing_summary(values: list[float]) -> dict[str, float | None]:
    ordered = sorted(values)
    return {
        "p50": percentile(ordered, 50),
        "p90": percentile(ordered, 90),
        "min": ordered[0] if ordered else None,
        "max": ordered[-1] if ordered else None,
    }


def mulaw_wav(payload: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Wrap headerless μ-law in an 8 kHz mono WAV (format code 7)."""
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(payload),
        b"WAVE",
        b"fmt ",
        16,
        7,
        1,
        rate,
        rate,
        1,
        8,
        b"data",
        len(payload),
    )
    return header + payload


def _clips(app) -> list[dict[str, Any]]:
    clips = getattr(app.state, "tts_probe_clips", None)
    if clips is None:
        clips = []
        app.state.tts_probe_clips = clips
    return clips


def _parse_probe_body(payload: Any) -> dict[str, Any]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise APIError(422, "invalid_body", "Body must be a JSON object.")
    sentences = payload.get("sentences", None)
    if sentences is None:
        chosen = list(DEFAULT_SENTENCES)
    else:
        if not isinstance(sentences, list) or not sentences:
            raise APIError(422, "invalid_body", "sentences must be a non-empty list of strings.")
        if len(sentences) > MAX_SENTENCES:
            raise APIError(422, "invalid_body", f"sentences is limited to {MAX_SENTENCES}.")
        chosen = []
        for item in sentences:
            if not isinstance(item, str) or not item.strip():
                raise APIError(422, "invalid_body", "Each sentence must be a non-empty string.")
            chosen.append(item.strip())
    runs = payload.get("runs_per_sentence", 1)
    samples = payload.get("save_samples", 3)
    style = payload.get("style", DEFAULT_STYLE)
    voice = payload.get("voice", DEFAULT_VOICE)
    model = payload.get("model", DEFAULT_MODEL)
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1 or runs > MAX_RUNS:
        raise APIError(422, "invalid_body", f"runs_per_sentence must be an integer from 1 to {MAX_RUNS}.")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0 or samples > MAX_SAMPLES:
        raise APIError(422, "invalid_body", f"save_samples must be an integer from 0 to {MAX_SAMPLES}.")
    if not isinstance(style, str) or not style.strip():
        raise APIError(422, "invalid_body", "style must be a non-empty string.")
    if not isinstance(voice, str) or not voice.strip():
        raise APIError(422, "invalid_body", "voice must be a non-empty string.")
    if not isinstance(model, str) or not model.strip():
        raise APIError(422, "invalid_body", "model must be a non-empty string.")
    return {
        "sentences": chosen,
        "runs_per_sentence": runs,
        "save_samples": samples,
        "style": style.strip(),
        "voice": voice.strip(),
        "model": model.strip(),
    }


async def synthesize(
    api_key: str,
    body: dict[str, Any],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """POST one streaming interaction. The clock is read immediately before send."""
    start = clock()
    error: str | None = None
    lines: AsyncIterator[str]
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream(
                "POST",
                INTERACTIONS_URL,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=body,
            ) as response:
                if response.status_code >= 400:
                    raw = (await response.aread()).decode("utf-8", errors="replace")
                    error = f"HTTP {response.status_code}: {raw[:500]}"

                    async def _empty() -> AsyncIterator[str]:
                        if False:
                            yield ""

                    lines = _empty()
                else:
                    lines = response.aiter_lines()
                audio, first, end, stream_error = await timed_audio(lines, clock=clock, start=start)
        except httpx.HTTPError as exc:
            audio = b""
            first = None
            end = clock()
            error = str(exc)[:500]
            stream_error = None
    if stream_error and error is None:
        error = stream_error
    if error is None and not audio:
        error = "Gemini stream contained no audio."
    ttfa_ms = None if first is None else (first - start) * 1000
    return {
        "ttfa_ms": ttfa_ms,
        "total_ms": (end - start) * 1000,
        "audio": audio,
        "error": error,
    }


@router.post("/probe/tts")
async def probe_tts(request: Request) -> dict[str, Any]:
    require_bearer(request)
    settings: Settings = request.app.state.settings
    if not settings.gemini_api_key:
        raise APIError(503, "not_configured", "GEMINI_API_KEY not set")
    raw = await request.body()
    if not raw.strip():
        payload: Any = {}
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise APIError(422, "invalid_body", "Body must be JSON.") from exc
    options = _parse_probe_body(payload)
    rows: list[dict[str, Any]] = []
    ttfa_values: list[float] = []
    total_values: list[float] = []
    clips: list[dict[str, Any]] = []
    for sentence in options["sentences"]:
        runs: list[dict[str, Any]] = []
        for _ in range(options["runs_per_sentence"]):
            body = interaction_body(
                sentence,
                model=options["model"],
                style=options["style"],
                voice=options["voice"],
            )
            result = await synthesize(settings.gemini_api_key, body)
            audio: bytes = result["audio"]
            row = {
                "ttfa_ms": result["ttfa_ms"],
                "total_ms": result["total_ms"],
                "audio_bytes": len(audio),
                "audio_seconds": len(audio) / SAMPLE_RATE,
                "error": result["error"],
            }
            runs.append(row)
            if result["ttfa_ms"] is not None and result["error"] is None:
                ttfa_values.append(result["ttfa_ms"])
                total_values.append(result["total_ms"])
            if audio and len(clips) < options["save_samples"]:
                clips.append({"sentence": sentence, "audio": audio})
        rows.append({"sentence": sentence, "runs": runs})
    request.app.state.tts_probe_clips = clips
    return {
        "model": options["model"],
        "style": options["style"],
        "voice": options["voice"],
        "mime_type": "audio/mulaw",
        "sample_rate": SAMPLE_RATE,
        "region": server_region(),
        "sentences": rows,
        "ttfa_ms": timing_summary(ttfa_values),
        "total_ms": timing_summary(total_values),
        "samples": [
            {
                "n": index,
                "sentence": clip["sentence"],
                "audio_base64": base64.b64encode(clip["audio"]).decode("ascii"),
            }
            for index, clip in enumerate(clips)
        ],
    }


@router.get("/probe/tts/sample/{n}.wav")
async def probe_sample(n: int, request: Request) -> Response:
    require_bearer(request)
    clips = _clips(request.app)
    if n < 0 or n >= len(clips):
        raise APIError(404, "not_found", "No probe sample at that index. POST /probe/tts first.")
    wav = mulaw_wav(clips[n]["audio"])
    return Response(content=wav, media_type="audio/wav")
