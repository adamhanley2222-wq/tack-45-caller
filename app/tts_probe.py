"""Admin-only TTS latency probe.

Gemini Flash-Lite is the default. ``provider: "cloudtts"`` measures Google
Cloud Text-to-Speech Chirp 3 HD instead. This does not dial, open a voice
session, or change call configuration. Audio is 8 kHz G.711 μ-law, the same
codec Twilio media uses.

Cloud TTS prefers a service account in ``GOOGLE_TTS_CREDENTIALS_JSON``. The
probe mints an OAuth access token (cloud-platform scope), caches it, and
refreshes it when it expires. ``GOOGLE_TTS_API_KEY`` is only a fallback.
``streaming: true`` also measures gRPC StreamingSynthesize and reports time
to first audio for both methods. Credentials and tokens are not logged.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import struct
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.api import APIError, require_bearer
from app.config import Settings
from app.pcm_bridge import linear16_to_mulaw8k, mulaw_to_8k

logger = logging.getLogger("tack45.tts_probe")

router = APIRouter()

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_MODEL = "gemini-3.8-flash-lite-tts"
DEFAULT_STYLE = "Australian English accent, relaxed and friendly"
DEFAULT_VOICE = "Kore"
SAMPLE_RATE = 8000
CLOUD_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
CLOUD_TTS_METHOD = "text:synthesize"
CLOUD_TTS_STREAM_METHOD = "streaming_synthesize"
CLOUD_TTS_VOICE = "en-AU-Chirp3-HD-Charon"
CLOUD_TTS_LANGUAGE = "en-AU"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_ACCESS_TOKEN_RE = re.compile(r"ya29\.[A-Za-z0-9_\-.]+")
_auth_lock = threading.Lock()
_auth_cache: dict[str, Any] = {}
_clients: dict[int, Any] = {}
GEMINI_METHOD = "interactions"
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


def interaction_body(
    text: str,
    *,
    model: str,
    style: str,
    voice: str,
    language: str | None = None,
) -> dict[str, Any]:
    """One Interactions API speech request. Voice is the documented speech_config array.

    ``style`` is speech metadata and is not part of the spoken transcript.
    ``language`` is the SpeechConfig language (for example ``en-AU``). Omit it
    and the body matches the original probe request.
    """
    speech: dict[str, Any] = {"voice": voice}
    if language:
        speech["language"] = language
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
            "speech_config": [speech],
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


def cloudtts_body(text: str, voice: str) -> dict[str, Any]:
    """Unary Cloud TTS request. Chirp 3 HD, μ-law at 8 kHz, language en-AU."""
    return {
        "input": {"text": text},
        "voice": {"languageCode": CLOUD_TTS_LANGUAGE, "name": voice},
        "audioConfig": {"audioEncoding": "MULAW", "sampleRateHertz": SAMPLE_RATE},
    }


def _wav_format_and_data(payload: bytes) -> tuple[int, int, int, bytes] | None:
    """Return format code, sample rate, bits, and the data chunk. None if this is not WAV."""
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        return None
    offset = 12
    audio_format = rate = bits = None
    data = None
    while offset + 8 <= len(payload):
        chunk_id = payload[offset : offset + 4]
        size = struct.unpack_from("<I", payload, offset + 4)[0]
        body = payload[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt " and len(body) >= 16:
            audio_format, _channels, rate, _byte_rate, _block, bits = struct.unpack_from("<HHIIHH", body, 0)
        elif chunk_id == b"data":
            data = body
        offset += 8 + size + (size % 2)
    if audio_format is None or rate is None or bits is None or data is None:
        return None
    return audio_format, rate, bits, data


def mulaw_from_cloud_audio(
    payload: bytes,
    clock: Callable[[], float],
) -> tuple[bytes, float, str | None]:
    """Raw 8 kHz μ-law, conversion milliseconds, and an error.

    Cloud TTS wraps MULAW and LINEAR16 in a WAV header. A headerless body is
    treated as μ-law already. LINEAR16 is converted here, and only that work
    is on the clock.
    """
    parsed = _wav_format_and_data(payload)
    if parsed is None:
        return payload, 0.0, None
    audio_format, rate, bits, data = parsed
    if audio_format == 7 and rate == SAMPLE_RATE:
        return data, 0.0, None
    if audio_format not in {1, 7} or (audio_format == 1 and bits != 16):
        return b"", 0.0, f"Cloud TTS audio format {audio_format} at {rate} Hz is not μ-law or 16-bit PCM."
    started = clock()
    if audio_format == 7:
        mulaw = mulaw_to_8k(data, rate)
    else:
        mulaw = linear16_to_mulaw8k(data, rate)
    return mulaw, (clock() - started) * 1000, None


def _now() -> float:
    """Indirection so tests can replace the clock without touching time.monotonic."""
    return time.monotonic()


def _redact(text: str, *secrets: str) -> str:
    """Strip private keys, access tokens, and any other secret values."""
    cleaned = _PRIVATE_KEY_RE.sub("[redacted]", text)
    cleaned = _ACCESS_TOKEN_RE.sub("[redacted]", cleaned)
    pieces: list[str] = []
    for secret in secrets:
        if not secret:
            continue
        pieces.append(secret)
        pieces.extend(line.strip() for line in secret.splitlines() if len(line.strip()) >= 12)
    for piece in sorted(set(pieces), key=len, reverse=True):
        cleaned = cleaned.replace(piece, "[redacted]")
    return cleaned[:500]


def reset_cloudtts_auth_cache() -> None:
    """Drop cached service-account credentials and speech clients. Tests use this."""
    with _auth_lock:
        _auth_cache.clear()
        _clients.clear()


def _credential_secrets(raw: str, *extra: str) -> list[str]:
    found = [raw, *extra]
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        info = None
    if isinstance(info, dict):
        for key in ("private_key", "private_key_id", "client_email", "client_id", "token_uri"):
            value = info.get(key)
            if isinstance(value, str) and value:
                found.append(value)
    return [item for item in found if item]


def _needs_refresh(credentials: Any) -> bool:
    """True when the access token is missing, expired, or inside the refresh window."""
    from google.auth.credentials import TokenState

    state = getattr(credentials, "token_state", None)
    if state is None:
        return not bool(getattr(credentials, "valid", False))
    return state != TokenState.FRESH


def _service_account_credentials(info: dict[str, Any]) -> Any:
    from google.oauth2 import service_account

    return service_account.Credentials.from_service_account_info(
        info,
        scopes=[CLOUD_PLATFORM_SCOPE],
    )


def _auth_request() -> Any:
    from google.auth.transport.requests import Request

    return Request()


def _build_credentials(raw: str) -> Any:
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        raise APIError(
            400,
            "not_configured",
            "GOOGLE_TTS_CREDENTIALS_JSON is not valid JSON.",
        ) from None
    if not isinstance(info, dict):
        raise APIError(
            400,
            "not_configured",
            "GOOGLE_TTS_CREDENTIALS_JSON is not a service account key.",
        )
    try:
        return _service_account_credentials(info)
    except Exception:
        raise APIError(
            400,
            "not_configured",
            "GOOGLE_TTS_CREDENTIALS_JSON is not a service account key.",
        ) from None


def cloudtts_credentials(raw: str) -> Any:
    """Cached service-account credentials with a usable access token.

    Refresh runs only when the token is missing, expired, or inside
    google-auth's refresh window. The JSON and the token are not logged.
    """
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    secrets = _credential_secrets(raw)
    with _auth_lock:
        credentials = _auth_cache.get(digest)
        if credentials is None:
            credentials = _build_credentials(raw)
            _auth_cache[digest] = credentials
        if _needs_refresh(credentials):
            token = getattr(credentials, "token", None) or ""
            try:
                credentials.refresh(_auth_request())
            except Exception as exc:
                message = _redact(
                    f"Cloud TTS service account token refresh failed: {exc}",
                    *secrets,
                    token if isinstance(token, str) else "",
                )
                raise APIError(400, "not_configured", message) from None
            refreshed = getattr(credentials, "token", None)
            if not isinstance(refreshed, str) or not refreshed:
                raise APIError(
                    400,
                    "not_configured",
                    "Cloud TTS service account did not return an access token.",
                )
        return credentials


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
    provider = payload.get("provider", "gemini")
    if not isinstance(provider, str) or provider not in {"gemini", "cloudtts"}:
        raise APIError(422, "invalid_body", 'provider must be "gemini" or "cloudtts".')
    runs = payload.get("runs_per_sentence", 1)
    samples = payload.get("save_samples", 3)
    style = payload.get("style", DEFAULT_STYLE)
    voice_default = CLOUD_TTS_VOICE if provider == "cloudtts" else DEFAULT_VOICE
    voice = payload.get("voice", voice_default)
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
    streaming = payload.get("streaming", False)
    if not isinstance(streaming, bool):
        raise APIError(422, "invalid_body", "streaming must be a boolean.")
    return {
        "provider": provider,
        "sentences": chosen,
        "runs_per_sentence": runs,
        "save_samples": samples,
        "style": style.strip(),
        "voice": voice.strip(),
        "model": model.strip(),
        "streaming": streaming,
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
        "conversion_ms": 0.0,
        "audio": audio,
        "error": error,
    }


async def synthesize_cloudtts(
    body: dict[str, Any],
    *,
    token: str = "",
    api_key: str = "",
    secrets: list[str] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """POST one unary text:synthesize. The clock is read immediately before send.

    A service-account access token is sent as ``Authorization: Bearer``.
    ``X-Goog-Api-Key`` is sent only when no token was minted. Neither value
    is written into the URL or into logs.
    """
    tick = clock or _now
    hidden = [item for item in (token, api_key, *(secrets or [])) if item]
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif api_key:
        headers["X-Goog-Api-Key"] = api_key
    start = tick()
    conversion_ms = 0.0
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                CLOUD_TTS_URL,
                headers=headers,
                json=body,
            )
    except httpx.HTTPError as exc:
        end = tick()
        return {
            "ttfa_ms": None,
            "total_ms": (end - start) * 1000,
            "conversion_ms": 0.0,
            "audio": b"",
            "error": _redact(str(exc), *hidden),
        }
    if response.status_code >= 400:
        raw = _redact(response.text, *hidden)
        end = tick()
        return {
            "ttfa_ms": None,
            "total_ms": (end - start) * 1000,
            "conversion_ms": 0.0,
            "audio": b"",
            "error": f"HTTP {response.status_code}: {raw}",
        }
    try:
        payload = response.json()
    except json.JSONDecodeError:
        end = tick()
        return {
            "ttfa_ms": None,
            "total_ms": (end - start) * 1000,
            "conversion_ms": 0.0,
            "audio": b"",
            "error": "Cloud TTS response was not JSON.",
        }
    encoded = payload.get("audioContent") if isinstance(payload, dict) else None
    if not isinstance(encoded, str) or not encoded:
        end = tick()
        return {
            "ttfa_ms": None,
            "total_ms": (end - start) * 1000,
            "conversion_ms": 0.0,
            "audio": b"",
            "error": "Cloud TTS response contained no audio.",
        }
    try:
        raw_audio = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        end = tick()
        return {
            "ttfa_ms": None,
            "total_ms": (end - start) * 1000,
            "conversion_ms": 0.0,
            "audio": b"",
            "error": "Cloud TTS audioContent was not base64.",
        }
    first = tick()
    audio, conversion_ms, convert_error = mulaw_from_cloud_audio(raw_audio, tick)
    end = tick()
    if convert_error:
        audio = b""
    elif not audio:
        convert_error = "Cloud TTS response contained no audio."
    return {
        "ttfa_ms": None if convert_error else (first - start) * 1000,
        "total_ms": (end - start) * 1000,
        "conversion_ms": conversion_ms,
        "audio": audio,
        "error": convert_error,
    }


def _audio_chunk(value: Any) -> bytes:
    if not value:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    try:
        converted = bytes(value)
    except (TypeError, ValueError):
        return b""
    return converted


def _text_to_speech_client(credentials: Any) -> Any:
    """One gRPC client per credentials object, reused across probe runs."""
    key = id(credentials)
    with _auth_lock:
        cached = _clients.get(key)
        if cached is not None:
            return cached
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient(credentials=credentials)
    with _auth_lock:
        existing = _clients.get(key)
        if existing is not None:
            return existing
        _clients[key] = client
        return client


def _streaming_requests(text: str, voice: str) -> tuple[Any, Any]:
    """Config frame, then the sentence. Chirp 3 HD, μ-law at 8 kHz, en-AU."""
    from google.cloud import texttospeech

    config = texttospeech.StreamingSynthesizeRequest(
        streaming_config=texttospeech.StreamingSynthesizeConfig(
            voice=texttospeech.VoiceSelectionParams(
                language_code=CLOUD_TTS_LANGUAGE,
                name=voice,
            ),
            streaming_audio_config=texttospeech.StreamingAudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MULAW,
                sample_rate_hertz=SAMPLE_RATE,
            ),
        )
    )
    audio_input = texttospeech.StreamingSynthesizeRequest(
        input=texttospeech.StreamingSynthesisInput(text=text)
    )
    return config, audio_input


def synthesize_cloudtts_streaming(
    credentials: Any,
    text: str,
    voice: str,
    secrets: list[str] | None = None,
) -> dict[str, Any]:
    """One bidirectional StreamingSynthesize call.

    The clock starts immediately before the RPC. Time to first audio is the
    first non-empty ``audio_content`` chunk. Total time ends once that audio
    is 8 kHz μ-law. Conversion is timed on its own.
    """
    hidden = list(secrets or [])
    token = getattr(credentials, "token", None)
    if isinstance(token, str) and token and token not in hidden:
        hidden.append(token)
    try:
        client = _text_to_speech_client(credentials)
        requests = _streaming_requests(text, voice)
    except Exception as exc:
        return {
            "ttfa_ms": None,
            "total_ms": 0.0,
            "conversion_ms": 0.0,
            "audio": b"",
            "error": _redact(f"Cloud TTS stream could not start: {exc}", *hidden),
        }
    start = _now()
    first: float | None = None
    chunks: list[bytes] = []
    try:
        for response in client.streaming_synthesize(iter(requests)):
            chunk = _audio_chunk(getattr(response, "audio_content", b""))
            if not chunk:
                continue
            if first is None:
                first = _now()
            chunks.append(chunk)
    except Exception as exc:
        end = _now()
        return {
            "ttfa_ms": None if first is None else (first - start) * 1000,
            "total_ms": (end - start) * 1000,
            "conversion_ms": 0.0,
            "audio": b"",
            "error": _redact(f"Cloud TTS stream failed: {exc}", *hidden),
        }
    audio, conversion_ms, convert_error = mulaw_from_cloud_audio(b"".join(chunks), _now)
    end = _now()
    if convert_error:
        audio = b""
    elif not audio:
        convert_error = "Cloud TTS stream contained no audio."
    return {
        "ttfa_ms": None if first is None or convert_error else (first - start) * 1000,
        "total_ms": (end - start) * 1000,
        "conversion_ms": conversion_ms,
        "audio": audio,
        "error": convert_error,
    }


async def _collect_runs(
    sentences: list[str],
    runs_per_sentence: int,
    save_samples: int,
    synthesize_one: Callable[[str], Awaitable[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[float], list[float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ttfa_values: list[float] = []
    total_values: list[float] = []
    clips: list[dict[str, Any]] = []
    for sentence in sentences:
        runs: list[dict[str, Any]] = []
        for _ in range(runs_per_sentence):
            result = await synthesize_one(sentence)
            audio: bytes = result["audio"]
            runs.append(
                {
                    "ttfa_ms": result["ttfa_ms"],
                    "total_ms": result["total_ms"],
                    "conversion_ms": result.get("conversion_ms", 0.0),
                    "audio_bytes": len(audio),
                    "audio_seconds": len(audio) / SAMPLE_RATE,
                    "error": result["error"],
                }
            )
            if result["ttfa_ms"] is not None and result["error"] is None:
                ttfa_values.append(result["ttfa_ms"])
                total_values.append(result["total_ms"])
            if audio and len(clips) < save_samples:
                clips.append({"sentence": sentence, "audio": audio})
        rows.append({"sentence": sentence, "runs": runs})
    return rows, ttfa_values, total_values, clips


def _sample_payload(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "n": index,
            "sentence": clip["sentence"],
            "audio_base64": base64.b64encode(clip["audio"]).decode("ascii"),
        }
        for index, clip in enumerate(clips)
    ]


def _cloudtts_auth(settings: Settings, streaming: bool) -> tuple[Any, str, str, list[str]]:
    """Service-account token, or the API-key fallback when streaming is off.

    Returns credentials, bearer token, API key, and values that must be
    redacted. The API key is not returned once a token exists.
    """
    raw = (settings.google_tts_credentials_json or "").strip()
    fallback = (settings.google_tts_api_key or "").strip()
    if raw:
        credentials = cloudtts_credentials(raw)
        token = credentials.token if isinstance(credentials.token, str) else ""
        return credentials, token, "", _credential_secrets(raw, token, fallback)
    if fallback and not streaming:
        return None, "", fallback, [fallback]
    raise APIError(400, "not_configured", "GOOGLE_TTS_CREDENTIALS_JSON not set")


@router.post("/probe/tts")
async def probe_tts(request: Request) -> dict[str, Any]:
    require_bearer(request)
    settings: Settings = request.app.state.settings
    raw = await request.body()
    if not raw.strip():
        payload: Any = {}
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise APIError(422, "invalid_body", "Body must be JSON.") from exc
    options = _parse_probe_body(payload)
    if options["streaming"] and options["provider"] != "cloudtts":
        raise APIError(422, "invalid_body", "streaming is only supported for provider cloudtts.")
    credentials: Any = None
    token = ""
    api_key = ""
    secrets: list[str] = []
    if options["provider"] == "cloudtts":
        credentials, token, api_key, secrets = await asyncio.to_thread(
            _cloudtts_auth,
            settings,
            options["streaming"],
        )
    elif not settings.gemini_api_key:
        raise APIError(503, "not_configured", "GEMINI_API_KEY not set")

    async def synthesize_one(sentence: str) -> dict[str, Any]:
        if options["provider"] == "cloudtts":
            return await synthesize_cloudtts(
                cloudtts_body(sentence, options["voice"]),
                token=token,
                api_key=api_key,
                secrets=secrets,
            )
        return await synthesize(
            settings.gemini_api_key,
            interaction_body(
                sentence,
                model=options["model"],
                style=options["style"],
                voice=options["voice"],
            ),
        )

    rows, ttfa_values, total_values, clips = await _collect_runs(
        options["sentences"],
        options["runs_per_sentence"],
        options["save_samples"],
        synthesize_one,
    )
    request.app.state.tts_probe_clips = clips
    reported = {
        "provider": options["provider"],
        "method": CLOUD_TTS_METHOD if options["provider"] == "cloudtts" else GEMINI_METHOD,
        "voice": options["voice"],
        "mime_type": "audio/mulaw",
        "sample_rate": SAMPLE_RATE,
        "region": server_region(),
        "sentences": rows,
        "ttfa_ms": timing_summary(ttfa_values),
        "total_ms": timing_summary(total_values),
        "samples": _sample_payload(clips),
    }
    if options["provider"] == "cloudtts":
        reported["language_code"] = CLOUD_TTS_LANGUAGE
        if options["streaming"]:
            async def synthesize_stream(sentence: str) -> dict[str, Any]:
                return await asyncio.to_thread(
                    synthesize_cloudtts_streaming,
                    credentials,
                    sentence,
                    options["voice"],
                    secrets,
                )

            stream_rows, stream_ttfa, stream_total, stream_clips = await _collect_runs(
                options["sentences"],
                options["runs_per_sentence"],
                options["save_samples"],
                synthesize_stream,
            )
            reported["streaming"] = {
                "method": CLOUD_TTS_STREAM_METHOD,
                "sentences": stream_rows,
                "ttfa_ms": timing_summary(stream_ttfa),
                "total_ms": timing_summary(stream_total),
                "samples": _sample_payload(stream_clips),
            }
    else:
        reported["model"] = options["model"]
        reported["style"] = options["style"]
    return reported


@router.get("/probe/tts/sample/{n}.wav")
async def probe_sample(n: int, request: Request) -> Response:
    require_bearer(request)
    clips = _clips(request.app)
    if n < 0 or n >= len(clips):
        raise APIError(404, "not_found", "No probe sample at that index. POST /probe/tts first.")
    wav = mulaw_wav(clips[n]["audio"])
    return Response(content=wav, media_type="audio/wav")
