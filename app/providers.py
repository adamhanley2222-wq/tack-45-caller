"""Realtime session setup for xAI Grok Voice, OpenAI Realtime, GPT-Live, and Gemini Live.

``VOICE_PROVIDER=gemini`` does not open one of these sockets. It returns a
local cascade plan (instructions plus the opener). The media route runs that
plan through the Deepgram / Gemini pipeline instead of ``run_relay``.

``VOICE_PROVIDER=geminilive`` does open a socket: the Gemini Live API. Twilio
μ-law is resampled to 16 kHz PCM on the way in and from 24 kHz PCM on the way
out. Grok, OpenAI, and GPT-Live still forward μ-law unchanged.

Grok and OpenAI Realtime are configured for ``audio/pcmu`` (G.711 μ-law,
8 kHz) with no sample-rate field: 8 kHz is the codec, and adding a PCM rate
would be a different format. The relay forwards the base64 payload unchanged.

xAI session shape follows https://docs.x.ai/docs/guides/voice/agent
and the force_message opener in that guide. OpenAI uses the GA Realtime
session, where ``g711_ulaw`` is named ``audio/pcmu``.

GPT-Live (``VOICE_PROVIDER=gptlive``) uses ``session.start`` on
``wss://api.openai.com/v1/live/sessions``. Its pcmu format includes
``rate: 8000``, which is the Live API's G.711 μ-law format, not a PCM rate.
Tools are registered on Responses delegation. The Live model does not receive
a top-level ``tools`` field; startup rejects unknown session fields.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from urllib.parse import urlencode

from app.config import Settings


@dataclass(frozen=True)
class NormalizedEvent:
    kind: str
    audio_b64: str | None = None
    text: str = ""
    item_id: str | None = None
    final: bool = False
    cumulative: bool = False
    name: str | None = None
    arguments: str | dict | None = None
    call_id: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass(frozen=True)
class SessionPlan:
    setup_messages: list[dict]
    provider: str


def tool_definitions() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "end_call",
            "description": (
                "End the phone call after you have finished speaking. "
                "Call this once the conversation is over or the person wants to hang up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "A few words on why the call is ending.",
                    }
                },
                "required": ["reason"],
            },
        },
        {
            "type": "function",
            "name": "record_opt_out",
            "description": (
                "Record that this person asked not to be called again. "
                "Confirm that aloud first, then call this, then call end_call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "What the person said.",
                    }
                },
                "required": ["reason"],
            },
        },
    ]


def force_message(text: str) -> dict:
    """xAI extension: speak this text verbatim. Do not also send response.create."""
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "force_message",
            "role": "assistant",
            "interruptible": False,
            "content": [{"type": "output_text", "text": text}],
        },
    }


def opener_response(text: str) -> dict:
    """Fallback opener for OpenAI, which has no force_message."""
    return {
        "type": "response.create",
        "response": {
            "instructions": (
                "Say exactly the following words and nothing else, then wait. "
                "Do not add anything before or after:\n\n"
                + text
            )
        },
    }


def is_spoken_opener(message: dict) -> bool:
    """A setup message that makes the model speak, rather than session config."""
    if not isinstance(message, dict):
        return False
    if message.get("type") == "response.create":
        return True
    item = message.get("item")
    return isinstance(item, dict) and item.get("type") == "force_message"


def gptlive_greeting_text(opener: str) -> str:
    return (
        "Greet the caller now in English. Say exactly these words "
        "and nothing else, then pause and listen:\n\n"
        + opener
    )


def gptlive_opener_commentary(opener: str) -> dict:
    """Speak the opener after session.started. Not part of session.start.

    Sending the greeting inside session.start makes GPT-Live talk as soon as
    the socket is up. commentary.append is the documented follow-up that asks
    it to speak once the session is already running.
    """
    return {
        "type": "session.commentary.append",
        "event_id": "opener_1",
        "delegation_id": None,
        "content": gptlive_greeting_text(opener),
    }


def without_gptlive_greeting(message: dict) -> dict:
    """session.start with the speak-now developer input removed. Other messages pass through."""
    if not isinstance(message, dict) or message.get("type") != "session.start":
        return message
    session = message.get("session")
    if not isinstance(session, dict) or "input" not in session:
        return message
    cloned = copy.deepcopy(message)
    cloned["session"].pop("input", None)
    return cloned


def append_audio(payload_b64: str) -> dict:
    return {"type": "input_audio_buffer.append", "audio": payload_b64}


def live_append_audio(payload_b64: str) -> dict:
    return {"type": "session.input_audio.append", "audio": payload_b64}


def function_output(call_id: str, payload: dict) -> dict:
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(payload),
        },
    }


def live_function_output(call_id: str, payload: dict) -> dict:
    """Tool result for GPT-Live Responses delegation. Grok and Realtime use function_output."""
    return {
        "type": "response.item.create",
        "event_id": f"tool_result_{call_id}",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(payload),
        },
    }


def response_create() -> dict:
    return {"type": "response.create"}


def session_close() -> dict:
    return {"type": "session.close", "event_id": "event_close"}


def realtime_ws_url(base: str, model: str) -> str:
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}{urlencode({'model': model})}"


def _grok_session(settings: Settings, instructions: str) -> dict:
    return {
        "type": "session.update",
        "session": {
            "voice": settings.xai_voice,
            "instructions": instructions,
            "reasoning": {"effort": settings.xai_reasoning_effort},
            "turn_detection": {"type": "server_vad"},
            "tools": tool_definitions(),
            "replace": {"Tack45": "Tack forty five"},
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "transport": "json",
                    "transcription": {
                        "model": "grok-transcribe",
                        "language_hint": "en",
                    },
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "transport": "json",
                },
            },
        },
    }


def _openai_session(settings: Settings, instructions: str) -> dict:
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": instructions,
            "output_modalities": ["audio"],
            "tool_choice": "auto",
            "tools": tool_definitions(),
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "transcription": {"model": settings.openai_transcribe_model},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": settings.openai_voice,
                },
            },
        },
    }


_LIVE_TOOL_POLICY = (
    "\n\nDelegation policy:\n"
    "You speak with the caller. The responses backend calls tools; you do not invent tool results.\n"
    "Delegate record_opt_out when they ask not to be called again, after you have confirmed that aloud.\n"
    "Delegate end_call after your goodbye, when the conversation is finished or they want to hang up.\n"
    "Stop speaking when the caller interrupts, then listen.\n"
)

_BACKEND_TOOL_POLICY = (
    "\n\nYou are the tool backend for this phone call. "
    "Call record_opt_out when the caller asks not to be called again. "
    "Call end_call when the conversation is over or the caller wants to hang up. "
    "Do not call any other tool.\n"
)


def _gptlive_session(settings: Settings, instructions: str, opener: str) -> dict:
    """One session.start. The opener is a developer input so it is in the first message.

    Responses delegation is what carries end_call and record_opt_out. Client
    delegation's session.delegation.created event has no tool name or arguments.
    """
    return {
        "type": "session.start",
        "event_id": "event_start",
        "session": {
            "model": settings.gptlive_model,
            "instructions": instructions + _LIVE_TOOL_POLICY,
            "audio": {
                "format": {"type": "audio/pcmu", "rate": 8000},
                # Built-in voices, including ripple, are the API name as a string.
                # A custom voice would be {"id": "voice_..."}. The default if this
                # is omitted is marin.
                "output": {"voice": settings.gptlive_voice},
            },
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": settings.gptlive_responses_model,
                    "instructions": instructions + _BACKEND_TOOL_POLICY,
                    "tools": tool_definitions(),
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                },
            },
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": gptlive_greeting_text(opener),
                        }
                    ],
                }
            ],
        },
    }


def build_session_plan(settings: Settings, instructions: str, opener: str) -> SessionPlan:
    provider = settings.voice_provider.strip().lower()
    if provider == "grok":
        messages = [_grok_session(settings, instructions)]
        if settings.use_force_message:
            messages.append(force_message(opener))
        else:
            messages.append(opener_response(opener))
        return SessionPlan(setup_messages=messages, provider=provider)
    if provider == "openai":
        messages = [_openai_session(settings, instructions), opener_response(opener)]
        return SessionPlan(setup_messages=messages, provider=provider)
    if provider == "gptlive":
        return SessionPlan(
            setup_messages=[_gptlive_session(settings, instructions, opener)],
            provider=provider,
        )
    if provider == "gemini":
        return SessionPlan(
            setup_messages=[_gemini_cascade(settings, instructions, opener)],
            provider=provider,
        )
    if provider == "geminilive":
        from app.gemini_live import gemini_live_setup

        return SessionPlan(
            setup_messages=[gemini_live_setup(settings, instructions)],
            provider=provider,
        )
    raise ValueError(f"Unsupported VOICE_PROVIDER: {settings.voice_provider}")


_GEMINI_TURN_POLICY = (
    "\n\nSPEECH OUTPUT\n"
    "A separate text-to-speech engine reads your words aloud. "
    "Write only the words the caller should hear.\n"
    "The first sentence of every reply is one short clause, twelve words or fewer, "
    "and it ends with a full stop. Later sentences may follow.\n"
    "Do not put accent, voice, style, or stage directions in the words you write. "
    "Do not say \"Australian English accent\" or describe how you sound.\n"
    "Call record_opt_out and end_call as tools. Never speak a tool name.\n"
)


def _gemini_cascade(settings: Settings, instructions: str, opener: str) -> dict:
    """Local plan. Nothing in here is sent to a realtime websocket."""
    return {
        "type": "gemini.cascade",
        "instructions": instructions + _GEMINI_TURN_POLICY,
        "opener": opener,
        "tools": ["end_call", "record_opt_out"],
        "tts": {
            "model": settings.gemini_tts_model,
            "voice": settings.gemini_tts_voice,
            "style": settings.gemini_tts_style,
            "language": settings.gemini_tts_language,
        },
        "stt": {
            "provider": "deepgram",
            "model": settings.deepgram_model,
            "language": settings.deepgram_language,
        },
    }


_AUDIO_DELTAS = {"response.output_audio.delta", "response.audio.delta"}
_ASSISTANT_DONE = {
    "response.output_audio_transcript.done",
    "response.audio_transcript.done",
}
_ASSISTANT_PARTIAL = {
    "response.output_audio_transcript.delta",
    "response.audio_transcript.delta",
}


def _timeline_ms(event: dict, field: str) -> int | None:
    value = event.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _error_message(event: dict) -> str:
    detail = event.get("error")
    if isinstance(detail, dict):
        message = detail.get("message")
        if message:
            return str(message)
        return str(detail)
    if isinstance(detail, str) and detail:
        return detail
    return str(detail or event)


def _normalize_delegated_response(event: dict) -> NormalizedEvent:
    """Events nested in a GPT-Live ``response.event`` envelope.

    ``response.created`` is ignored. GPT-Live does not use it as a spoken-turn
    boundary, and treating it as one would drop later audio.
    """
    kind = event.get("type") or ""
    if kind == "response.output_item.done":
        item = event.get("item") or {}
        if item.get("type") == "function_call":
            return NormalizedEvent(
                "function_call",
                name=item.get("name"),
                arguments=item.get("arguments") if item.get("arguments") is not None else "{}",
                call_id=item.get("call_id"),
            )
        return NormalizedEvent("ignore", text=kind)
    if kind in {"response.done", "response.completed"}:
        return NormalizedEvent("response_done")
    if kind == "error":
        return NormalizedEvent("error", text=_error_message(event))
    return NormalizedEvent("ignore", text=kind)


def normalize_event(event: dict) -> NormalizedEvent:
    kind = event.get("type") or ""
    if kind == "response.event":
        inner = event.get("event")
        if isinstance(inner, dict):
            return _normalize_delegated_response(inner)
        return NormalizedEvent("ignore", text=kind)
    if kind == "session.output_audio.delta":
        return NormalizedEvent(
            "audio",
            audio_b64=event.get("delta") or event.get("audio"),
            start_ms=_timeline_ms(event, "start_ms"),
        )
    if kind == "session.input_transcript.delta":
        return NormalizedEvent(
            "user_transcript",
            text=event.get("delta") or "",
            final=False,
            cumulative=False,
            start_ms=_timeline_ms(event, "start_ms"),
            end_ms=_timeline_ms(event, "end_ms"),
        )
    if kind == "session.output_transcript.delta":
        return NormalizedEvent(
            "assistant_transcript",
            text=event.get("delta") or "",
            final=False,
            cumulative=False,
            start_ms=_timeline_ms(event, "start_ms"),
            end_ms=_timeline_ms(event, "end_ms"),
        )
    if kind == "session.started":
        return NormalizedEvent("session_started")
    if kind == "session.closed":
        return NormalizedEvent("session_closed")
    if kind in _AUDIO_DELTAS:
        return NormalizedEvent("audio", audio_b64=event.get("delta") or event.get("audio"))
    if kind == "input_audio_buffer.speech_started":
        return NormalizedEvent("speech_started")
    if kind == "conversation.item.input_audio_transcription.completed":
        return NormalizedEvent(
            "user_transcript",
            text=event.get("transcript") or "",
            item_id=event.get("item_id"),
            final=True,
        )
    if kind == "conversation.item.input_audio_transcription.updated":
        return NormalizedEvent(
            "user_transcript",
            text=event.get("transcript") or "",
            item_id=event.get("item_id"),
            final=False,
            cumulative=True,
        )
    if kind == "conversation.item.input_audio_transcription.delta":
        return NormalizedEvent(
            "user_transcript",
            text=event.get("delta") or "",
            item_id=event.get("item_id"),
            final=False,
            cumulative=False,
        )
    if kind in _ASSISTANT_DONE:
        return NormalizedEvent(
            "assistant_transcript",
            text=event.get("transcript") or "",
            item_id=event.get("item_id"),
            final=True,
        )
    if kind in _ASSISTANT_PARTIAL:
        return NormalizedEvent(
            "assistant_transcript",
            text=event.get("delta") or "",
            item_id=event.get("item_id"),
            final=False,
        )
    if kind == "response.function_call_arguments.done":
        return NormalizedEvent(
            "function_call",
            name=event.get("name"),
            arguments=event.get("arguments") or "{}",
            call_id=event.get("call_id"),
        )
    if kind == "response.created":
        return NormalizedEvent("response_created")
    if kind == "response.done":
        return NormalizedEvent("response_done")
    if kind == "error":
        return NormalizedEvent("error", text=_error_message(event))
    return NormalizedEvent("ignore", text=kind)


def function_calls_in_response(event: dict) -> list[dict]:
    """Pull function calls embedded on ``response.done`` if a dedicated event was missed."""
    response = event.get("response") or {}
    found: list[dict] = []
    for item in response.get("output") or []:
        if item.get("type") != "function_call":
            continue
        found.append(
            {
                "name": item.get("name"),
                "arguments": item.get("arguments") or "{}",
                "call_id": item.get("call_id"),
            }
        )
    return found


def provider_connection_target(settings: Settings) -> tuple[str, dict[str, str]]:
    provider = settings.voice_provider.strip().lower()
    if provider == "grok":
        url = realtime_ws_url(settings.xai_realtime_url, settings.xai_realtime_model)
        return url, {"Authorization": f"Bearer {settings.xai_api_key}"}
    if provider == "openai":
        url = realtime_ws_url(settings.openai_realtime_url, settings.openai_realtime_model)
        return url, {"Authorization": f"Bearer {settings.openai_api_key}"}
    if provider == "gptlive":
        # Live rejects a model query string. The model is session.start's session.model.
        return settings.gptlive_url, {"Authorization": f"Bearer {settings.openai_api_key}"}
    if provider == "gemini":
        raise ValueError(
            "VOICE_PROVIDER=gemini uses the cascaded pipeline, not a realtime websocket."
        )
    if provider == "geminilive":
        from app.gemini_live import gemini_live_ws_url

        return gemini_live_ws_url(settings.geminilive_url, settings.gemini_api_key), {}
    raise ValueError(f"Unsupported VOICE_PROVIDER: {settings.voice_provider}")
