"""Realtime session setup for xAI Grok Voice and OpenAI Realtime.

Both providers are configured for ``audio/pcmu`` (G.711 μ-law, 8 kHz), which
is Twilio Media Streams' native codec. The relay forwards the base64 payload
unchanged. There is no sample-rate field on the pcmu format: 8 kHz is the
codec, and adding a PCM rate would be a different format.

xAI session shape follows https://docs.x.ai/docs/guides/voice/agent
and the force_message opener in that guide. OpenAI uses the GA Realtime
session, where ``g711_ulaw`` is named ``audio/pcmu``.
"""

from __future__ import annotations

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


def append_audio(payload_b64: str) -> dict:
    return {"type": "input_audio_buffer.append", "audio": payload_b64}


def function_output(call_id: str, payload: dict) -> dict:
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(payload),
        },
    }


def response_create() -> dict:
    return {"type": "response.create"}


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
    raise ValueError(f"Unsupported VOICE_PROVIDER: {settings.voice_provider}")


_AUDIO_DELTAS = {"response.output_audio.delta", "response.audio.delta"}
_ASSISTANT_DONE = {
    "response.output_audio_transcript.done",
    "response.audio_transcript.done",
}
_ASSISTANT_PARTIAL = {
    "response.output_audio_transcript.delta",
    "response.audio_transcript.delta",
}


def normalize_event(event: dict) -> NormalizedEvent:
    kind = event.get("type") or ""
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
        detail = event.get("error")
        text = detail if isinstance(detail, str) else str(detail or event)
        return NormalizedEvent("error", text=text)
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
    raise ValueError(f"Unsupported VOICE_PROVIDER: {settings.voice_provider}")
