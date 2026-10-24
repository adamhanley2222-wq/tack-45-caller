"""Gemini Live session messages and server-event parsing.

Voice-to-voice over one WebSocket, same role as Grok and GPT-Live. The wire
format is ``BidiGenerateContent``: setup first, then 16 kHz PCM in and 24 kHz
PCM out. This module does not open a socket. ``app.relay`` sends the messages
and turns the parsed events into Twilio media, transcripts, and tool calls.

Model id ``gemini-3.8-live`` is the stable id on
https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live
(AI Studio label "Gemini 3.8 Live"). The socket is the v1beta Live endpoint in
https://ai.google.dev/api/live
"""

from __future__ import annotations

import base64
from urllib.parse import urlencode

from app.config import Settings
from app.pcm_bridge import PcmBridge
from app.providers import NormalizedEvent, tool_definitions

GEMINI_LIVE_MODEL = "gemini-3.8-live"
GEMINI_LIVE_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

_SPEECH_POLICY = (
    "\n\nSPEECH\n"
    "Speak with this accent and manner: {style}. Language is {language}.\n"
    "That accent is how you sound. Never say the accent, the language code, "
    "or a description of your voice, and never put accent or stage directions "
    "into the words you speak.\n"
    "Call record_opt_out and end_call as tools. Never speak a tool name.\n"
    "When asked to say a greeting verbatim, say only those words and then wait.\n"
)


def gemini_live_ws_url(base: str, api_key: str) -> str:
    """Live authenticates with the API key on the WebSocket URL."""
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}{urlencode({'key': api_key})}"


def gemini_live_setup(settings: Settings, instructions: str) -> dict:
    """First WebSocket message. Nothing after this may be sent before setupComplete."""
    model = _model_resource(settings.geminilive_model)
    voice = settings.geminilive_voice.strip() or "Charon"
    language = settings.geminilive_language.strip() or "en-AU"
    style = settings.geminilive_style.strip() or "Australian English accent, relaxed and friendly"
    return {
        "setup": {
            "model": model,
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "languageCode": language,
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}},
                },
            },
            "systemInstruction": {
                "parts": [
                    {
                        "text": instructions
                        + _SPEECH_POLICY.format(style=style, language=language)
                    }
                ]
            },
            "tools": [{"functionDeclarations": _function_declarations()}],
            "inputAudioTranscription": {"languageCodes": [language]},
            "outputAudioTranscription": {"languageCodes": [language]},
            "realtimeInputConfig": {"activityHandling": "START_OF_ACTIVITY_INTERRUPTS"},
            "contextWindowCompression": {"slidingWindow": {}},
        }
    }


def session_identity(setup_message: dict) -> tuple[str, str]:
    """Voice name and model resource from the setup message, for the call record."""
    setup = setup_message.get("setup") if isinstance(setup_message, dict) else None
    if not isinstance(setup, dict):
        return "", ""
    model = str(setup.get("model") or "")
    generation = setup.get("generationConfig") if isinstance(setup.get("generationConfig"), dict) else {}
    speech = generation.get("speechConfig") if isinstance(generation.get("speechConfig"), dict) else {}
    voice_config = speech.get("voiceConfig") if isinstance(speech.get("voiceConfig"), dict) else {}
    prebuilt = (
        voice_config.get("prebuiltVoiceConfig")
        if isinstance(voice_config.get("prebuiltVoiceConfig"), dict)
        else {}
    )
    return str(prebuilt.get("voiceName") or ""), model


def gemini_live_opener(opener: str) -> dict:
    """Ask the model to speak the compliance line. Sent only after setupComplete.

    The accent stays in the system instruction and speechConfig. It is not
    part of this text.
    """
    text = (
        "The call has just connected. Say exactly the following words and nothing else, "
        "then wait and listen. Do not add anything before or after:\n\n"
        + opener
    )
    return {
        "clientContent": {
            "turns": [{"role": "user", "parts": [{"text": text}]}],
            "turnComplete": True,
        }
    }


def realtime_audio_message(bridge: PcmBridge, mulaw_b64: str) -> dict | None:
    """Twilio μ-law payload → one Live ``realtimeInput`` audio message."""
    try:
        mulaw = base64.b64decode(mulaw_b64, validate=False)
    except Exception:
        return None
    pcm = bridge.mulaw8k_to_pcm16k(mulaw)
    if not pcm:
        return None
    return {
        "realtimeInput": {
            "audio": {
                "mimeType": "audio/pcm;rate=16000",
                "data": base64.b64encode(pcm).decode("ascii"),
            }
        }
    }


def gemini_live_events(message: dict, bridge: PcmBridge) -> list[NormalizedEvent]:
    """One server message → ordered relay events.

    An interruption is first, so audio in the same message is not played.
    Transcription fragments are deltas. The relay commits them at turn
    boundaries. Tool calls are returned here and executed by the relay; this
    function does not run them.
    """
    if not isinstance(message, dict):
        return []
    if "setupComplete" in message or "setup_complete" in message:
        return [NormalizedEvent("session_started")]
    tool_call = message.get("toolCall") or message.get("tool_call")
    if isinstance(tool_call, dict):
        return _tool_events(tool_call)
    cancelled = message.get("toolCallCancellation") or message.get("tool_call_cancellation")
    if isinstance(cancelled, dict):
        return _cancel_events(cancelled)
    if "goAway" in message or "go_away" in message:
        notice = message.get("goAway") or message.get("go_away") or {}
        left = notice.get("timeLeft") or notice.get("time_left") if isinstance(notice, dict) else None
        return [NormalizedEvent("go_away", text=str(left or ""))]
    content = message.get("serverContent") or message.get("server_content")
    if isinstance(content, dict):
        return _content_events(content, bridge)
    if "error" in message:
        return [NormalizedEvent("error", text=_error_text(message))]
    return []


def _model_resource(model: str) -> str:
    cleaned = (model or GEMINI_LIVE_MODEL).strip() or GEMINI_LIVE_MODEL
    if cleaned.startswith("models/"):
        return cleaned
    return f"models/{cleaned}"


def _function_declarations() -> list[dict]:
    declared = []
    for tool in tool_definitions():
        declared.append(
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
                # 3.8 Live defaults to async calls. Blocking matches the
                # existing "speak, then end" tool flow and waits for our result.
                "behavior": "BLOCKING",
            }
        )
    return declared


def _tool_events(tool_call: dict) -> list[NormalizedEvent]:
    calls = tool_call.get("functionCalls") or tool_call.get("function_calls") or []
    events: list[NormalizedEvent] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        call_id = call.get("id") or call.get("call_id")
        args = call.get("args")
        if args is None:
            args = call.get("arguments") or {}
        events.append(
            NormalizedEvent(
                "function_call",
                name=str(name) if name else None,
                arguments=args if isinstance(args, (dict, str)) else {},
                call_id=str(call_id) if call_id else None,
            )
        )
    return events


def _cancel_events(cancelled: dict) -> list[NormalizedEvent]:
    ids = cancelled.get("ids") or []
    return [
        NormalizedEvent("tool_cancelled", call_id=str(call_id))
        for call_id in ids
        if call_id
    ]


def _content_events(content: dict, bridge: PcmBridge) -> list[NormalizedEvent]:
    events: list[NormalizedEvent] = []
    if content.get("interrupted"):
        events.append(NormalizedEvent("speech_started"))
    incoming = content.get("inputTranscription") or content.get("input_transcription")
    if isinstance(incoming, dict) and incoming.get("text"):
        events.append(
            NormalizedEvent(
                "user_transcript",
                text=str(incoming.get("text") or ""),
                final=False,
                cumulative=False,
            )
        )
    outgoing = content.get("outputTranscription") or content.get("output_transcription")
    if isinstance(outgoing, dict) and outgoing.get("text"):
        events.append(
            NormalizedEvent(
                "assistant_transcript",
                text=str(outgoing.get("text") or ""),
                final=False,
            )
        )
    model_turn = content.get("modelTurn") or content.get("model_turn") or {}
    parts = model_turn.get("parts") if isinstance(model_turn, dict) else None
    for part in parts or []:
        if not isinstance(part, dict) or part.get("thought"):
            continue
        inline = part.get("inlineData") or part.get("inline_data")
        if not isinstance(inline, dict):
            continue
        mime = str(inline.get("mimeType") or inline.get("mime_type") or "")
        if "audio" not in mime:
            continue
        data = inline.get("data")
        if not isinstance(data, str) or not data:
            continue
        try:
            pcm = base64.b64decode(data, validate=False)
        except Exception:
            continue
        mulaw = bridge.pcm24k_to_mulaw8k(pcm, rate=_pcm_rate(mime))
        if not mulaw:
            continue
        events.append(
            NormalizedEvent("audio", audio_b64=base64.b64encode(mulaw).decode("ascii"))
        )
    if content.get("turnComplete") or content.get("turn_complete"):
        events.append(NormalizedEvent("response_done"))
    return events


def _pcm_rate(mime: str) -> int:
    for piece in mime.split(";"):
        if "rate=" not in piece:
            continue
        raw = piece.split("rate=", 1)[1].strip()
        try:
            rate = int(raw)
        except ValueError:
            return 24000
        return rate if rate > 0 else 24000
    return 24000


def _error_text(message: dict) -> str:
    detail = message.get("error")
    if isinstance(detail, dict):
        text = detail.get("message") or detail.get("status") or detail
        return str(text)
    return str(detail or message)
