"""Pure turn logic for the Gemini cascade.

No network calls. The runner in ``app.gemini_cascade`` speaks, listens, and
hangs up from the steps this controller returns. Grok, OpenAI Realtime, and
GPT-Live do not use this module.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlencode

logger = logging.getLogger("tack45.gemini_turn")

MULAW_RATE = 8000
# Short cached clips played the moment a caller turn ends, so the line is not
# silent while the first real sentence is still being synthesised (~480 ms).
BACKCHANNELS = ("Mm-hmm.", "Right.", "Okay.")
POLITE_END = "Sorry, I'm having trouble on my side. I'll leave it there. Goodbye."
OPT_OUT_LINE = "I'll remove this number and won't call again."
# First spoken chunk of a reply is cut here when the model writes one long sentence.
FIRST_SENTENCE_WORDS = 12

_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "eg",
    "ie",
    "am",
    "pm",
}
_DIRECTION = re.compile(
    r"(?i)(australian english accent|in an australian accent|abc-newsreader|stage direction|speech style)"
)
_TOOL_NAME = re.compile(r"(?i)\b(record_opt_out|end_call)\b")
_HTTP_STATUS = re.compile(r"^HTTP (\d{3})\b")


@dataclass
class PlayRequest:
    text: str
    purpose: str
    cached_only: bool = False


@dataclass
class GeminiStep:
    twilio_messages: list[dict] = field(default_factory=list)
    plays: list[PlayRequest] = field(default_factory=list)
    drop_queued_audio: bool = False
    cancel_speech: bool = False
    transcripts: list[dict] = field(default_factory=list)
    opt_out_reason: str | None = None
    hangup: bool = False
    stop: bool = False
    end_reason: str | None = None
    begin_reply: bool = False
    caller_text: str = ""
    caller_audio: str | None = None
    follow_with_end: bool = False


@dataclass
class TurnNotice:
    speech_started: bool = False
    text: str | None = None


@dataclass
class ModelEvent:
    text: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None
    rate_limited: bool = False
    error: str | None = None


def spoken_only(text: str) -> str:
    """Words that may be sent to TTS. Accent and tool names are never spoken."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^(?:\([^)\n]*\)|\[[^\]\n]*\])\s*", "", cleaned)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return ""
    if _DIRECTION.search(cleaned) or _TOOL_NAME.search(cleaned):
        return ""
    return cleaned


def status_from_error(error: str | None) -> int | None:
    if not error:
        return None
    match = _HTTP_STATUS.match(error.strip())
    if match is None:
        return None
    return int(match.group(1))


def is_rate_limit(status: int | None, error: str | None) -> bool:
    if status == 429:
        return True
    if not error:
        return False
    lowered = error.lower()
    return (
        lowered.startswith("http 429")
        or "resource_exhausted" in lowered
        or "resource exhausted" in lowered
    )


def take_sentences(buffer: str, *, flush: bool, early_clause: bool) -> tuple[list[str], str]:
    """Split complete sentences out of a streaming text buffer.

    A boundary is ``.`` ``!`` or ``?`` followed by whitespace, or by the end of
    the buffer when ``flush`` is set. Abbreviations and decimals stay intact.
    ``early_clause`` is for the first chunk of a reply: a comma clause of at
    least three words, or a hard cap of twelve words, so TTS can start before
    a long sentence finishes generating.
    """
    sentences: list[str] = []
    index = 0
    length = len(buffer)
    while index < length:
        end = _sentence_end(buffer, index)
        if end is None:
            break
        after = end + 1
        if after >= length and not flush:
            break
        piece = buffer[index:after].strip()
        if piece:
            sentences.append(piece)
        index = after
        while index < length and buffer[index].isspace():
            index += 1
    if early_clause and not sentences:
        clause = _clause_end(buffer, index)
        if clause is None:
            clause = _word_cap(buffer, index, FIRST_SENTENCE_WORDS)
        if clause is not None:
            piece = buffer[index:clause].strip()
            if piece:
                sentences.append(piece)
            index = clause
            while index < length and buffer[index].isspace():
                index += 1
    if flush:
        tail = buffer[index:].strip()
        if tail:
            sentences.append(tail)
        return sentences, ""
    return sentences, buffer[index:]


def _sentence_end(text: str, start: int) -> int | None:
    index = start
    while index < len(text):
        char = text[index]
        if char in ".!?":
            if char == "." and _skip_dot(text, index):
                index += 1
                continue
            after = index + 1
            if after >= len(text) or text[after].isspace():
                return index
        index += 1
    return None


def _skip_dot(text: str, index: int) -> bool:
    if (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    ):
        return True
    window = text[max(0, index - 3) : index + 1].lower()
    if window.endswith("a.m.") or window.endswith("p.m."):
        return True
    cursor = index - 1
    while cursor >= 0 and text[cursor].isalpha():
        cursor -= 1
    word = text[cursor + 1 : index].lower()
    if word in _ABBREVIATIONS:
        return True
    if word in {"g", "e"} and cursor >= 1 and text[cursor] == ".":
        prev = text[cursor - 1 : cursor].lower()
        if (word == "g" and prev == "e") or (word == "e" and prev == "i"):
            return True
    return False


def _clause_end(text: str, start: int) -> int | None:
    index = start
    while index < len(text):
        if text[index] in ",;":
            after = index + 1
            if after < len(text) and text[after].isspace() and text[after:].strip():
                left = text[start:after].strip()
                if len(left.split()) >= 3:
                    return after
        index += 1
    return None


def _word_cap(text: str, start: int, limit: int) -> int | None:
    words = 0
    index = start
    while index < len(text):
        if text[index].isspace() and index > start and not text[index - 1].isspace():
            words += 1
            if words >= limit and text[index:].strip():
                return index
        index += 1
    return None


def deepgram_listen_url(
    *,
    base: str,
    model: str,
    language: str,
    endpointing_ms: int,
) -> str:
    """Always-on μ-law 8 kHz listen socket with VAD and endpointing."""
    query = urlencode(
        {
            "model": model,
            "language": language,
            "encoding": "mulaw",
            "sample_rate": "8000",
            "channels": "1",
            "interim_results": "true",
            "endpointing": str(int(endpointing_ms)),
            "vad_events": "true",
            "punctuate": "true",
            "smart_format": "true",
            "utterance_end_ms": "1000",
        }
    )
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}{query}"


def deepgram_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Token {api_key}"}


class CallerTurns:
    """Assemble Deepgram results into one caller turn.

    ``SpeechStarted`` is barge-in. ``speech_final`` and ``UtteranceEnd`` close
    the turn. A second end event for the same utterance does not repeat it.
    """

    def __init__(self) -> None:
        self._finals: list[str] = []
        self._interim = ""
        self._emitted = False

    def push(self, message: dict) -> TurnNotice:
        kind = message.get("type") or ""
        if kind == "SpeechStarted":
            return TurnNotice(speech_started=True)
        if kind == "UtteranceEnd":
            return TurnNotice(text=self._take())
        if kind != "Results":
            return TurnNotice()
        transcript, is_final, speech_final = _result_fields(message)
        if self._emitted and (transcript or is_final or speech_final):
            self._emitted = False
            self._finals.clear()
            self._interim = ""
        if is_final and transcript:
            self._finals.append(transcript)
            self._interim = ""
        elif transcript:
            self._interim = transcript
        if speech_final:
            return TurnNotice(text=self._take())
        return TurnNotice()

    def _take(self) -> str | None:
        if self._emitted:
            return None
        text = " ".join(part.strip() for part in self._finals if part and part.strip()).strip()
        if not text:
            text = self._interim.strip()
        self._finals.clear()
        self._interim = ""
        if not text:
            return None
        self._emitted = True
        return text


def _result_fields(message: dict) -> tuple[str, bool, bool]:
    channel = message.get("channel")
    alternatives: list = []
    if isinstance(channel, dict):
        alternatives = channel.get("alternatives") or []
    elif isinstance(channel, list) and channel:
        first = channel[0]
        if isinstance(first, dict):
            alternatives = first.get("alternatives") or []
    transcript = ""
    if alternatives and isinstance(alternatives[0], dict):
        transcript = str(alternatives[0].get("transcript") or "")
    return transcript.strip(), bool(message.get("is_final")), bool(message.get("speech_final"))


def events_from_gemini_payload(payload: object) -> list[ModelEvent]:
    """Text and tool calls from one ``streamGenerateContent`` JSON object."""
    if isinstance(payload, list):
        events: list[ModelEvent] = []
        for item in payload:
            events.extend(events_from_gemini_payload(item))
        return events
    if not isinstance(payload, dict):
        return []
    error = payload.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or error)
        status = error.get("code")
        if status == 429 or is_rate_limit(status if isinstance(status, int) else None, message):
            return [ModelEvent(rate_limited=True, error=message)]
        return [ModelEvent(error=message)]
    candidates = payload.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return []
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    events = []
    for part in parts:
        if not isinstance(part, dict) or part.get("thought"):
            continue
        function_call = part.get("functionCall") or part.get("function_call")
        if isinstance(function_call, dict) and function_call.get("name"):
            args = function_call.get("args")
            if args is None:
                args = function_call.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            events.append(ModelEvent(tool_name=str(function_call["name"]), tool_args=args))
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            events.append(ModelEvent(text=text))
    return events


def gemini_function_declarations() -> list[dict]:
    from app.providers import tool_definitions

    declarations = []
    for tool in tool_definitions():
        declarations.append(
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
        )
    return declarations


def conversation_contents(transcript: list[dict]) -> list[dict]:
    """Gemini contents, user then model, from the stored transcript."""
    contents: list[dict] = []
    for line in transcript:
        text = " ".join(str(line.get("text") or "").split())
        if not text:
            continue
        role = "model" if line.get("role") == "assistant" else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += " " + text
            continue
        contents.append({"role": role, "parts": [{"text": text}]})
    if contents and contents[0]["role"] != "user":
        contents.insert(0, {"role": "user", "parts": [{"text": "(The call has connected.)"}]})
    if not contents:
        contents.append({"role": "user", "parts": [{"text": "(The call has connected.)"}]})
    return contents


def text_request(instructions: str, transcript: list[dict], settings) -> dict:
    """One streaming generateContent body. The TTS style string is not included."""
    body: dict = {
        "systemInstruction": {"parts": [{"text": instructions}]},
        "contents": conversation_contents(transcript),
        "tools": [{"functionDeclarations": gemini_function_declarations()}],
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 256,
        },
    }
    budget = settings.gemini_thinking_budget
    if isinstance(budget, int) and not isinstance(budget, bool) and budget >= 0:
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": budget}
    return body


def clip_key(text: str, settings) -> str:
    return "\n".join(
        (
            settings.gemini_tts_model,
            settings.gemini_tts_voice,
            settings.gemini_tts_style,
            settings.gemini_tts_language,
            text,
        )
    )


def prerender_lines(opener: str) -> list[str]:
    """Fixed lines rendered once and reused. The opener is first."""
    ordered = [opener, POLITE_END, OPT_OUT_LINE, *BACKCHANNELS]
    seen: set[str] = set()
    lines: list[str] = []
    for line in ordered:
        cleaned = " ".join((line or "").split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        lines.append(cleaned)
    return lines


class GeminiController:
    """Barge-in, opener protection, sentence playback, and immediate opt-out."""

    def __init__(
        self,
        *,
        call_id: str,
        call_sid: str | None,
        phone: str,
        opener: str,
        instructions: str,
        hangup_grace_seconds: float = 8.0,
        clock=time.monotonic,
        utterance_prefix: str = "gemini",
    ) -> None:
        self.call_id = call_id
        self.call_sid = call_sid
        self.phone = phone
        self.opener = opener
        self.instructions = instructions
        self.hangup_grace_seconds = hangup_grace_seconds
        self.clock = clock
        self.utterance_prefix = utterance_prefix or "gemini"

        self.stream_sid: str | None = None
        self.started = False
        self.protecting_opener = True
        self.interrupted = False
        self.hangup_requested = False
        self.hung_up = False
        self.end_reason = "completed"
        self.opted_out = False
        self.opt_out_reason = ""
        self.rate_limited = False
        self._fell_back = False
        self.transcript: list[dict] = []
        self.pending_caller: str | None = None
        self.generation = 0
        self.reply_active = False
        self._spoke_this_reply = False
        self.audio_bytes = 0
        self.response_first_audio_at: float | None = None
        self.mark_seq = 0
        self.awaiting_mark: str | None = None
        self.mark_purpose: str | None = None
        self.hangup_deadline: float | None = None
        self._backchannel_i = 0
        self._partial_seq = 0

    def on_twilio(self, message: dict) -> GeminiStep:
        event = message.get("event")
        if event == "connected":
            return GeminiStep()
        if event == "start":
            return self._on_start(message)
        if event == "media":
            return self._on_media(message)
        if event == "mark":
            return self._on_mark(message)
        if event == "stop":
            if not self.hangup_requested and self.end_reason == "completed":
                self.end_reason = "caller_hangup"
            return GeminiStep(stop=True, end_reason=self.end_reason)
        return GeminiStep()

    def on_caller_speech(self) -> GeminiStep:
        """Caller started talking. Clear Twilio and stop TTS, except during the opener."""
        if not self.started or self.protecting_opener or self.hangup_requested or self._fell_back:
            return GeminiStep()
        if not self._assistant_audible() and not self.reply_active:
            return GeminiStep()
        self.interrupted = True
        self.reply_active = False
        self.generation += 1
        self.audio_bytes = 0
        self.response_first_audio_at = None
        step = GeminiStep(cancel_speech=True, drop_queued_audio=True)
        if self.stream_sid:
            step.twilio_messages.append({"event": "clear", "streamSid": self.stream_sid})
        return step

    def on_caller_turn(self, text: str) -> GeminiStep:
        if not self.started:
            return GeminiStep()
        cleaned = " ".join((text or "").split())
        step = GeminiStep()
        line = self._commit("caller", cleaned)
        if line:
            step.transcripts.append(line)
        if not cleaned or self.hangup_requested or self._fell_back:
            return step
        if self.protecting_opener:
            if self.pending_caller:
                self.pending_caller = f"{self.pending_caller} {cleaned}".strip()
            else:
                self.pending_caller = cleaned
            return step
        if self.reply_active or self._assistant_audible():
            self.interrupted = True
            self.reply_active = False
            self.generation += 1
            self.audio_bytes = 0
            self.response_first_audio_at = None
            step.cancel_speech = True
            step.drop_queued_audio = True
            if self.stream_sid:
                step.twilio_messages.append({"event": "clear", "streamSid": self.stream_sid})
        return self._start_reply(step, cleaned)

    def on_assistant_sentence(self, text: str, generation: int) -> GeminiStep:
        if generation != self.generation or self.interrupted or self._fell_back:
            return GeminiStep()
        spoken = spoken_only(text)
        if (text or "").strip() and not spoken:
            logger.info("Dropped model text that was a direction or a tool name.")
            return GeminiStep()
        if not spoken:
            return GeminiStep()
        self._spoke_this_reply = True
        line = self._commit("assistant", spoken)
        step = GeminiStep(plays=[PlayRequest(spoken, "speech")])
        if line:
            step.transcripts.append(line)
        return step

    def on_tool(self, name: str | None, args: dict | None, generation: int) -> GeminiStep:
        if generation != self.generation or self._fell_back:
            return GeminiStep()
        arguments = args or {}
        if name == "record_opt_out":
            reason = str(arguments.get("reason") or "asked not to be called").strip()
            already = self.opted_out
            self.opted_out = True
            self.opt_out_reason = reason
            self.end_reason = "opt_out"
            self.hangup_requested = True
            step = GeminiStep()
            if not already:
                # The do-not-call write happens on this step, before end_call and
                # before the confirmation finishes playing.
                step.opt_out_reason = reason
            return step
        if name == "end_call":
            reason = str(arguments.get("reason") or "conversation finished").strip()
            self.hangup_requested = True
            if not self.opted_out:
                self.end_reason = f"end_call:{reason}"[:64]
            return GeminiStep()
        logger.info("Ignoring unknown tool %s", name)
        return GeminiStep()

    def on_rate_limited(self, source: str) -> GeminiStep:
        logger.warning(
            "Gemini 429 from %s on call %s. Tier 1 allows 10 requests/minute and "
            "100/day, which cannot carry a live cascaded call. Falling back.",
            source,
            self.call_id,
        )
        return self._begin_fallback("rate_limited")

    def on_model_error(self, message: str) -> GeminiStep:
        logger.error("Gemini cascade error on call %s: %s", self.call_id, message)
        return self._begin_fallback("model_error")

    def on_reply_finished(self, generation: int) -> GeminiStep:
        if generation != self.generation:
            return GeminiStep()
        self.reply_active = False
        step = GeminiStep()
        if self.opted_out and not self._spoke_this_reply and not self._fell_back:
            self._spoke_this_reply = True
            line = self._commit("assistant", OPT_OUT_LINE)
            step.plays.append(PlayRequest(OPT_OUT_LINE, "speech"))
            if line:
                step.transcripts.append(line)
            step.follow_with_end = True
            return step
        if self.hangup_requested:
            if self.audio_bytes <= 0:
                step.hangup = True
                step.stop = True
                step.end_reason = self.end_reason
            else:
                mark = self.queue_mark("hangup")
                step.twilio_messages.extend(mark.twilio_messages)
                step.hangup = mark.hangup
                step.stop = mark.stop
                step.end_reason = mark.end_reason
        return step

    def queue_mark(self, purpose: str) -> GeminiStep:
        if not self.stream_sid:
            if purpose == "opener":
                self.protecting_opener = False
                return self._release_pending(GeminiStep())
            if purpose == "hangup":
                return GeminiStep(hangup=True, stop=True, end_reason=self.end_reason)
            return GeminiStep()
        self.mark_seq += 1
        name = f"{purpose}-{self.mark_seq}"
        self.awaiting_mark = name
        self.mark_purpose = purpose
        if purpose == "hangup":
            self.hangup_deadline = self.clock() + self._remaining() + self.hangup_grace_seconds
        return GeminiStep(
            twilio_messages=[
                {
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": name},
                }
            ]
        )

    def note_playback(self, nbytes: int, generation: int) -> bool:
        if generation != self.generation or self.interrupted or nbytes <= 0:
            return False
        if self.response_first_audio_at is None:
            self.response_first_audio_at = self.clock()
        self.audio_bytes += nbytes
        return True

    def _on_start(self, message: dict) -> GeminiStep:
        start = message.get("start") or {}
        self.stream_sid = start.get("streamSid") or message.get("streamSid")
        self.call_sid = start.get("callSid") or self.call_sid
        self.started = True
        line = self._commit("assistant", self.opener)
        step = GeminiStep(plays=[PlayRequest(self.opener, "opener")])
        if line:
            step.transcripts.append(line)
        return step

    def _on_media(self, message: dict) -> GeminiStep:
        if not self.started or self.hangup_requested:
            return GeminiStep()
        media = message.get("media") or {}
        track = media.get("track")
        if track not in (None, "", "inbound"):
            return GeminiStep()
        payload = media.get("payload") or ""
        if not payload:
            return GeminiStep()
        return GeminiStep(caller_audio=payload)

    def _on_mark(self, message: dict) -> GeminiStep:
        name = ((message.get("mark") or {}).get("name")) or ""
        if not name or name != self.awaiting_mark:
            return GeminiStep()
        purpose = self.mark_purpose
        self.awaiting_mark = None
        self.mark_purpose = None
        if purpose == "opener":
            self.protecting_opener = False
            self.audio_bytes = 0
            self.response_first_audio_at = None
            return self._release_pending(GeminiStep())
        if purpose == "hangup":
            self.hangup_deadline = None
            return GeminiStep(hangup=True, stop=True, end_reason=self.end_reason)
        return GeminiStep()

    def _start_reply(self, step: GeminiStep, text: str) -> GeminiStep:
        self.interrupted = False
        self.reply_active = True
        self._spoke_this_reply = False
        self.audio_bytes = 0
        self.response_first_audio_at = None
        step.begin_reply = True
        step.caller_text = text
        if not self.opted_out:
            step.plays.append(
                PlayRequest(self._next_backchannel(), "backchannel", cached_only=True)
            )
        return step

    def _release_pending(self, step: GeminiStep) -> GeminiStep:
        text = self.pending_caller
        self.pending_caller = None
        if not text or self.hangup_requested or self._fell_back:
            return step
        return self._start_reply(step, text)

    def _begin_fallback(self, reason: str) -> GeminiStep:
        if self._fell_back:
            self.hangup_requested = True
            if self.end_reason == "completed":
                self.end_reason = reason
            return GeminiStep(
                hangup=True,
                stop=True,
                end_reason=self.end_reason,
                cancel_speech=True,
            )
        self._fell_back = True
        if reason == "rate_limited":
            self.rate_limited = True
        self.hangup_requested = True
        self.end_reason = reason
        self.protecting_opener = False
        self.reply_active = False
        self.interrupted = False
        self.pending_caller = None
        self.generation += 1
        self.audio_bytes = 0
        self.response_first_audio_at = None
        step = GeminiStep(
            cancel_speech=True,
            drop_queued_audio=True,
            plays=[PlayRequest(POLITE_END, "fallback")],
            end_reason=reason,
        )
        if self.stream_sid:
            step.twilio_messages.append({"event": "clear", "streamSid": self.stream_sid})
        return step

    def _next_backchannel(self) -> str:
        clip = BACKCHANNELS[self._backchannel_i % len(BACKCHANNELS)]
        self._backchannel_i += 1
        return clip

    def _assistant_audible(self) -> bool:
        return self._remaining() > 0

    def _remaining(self) -> float:
        if self.response_first_audio_at is None or self.audio_bytes <= 0:
            return 0.0
        ends = self.response_first_audio_at + (self.audio_bytes / MULAW_RATE)
        return max(0.0, ends - self.clock())

    def _commit(self, role: str, text: str) -> dict | None:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return None
        opener = " ".join(self.opener.split())
        if role == "assistant" and cleaned == opener and any(
            line["role"] == "assistant" and line["text"] == opener for line in self.transcript
        ):
            return None
        if self.transcript:
            last = self.transcript[-1]
            if last["role"] == role and last["text"] == cleaned:
                return None
        self._partial_seq += 1
        line = {
            "role": role,
            "text": cleaned,
            "at": datetime.now(timezone.utc).isoformat(),
            "utterance_id": f"{self.utterance_prefix}-{self._partial_seq}",
        }
        self.transcript.append(line)
        return line
