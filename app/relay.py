"""Media Streams relay.

Caller audio and assistant audio stay base64 G.711 μ-law. On barge-in the
relay tells Twilio to drop its playback buffer and discards any μ-law frames
still waiting to be sent. ``end_call`` hangs up through the Twilio REST API
after the audio already queued has finished (Twilio mark echo), with a
watchdog if that echo never arrives.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from starlette.websockets import WebSocketDisconnect

from app.audio import passthrough_audio, payload_nbytes
from app.gemini_live import (
    gemini_live_events,
    gemini_live_opener,
    realtime_audio_message,
    session_identity,
)
from app.pcm_bridge import PcmBridge
from app.providers import (
    NormalizedEvent,
    SessionPlan,
    append_audio,
    function_calls_in_response,
    function_output,
    live_append_audio,
    live_function_output,
    normalize_event,
    response_create,
    session_close,
)

logger = logging.getLogger("tack45.relay")

_MULAW_RATE = 8000
# Long enough for "Hi, is this {name}?" and the no-name owner check.
# GPT-Live has no opener response boundary, so protection cannot wait forever.
_GPTLIVE_OPENER_PROTECT_SECONDS = 4.0
# Transcript deltas have no turn-complete event. A gap in the session timeline
# starts a new stored line. Delivery delay does not move start_ms.
_GPTLIVE_TURN_GAP_MS = 700


class CallStore(Protocol):
    async def append_transcript(self, call_id: str, lines: list[dict]) -> None: ...

    async def record_provider_session(
        self,
        call_id: str,
        *,
        voice: str | None,
        session_id: str | None,
    ) -> None: ...

    async def record_opt_out(
        self,
        *,
        call_id: str,
        phone: str,
        call_sid: str | None,
        reason: str,
    ) -> None: ...

    async def finish_call(
        self,
        call_id: str,
        *,
        end_reason: str,
        transcript: list[dict],
        status: str = "completed",
    ) -> None: ...

    async def mark_in_progress(self, call_id: str) -> None: ...


class HangupGateway(Protocol):
    async def hangup(self, call_sid: str) -> None: ...


@dataclass
class StepResult:
    provider_messages: list[dict] = field(default_factory=list)
    twilio_messages: list[dict] = field(default_factory=list)
    drop_queued_audio: bool = False
    transcripts: list[dict] = field(default_factory=list)
    opt_out_reason: str | None = None
    hangup: bool = False
    stop: bool = False
    end_reason: str | None = None
    capture_provider_session: bool = False
    provider_voice: str | None = None
    provider_session_id: str | None = None


def drain_queued_audio(queue: asyncio.Queue) -> int:
    """Remove media frames waiting to be sent. Keep marks and other events."""
    kept: list[Any] = []
    dropped = 0
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if isinstance(item, dict) and item.get("event") == "media":
            dropped += 1
            continue
        kept.append(item)
    for item in kept:
        queue.put_nowait(item)
    return dropped


def _parse_args(raw: str | dict | None) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class RelayController:
    """Pure call state. Network I/O lives in ``run_relay``."""

    def __init__(
        self,
        *,
        call_id: str,
        call_sid: str | None,
        phone: str,
        opener: str,
        plan: SessionPlan,
        hangup_grace_seconds: float = 8.0,
        clock=time.monotonic,
        setup_sent: bool = False,
    ) -> None:
        self.call_id = call_id
        self.call_sid = call_sid
        self.phone = phone
        self.opener = opener
        self.plan = plan
        self.hangup_grace_seconds = hangup_grace_seconds
        self.clock = clock
        self.setup_sent = setup_sent

        self.stream_sid: str | None = None
        self.started = False
        self.protecting_opener = True
        self.opener_protect_until = 0.0
        self.interrupted = False
        self.hangup_requested = False
        self.hung_up = False
        self.end_reason = "completed"
        self.opted_out = False
        self.opt_out_reason = ""
        self.transcript: list[dict] = []
        self.user_partials: dict[str, str] = {}
        self._open_user_id: str | None = None
        self.assistant_parts: list[str] = []
        self._assistant_utterance_id: str | None = None
        self.pending_tools: list[dict] = []
        self.seen_tool_ids: set[str] = set()
        self.response_first_audio_at: float | None = None
        self.audio_bytes = 0
        self.mark_seq = 0
        self.awaiting_mark: str | None = None
        self.mark_purpose: str | None = None
        self.hangup_deadline: float | None = None
        self._partial_seq = 0
        # GPT-Live rejects audio until session.started. Grok and Realtime do not.
        self.live_ready = plan.provider not in {"gptlive", "geminilive"}
        self.held_audio: list[str] = []
        self.playback_until = 0.0
        self.playback_started_at: float | None = None
        self.drop_output_before_ms: int | None = None
        self.session_close_sent = False
        # Gemini Live rejects audio until setupComplete, same gate as GPT-Live.
        self.pcm = PcmBridge()
        self._gemini_in_turn = False
        self._gemini_playback_until = 0.0
        self._tool_names: dict[str, str] = {}
        self._live_turns = {
            "caller": {"id": None, "text": "", "end_ms": None},
            "assistant": {"id": None, "text": "", "end_ms": None},
        }

    def on_twilio(self, message: dict) -> StepResult:
        event = message.get("event")
        if event == "connected":
            return StepResult()
        if event == "start":
            return self._on_start(message)
        if event == "media":
            return self._on_media(message)
        if event == "mark":
            return self._on_mark(message)
        if event == "stop":
            if not self.hangup_requested and self.end_reason == "completed":
                self.end_reason = "caller_hangup"
            return StepResult(
                provider_messages=self._session_close_messages(),
                stop=True,
                end_reason=self.end_reason,
                hangup=False,
            )
        return StepResult()

    def on_provider(self, event: dict) -> StepResult:
        if self.plan.provider == "geminilive":
            return self._on_gemini_live(event)
        norm = normalize_event(event)
        if self.plan.provider == "gptlive" and _gptlive_problem(event):
            logger.warning(
                "GPT-Live event: %s",
                json.dumps(_redact_audio(event), default=str),
            )
        if norm.kind == "session_started":
            return self._on_session_started(event)
        if norm.kind == "session_closed":
            return StepResult()
        if norm.kind == "audio":
            return self._on_audio(norm)
        if norm.kind == "speech_started":
            return self._on_speech_started()
        if norm.kind == "user_transcript":
            return self._on_user_transcript(norm)
        if norm.kind == "assistant_transcript":
            return self._on_assistant_transcript(norm)
        if norm.kind == "function_call":
            self._remember_tool(norm.name, norm.arguments, norm.call_id)
            return StepResult()
        if norm.kind == "response_created":
            return self._on_response_created()
        if norm.kind == "response_done":
            self._collect_function_calls(event)
            return self._on_response_done()
        if norm.kind == "error":
            if self.plan.provider != "gptlive":
                logger.error("Provider error: %s", norm.text)
            return StepResult()
        return StepResult()

    def _on_gemini_live(self, event: dict) -> StepResult:
        parts = [self._on_gemini_live_event(norm) for norm in gemini_live_events(event, self.pcm)]
        return _merge_steps(parts)

    def _on_gemini_live_event(self, norm: NormalizedEvent) -> StepResult:
        if norm.kind == "session_started":
            event = {"session_id": norm.item_id} if norm.item_id else {}
            return self._on_session_started(event)
        if norm.kind == "audio":
            if self.interrupted:
                return self._on_audio(norm)
            return _merge_steps([self._begin_gemini_turn(), self._on_audio(norm)])
        if norm.kind == "speech_started":
            return self._on_speech_started()
        if norm.kind == "user_transcript":
            return self._on_user_transcript(norm)
        if norm.kind == "assistant_transcript":
            if self.interrupted:
                return StepResult()
            return _merge_steps(
                [self._begin_gemini_turn(), self._on_assistant_transcript(norm)]
            )
        if norm.kind == "function_call":
            self._remember_tool(norm.name, norm.arguments, norm.call_id)
            return self._emit_gemini_tools()
        if norm.kind == "tool_cancelled":
            self._cancel_gemini_tool(norm.call_id or "")
            return StepResult()
        if norm.kind == "response_done":
            caller_lines = self._flush_user_partials()
            step = self._on_response_done()
            if caller_lines:
                step.transcripts = caller_lines + step.transcripts
            return step
        if norm.kind == "go_away":
            logger.warning(
                "Gemini Live is closing the socket (time left: %s).",
                norm.text or "unspecified",
            )
            return StepResult()
        if norm.kind == "error":
            logger.error("Gemini Live error: %s", norm.text)
            return StepResult()
        return StepResult()

    def _begin_gemini_turn(self) -> StepResult:
        if self._gemini_in_turn:
            return StepResult()
        self._gemini_in_turn = True
        return self._on_response_created()

    def _emit_gemini_tools(self) -> StepResult:
        tools = self.pending_tools
        self.pending_tools = []
        result = StepResult()
        for tool in tools:
            message, opted = self._execute_tool(tool)
            result.provider_messages.append(message)
            if opted is not None:
                result.opt_out_reason = opted
        if self.hangup_requested:
            self._arm_gemini_hangup(result)
        return result

    def _arm_gemini_hangup(self, result: StepResult) -> None:
        """Hang up after audio already sent. Gemini has no response.create."""
        if not self.hangup_requested or self.hung_up:
            return
        if self.awaiting_mark is not None and self.mark_purpose == "hangup":
            return
        remaining = self._remaining_playback()
        if remaining <= 0:
            remaining = max(0.0, self._gemini_playback_until - self.clock())
        if self.interrupted or remaining <= 0:
            result.provider_messages.extend(self._session_close_messages())
            result.hangup = True
            result.stop = True
            result.end_reason = self.end_reason
            return
        if self.awaiting_mark is not None:
            self.hangup_deadline = self.clock() + remaining + self.hangup_grace_seconds
            return
        self._queue_mark(result, "hangup")
        self.hangup_deadline = self.clock() + remaining + self.hangup_grace_seconds

    def _cancel_gemini_tool(self, call_id: str) -> None:
        """Drop a tool the model retracted.

        record_opt_out is not undone. The person already asked not to be called.
        end_call can still be cancelled before the REST hangup goes out.
        """
        if not call_id:
            return
        was_pending = any(tool["call_id"] == call_id for tool in self.pending_tools)
        self.pending_tools = [tool for tool in self.pending_tools if tool["call_id"] != call_id]
        if was_pending or self._tool_names.get(call_id) != "end_call":
            return
        if self.opted_out or self.hung_up:
            return
        self.hangup_requested = False
        self.hangup_deadline = None
        if self.mark_purpose == "hangup":
            self.awaiting_mark = None
            self.mark_purpose = None
        if self.end_reason.startswith("end_call:"):
            self.end_reason = "completed"

    def _on_start(self, message: dict) -> StepResult:
        start = message.get("start") or {}
        self.stream_sid = start.get("streamSid") or message.get("streamSid")
        self.call_sid = start.get("callSid") or self.call_sid
        self.started = True
        line = self._commit("assistant", self.opener)
        # The session and opener are normally already on the wire. Sending
        # them again would speak the check twice.
        messages = [] if self.setup_sent else list(self.plan.setup_messages)
        self.setup_sent = True
        result = StepResult(provider_messages=messages)
        if line:
            result.transcripts.append(line)
        return result

    def _on_media(self, message: dict) -> StepResult:
        if not self.started or self.hangup_requested:
            return StepResult()
        media = message.get("media") or {}
        track = media.get("track")
        if track not in (None, "", "inbound"):
            return StepResult()
        payload = media.get("payload") or ""
        if not payload:
            return StepResult()
        forwarded = passthrough_audio(payload)
        if self.plan.provider == "geminilive":
            # setupComplete has to land first, and the opener is not interrupted
            # by caller audio. Dropping it here keeps the model from barging in
            # on itself while Twilio playback is still protected.
            if not self.live_ready or self.protecting_opener:
                return StepResult()
            message = realtime_audio_message(self.pcm, forwarded)
            if message is None:
                return StepResult()
            return StepResult(provider_messages=[message])
        if self.plan.provider == "gptlive":
            if not self.live_ready:
                self.held_audio.append(forwarded)
                return StepResult()
            return StepResult(provider_messages=[live_append_audio(forwarded)])
        return StepResult(provider_messages=[append_audio(forwarded)])

    def _on_mark(self, message: dict) -> StepResult:
        name = ((message.get("mark") or {}).get("name")) or ""
        if not name or name != self.awaiting_mark:
            return StepResult()
        purpose = self.mark_purpose
        self.awaiting_mark = None
        self.mark_purpose = None
        if purpose == "opener":
            self.protecting_opener = False
            return StepResult()
        if purpose == "continue":
            return StepResult(provider_messages=[response_create()])
        if purpose == "hangup":
            self.hangup_deadline = None
            return StepResult(
                provider_messages=self._session_close_messages(),
                hangup=True,
                stop=True,
                end_reason=self.end_reason,
            )
        return StepResult()

    def _on_session_started(self, event: dict) -> StepResult:
        self.live_ready = True
        if self.plan.provider == "geminilive":
            # Caller audio buffered before setup is not useful and would
            # interrupt the greeting. The opener goes out on its own.
            self.held_audio.clear()
            if self.plan.setup_messages:
                voice, model = session_identity(self.plan.setup_messages[0])
            else:
                voice, model = "", ""
            server_id = str(event.get("session_id") or "")
            logger.info(
                "Gemini Live session ready model=%s voice=%s session=%s",
                model or "-",
                voice or "-",
                server_id or "-",
            )
            return StepResult(
                provider_messages=[gemini_live_opener(self.opener)],
                capture_provider_session=True,
                provider_voice=voice or None,
                provider_session_id=server_id or model or None,
            )
        messages = [live_append_audio(payload) for payload in self.held_audio]
        self.held_audio.clear()
        result = StepResult(provider_messages=messages)
        if self.plan.provider != "gptlive":
            return result
        session = event.get("session") if isinstance(event.get("session"), dict) else {}
        audio = session.get("audio") if isinstance(session.get("audio"), dict) else {}
        output = audio.get("output") if isinstance(audio.get("output"), dict) else {}
        voice = _voice_name(output.get("voice"))
        session_id = str(session.get("id") or "")
        audio_format = audio.get("format")
        logger.info(
            "GPT-Live session started id=%s model=%s voice=%s format=%s",
            session_id or "-",
            session.get("model") or "-",
            voice or "-",
            json.dumps(audio_format, default=str) if audio_format is not None else "-",
        )
        result.capture_provider_session = True
        result.provider_voice = voice or None
        result.provider_session_id = session_id or None
        return result

    def _on_audio(self, norm: NormalizedEvent) -> StepResult:
        if not norm.audio_b64 or not self.stream_sid:
            return StepResult()
        if (
            self.drop_output_before_ms is not None
            and norm.start_ms is not None
            and norm.start_ms < self.drop_output_before_ms
        ):
            return StepResult()
        if (
            self.drop_output_before_ms is not None
            and norm.start_ms is not None
            and norm.start_ms >= self.drop_output_before_ms
        ):
            self.drop_output_before_ms = None
        if self.interrupted:
            return StepResult()
        payload = passthrough_audio(norm.audio_b64)
        if self.plan.provider == "gptlive":
            now = self.clock()
            if self.playback_started_at is None:
                self.playback_started_at = now
            if now > self.playback_until:
                self.playback_until = now
            self.playback_until += payload_nbytes(payload) / _MULAW_RATE
        now = self.clock()
        if self.response_first_audio_at is None:
            self.response_first_audio_at = now
        self.audio_bytes += payload_nbytes(payload)
        return StepResult(
            twilio_messages=[
                {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": payload},
                }
            ]
        )

    def _on_speech_started(self) -> StepResult:
        # The next caller audio is a new utterance. A late final for the
        # previous one still matches on its own item id.
        self._open_user_id = None
        now = self.clock()
        if self.protecting_opener and (
            self.opener_protect_until == 0.0 or now < self.opener_protect_until
        ):
            return StepResult()
        if self.hangup_requested:
            return StepResult()
        self.protecting_opener = False
        self.interrupted = True
        if not self.stream_sid:
            return StepResult(drop_queued_audio=True)
        return StepResult(
            drop_queued_audio=True,
            twilio_messages=[{"event": "clear", "streamSid": self.stream_sid}],
        )

    def _on_user_transcript(self, norm: NormalizedEvent) -> StepResult:
        if self.plan.provider == "gptlive":
            return self._on_gptlive_transcript("caller", norm)
        item_id = self._caller_utterance_id(norm.item_id)
        if norm.final:
            text = norm.text or self.user_partials.pop(item_id, "")
            self.user_partials.pop(item_id, None)
            if self._open_user_id == item_id:
                self._open_user_id = None
            line = self._commit("caller", text, utterance_id=item_id)
            return StepResult(transcripts=[line] if line else [])
        # Interim fragments stay in the buffer. Only the final text is stored.
        if norm.cumulative:
            self.user_partials[item_id] = norm.text
        else:
            self.user_partials[item_id] = self.user_partials.get(item_id, "") + norm.text
        return StepResult()

    def _on_assistant_transcript(self, norm: NormalizedEvent) -> StepResult:
        if self.plan.provider == "gptlive":
            return self._on_gptlive_transcript("assistant", norm)
        if norm.final:
            self.assistant_parts.clear()
            utterance_id = norm.item_id or self._assistant_utterance_id
            line = self._commit("assistant", norm.text, utterance_id=utterance_id)
            return StepResult(transcripts=[line] if line else [])
        if norm.text:
            self.assistant_parts.append(norm.text)
        return StepResult()

    def _on_response_created(self) -> StepResult:
        lines = self._flush_user_partials() + self._flush_assistant()
        self._assistant_utterance_id = self._next_partial_id()
        self.interrupted = False
        self.response_first_audio_at = None
        self.audio_bytes = 0
        return StepResult(transcripts=lines)

    def _on_response_done(self) -> StepResult:
        lines = self._flush_assistant()
        tools = self.pending_tools
        self.pending_tools = []
        provider_messages: list[dict] = []
        opt_out_reason = None
        for tool in tools:
            message, opted = self._execute_tool(tool)
            provider_messages.append(message)
            if opted is not None:
                opt_out_reason = opted
        result = StepResult(provider_messages=provider_messages, transcripts=lines)
        if opt_out_reason is not None:
            result.opt_out_reason = opt_out_reason

        remaining = self._remaining_playback()
        had_audio = self.audio_bytes > 0
        interrupted = self.interrupted
        if self.protecting_opener:
            if had_audio:
                self.opener_protect_until = self.clock() + remaining
            else:
                self.protecting_opener = False

        if self.plan.provider == "geminilive":
            self._gemini_playback_until = self.clock() + remaining
            self._gemini_in_turn = False
        already_hanging_up = (
            self.plan.provider == "geminilive"
            and self.awaiting_mark is not None
            and self.mark_purpose == "hangup"
        )
        if self.hangup_requested and not already_hanging_up:
            if interrupted or not had_audio:
                # Playback is already finished, so closing the Live session
                # cannot cut off audio still in Twilio's buffer.
                result.provider_messages.extend(self._session_close_messages())
                result.hangup = True
                result.stop = True
                result.end_reason = self.end_reason
            else:
                self._queue_mark(result, "hangup")
                self.hangup_deadline = self.clock() + remaining + self.hangup_grace_seconds
        elif provider_messages and self.plan.provider != "geminilive":
            if interrupted or not had_audio:
                result.provider_messages.append(response_create())
            else:
                self._queue_mark(result, "continue")
        elif self.protecting_opener and had_audio and not self.hangup_requested:
            self._queue_mark(result, "opener")

        self.response_first_audio_at = None
        self.audio_bytes = 0
        self.interrupted = False
        return result

    def _execute_tool(self, tool: dict) -> tuple[dict, str | None]:
        name = tool.get("name")
        args = _parse_args(tool.get("arguments"))
        call_id = tool.get("call_id") or ""
        if name == "record_opt_out":
            reason = str(args.get("reason") or "asked not to be called").strip()
            self.opted_out = True
            self.opt_out_reason = reason
            self.end_reason = "opt_out"
            return (
                self._tool_result(
                    call_id,
                    {
                        "ok": True,
                        "detail": "Opt-out recorded. Confirm if you have not already, then call end_call.",
                    },
                ),
                reason,
            )
        if name == "end_call":
            reason = str(args.get("reason") or "conversation finished").strip()
            self.hangup_requested = True
            if not self.opted_out:
                self.end_reason = f"end_call:{reason}"[:64]
            return (
                self._tool_result(
                    call_id,
                    {
                        "ok": True,
                        "detail": "The call will end after the current audio finishes. Say nothing further.",
                    },
                ),
                None,
            )
        return (
            self._tool_result(call_id, {"ok": False, "error": f"Unknown tool {name}"}),
            None,
        )

    def _tool_result(self, call_id: str, payload: dict) -> dict:
        if self.plan.provider == "gptlive":
            return live_function_output(call_id, payload)
        if self.plan.provider == "geminilive":
            return {
                "toolResponse": {
                    "functionResponses": [
                        {
                            "id": call_id,
                            "name": self._tool_names.get(call_id) or "",
                            "response": payload,
                        }
                    ]
                }
            }
        return function_output(call_id, payload)

    def _session_close_messages(self) -> list[dict]:
        if self.plan.provider != "gptlive" or self.session_close_sent:
            return []
        self.session_close_sent = True
        return [session_close()]

    def _collect_function_calls(self, event: dict) -> None:
        sources = [event]
        inner = event.get("event")
        if isinstance(inner, dict):
            sources.append(inner)
        for source in sources:
            for item in function_calls_in_response(source):
                self._remember_tool(item["name"], item["arguments"], item["call_id"])

    def _on_gptlive_transcript(self, role: str, norm: NormalizedEvent) -> StepResult:
        barge = self._gptlive_barge_in(norm) if role == "caller" else StepResult()
        text = self._append_live_fragment(role, norm)
        if not text:
            return barge
        line = self._commit(role, text, utterance_id=self._live_turns[role]["id"])
        if line:
            barge.transcripts.append(line)
        return barge

    def _append_live_fragment(self, role: str, norm: NormalizedEvent) -> str:
        slot = self._live_turns[role]
        start = norm.start_ms
        new_turn = slot["id"] is None or (
            start is not None
            and slot["end_ms"] is not None
            and start >= slot["end_ms"] + _GPTLIVE_TURN_GAP_MS
        )
        if new_turn:
            if not norm.text:
                return ""
            slot["id"] = self._next_partial_id()
            slot["text"] = ""
            slot["end_ms"] = None
        if norm.text:
            slot["text"] += norm.text
        if norm.end_ms is not None:
            slot["end_ms"] = norm.end_ms
        elif start is not None:
            slot["end_ms"] = start
        return slot["text"]

    def _gptlive_barge_in(self, norm: NormalizedEvent) -> StepResult:
        """Clear Twilio's buffer when the caller talks over playback.

        GPT-Live has no speech-started event and no per-turn response.created.
        ``interrupted`` is left false so later output deltas still play.
        """
        now = self.clock()
        if self._gptlive_opener_blocks(now) or self.hangup_requested:
            return StepResult()
        if now >= self.playback_until:
            return StepResult()
        self.playback_until = now
        self.audio_bytes = 0
        self.response_first_audio_at = None
        if norm.start_ms is not None:
            self.drop_output_before_ms = norm.start_ms
        if not self.stream_sid:
            return StepResult(drop_queued_audio=True)
        return StepResult(
            drop_queued_audio=True,
            twilio_messages=[{"event": "clear", "streamSid": self.stream_sid}],
        )

    def _gptlive_opener_blocks(self, now: float) -> bool:
        if not self.protecting_opener:
            return False
        if self.playback_started_at is None or now < self.playback_until:
            if self.playback_started_at is not None and now >= (
                self.playback_started_at + _GPTLIVE_OPENER_PROTECT_SECONDS
            ):
                self.protecting_opener = False
                return False
            return True
        self.protecting_opener = False
        return False

    def _remember_tool(self, name: str | None, arguments: str | dict | None, call_id: str | None) -> None:
        if not name or not call_id or call_id in self.seen_tool_ids:
            return
        self.seen_tool_ids.add(call_id)
        self._tool_names[call_id] = name
        self.pending_tools.append(
            {"name": name, "arguments": arguments or "{}", "call_id": call_id}
        )

    def _queue_mark(self, result: StepResult, purpose: str) -> None:
        if not self.stream_sid:
            return
        self.mark_seq += 1
        name = f"{purpose}-{self.mark_seq}"
        self.awaiting_mark = name
        self.mark_purpose = purpose
        result.twilio_messages.append(
            {
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": name},
            }
        )

    def _remaining_playback(self) -> float:
        if self.plan.provider == "gptlive":
            return max(0.0, self.playback_until - self.clock())
        if self.response_first_audio_at is None or self.audio_bytes <= 0:
            return 0.0
        ends = self.response_first_audio_at + (self.audio_bytes / _MULAW_RATE)
        return max(0.0, ends - self.clock())

    def _caller_utterance_id(self, item_id: str | None) -> str:
        """One id per caller utterance, even when interim events omit item_id."""
        if item_id:
            self._open_user_id = item_id
            return item_id
        if self._open_user_id is None:
            self._open_user_id = self._next_partial_id()
        return self._open_user_id

    def _flush_user_partials(self) -> list[dict]:
        lines = []
        for item_id, text in self.user_partials.items():
            line = self._commit("caller", text, utterance_id=item_id)
            if line:
                lines.append(line)
        self.user_partials.clear()
        self._open_user_id = None
        return lines

    def _flush_assistant(self) -> list[dict]:
        if not self.assistant_parts:
            return []
        text = "".join(self.assistant_parts)
        self.assistant_parts.clear()
        utterance_id = self._assistant_utterance_id or self._next_partial_id()
        self._assistant_utterance_id = utterance_id
        line = self._commit("assistant", text, utterance_id=utterance_id)
        return [line] if line else []

    def _commit(self, role: str, text: str, utterance_id: str | None = None) -> dict | None:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return None
        opener = " ".join(self.opener.split())
        if role == "assistant" and cleaned == opener and any(
            line["role"] == "assistant" and line["text"] == opener for line in self.transcript
        ):
            return None
        if utterance_id:
            for line in self.transcript:
                if line.get("utterance_id") == utterance_id and line["role"] == role:
                    if line["text"] == cleaned:
                        return None
                    line["text"] = cleaned
                    line["at"] = _iso_now()
                    return line
        if self.transcript:
            last = self.transcript[-1]
            if last["role"] == role and last["text"] == cleaned:
                return None
            # A later final that only extends the previous fragment replaces it.
            if (
                last["role"] == role
                and last.get("utterance_id")
                and cleaned.startswith(last["text"])
            ):
                last["text"] = cleaned
                last["at"] = _iso_now()
                if utterance_id:
                    last["utterance_id"] = utterance_id
                return last
        line = {
            "role": role,
            "text": cleaned,
            "at": _iso_now(),
        }
        if utterance_id:
            line["utterance_id"] = utterance_id
        self.transcript.append(line)
        return line

    def _next_partial_id(self) -> str:
        self._partial_seq += 1
        return f"partial-{self._partial_seq}"


def _merge_steps(parts: list[StepResult]) -> StepResult:
    merged = StepResult()
    for part in parts:
        merged.provider_messages.extend(part.provider_messages)
        merged.twilio_messages.extend(part.twilio_messages)
        merged.transcripts.extend(part.transcripts)
        merged.drop_queued_audio = merged.drop_queued_audio or part.drop_queued_audio
        if part.opt_out_reason is not None:
            merged.opt_out_reason = part.opt_out_reason
        if part.hangup:
            merged.hangup = True
        if part.stop:
            merged.stop = True
        if part.end_reason:
            merged.end_reason = part.end_reason
        if part.capture_provider_session:
            merged.capture_provider_session = True
            merged.provider_voice = part.provider_voice
            merged.provider_session_id = part.provider_session_id
    return merged


def _voice_name(voice: Any) -> str:
    """Built-in Live voices are strings. A custom voice is ``{"id": "..."}``."""
    if isinstance(voice, str):
        return voice
    if isinstance(voice, dict):
        return str(voice.get("id") or "")
    return ""


def _gptlive_problem(event: dict) -> bool:
    kind = str(event.get("type") or "")
    if kind in {"error", "warning"} or "warning" in kind:
        return True
    inner = event.get("event")
    if isinstance(inner, dict):
        inner_kind = str(inner.get("type") or "")
        return inner_kind in {"error", "warning"} or "warning" in inner_kind
    return False


def _redact_audio(value: Any) -> Any:
    """Drop audio bytes from a provider event before it is written to the log."""
    if isinstance(value, dict):
        kind = str(value.get("type") or "")
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key == "audio" or (key == "delta" and "audio" in kind):
                cleaned[key] = "<audio>"
            else:
                cleaned[key] = _redact_audio(item)
        return cleaned
    if isinstance(value, list):
        return [_redact_audio(item) for item in value]
    return value


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


async def run_relay(
    *,
    twilio_ws,
    provider,
    controller: RelayController,
    store: CallStore,
    gateway: HangupGateway | None,
) -> None:
    """Bridge one Twilio media stream to one provider socket until the call ends.

    Session configuration and the opener go out as soon as the provider
    socket is up. They do not wait for Twilio's ``start`` event. Provider
    audio is held until that event, so frames that arrive during the
    handshake are forwarded instead of dropped.
    """
    if not controller.setup_sent:
        await provider.connect()
        for message in controller.plan.setup_messages:
            await provider.send(message)
        controller.setup_sent = True
        logger.info("Sent realtime session for call %s", controller.call_id)
    else:
        logger.info("Using prewarmed realtime session for call %s", controller.call_id)
    provider_out: asyncio.Queue = asyncio.Queue()
    twilio_out: asyncio.Queue = asyncio.Queue()
    stream_ready = asyncio.Event()
    if controller.started:
        stream_ready.set()
    finished = False

    async def finish(end_reason: str) -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        controller.hung_up = True
        for message in controller._session_close_messages():
            try:
                await provider.send(message)
            except Exception:
                logger.exception("Failed to close the GPT-Live session.")
        try:
            await store.finish_call(
                controller.call_id,
                end_reason=end_reason,
                transcript=list(controller.transcript),
                status="completed",
            )
        except Exception:
            logger.exception("Failed to store the final call record.")
        if gateway and controller.call_sid and end_reason != "caller_hangup":
            try:
                await gateway.hangup(controller.call_sid)
            except Exception:
                logger.exception("Twilio hangup failed for %s", controller.call_sid)

    async def dispatch(result: StepResult) -> None:
        if result.drop_queued_audio:
            drain_queued_audio(twilio_out)
        for message in result.provider_messages:
            await provider_out.put(message)
        for message in result.twilio_messages:
            await twilio_out.put(message)
        if result.transcripts:
            try:
                await store.append_transcript(controller.call_id, result.transcripts)
            except Exception:
                logger.exception("Failed to append transcript.")
        if result.capture_provider_session:
            try:
                await store.record_provider_session(
                    controller.call_id,
                    voice=result.provider_voice,
                    session_id=result.provider_session_id,
                )
            except Exception:
                logger.exception("Failed to store the GPT-Live session.")
        if result.opt_out_reason is not None:
            try:
                await store.record_opt_out(
                    call_id=controller.call_id,
                    phone=controller.phone,
                    call_sid=controller.call_sid,
                    reason=result.opt_out_reason,
                )
            except Exception:
                logger.exception("Failed to record opt-out.")
        if result.hangup:
            await finish(result.end_reason or controller.end_reason)

    async def provider_writer() -> None:
        while True:
            message = await provider_out.get()
            if message is None:
                return
            await provider.send(message)

    async def twilio_writer() -> None:
        while True:
            message = await twilio_out.get()
            if message is None:
                return
            await twilio_ws.send_json(message)

    async def twilio_reader() -> None:
        marked = False
        while True:
            raw = await twilio_ws.receive_text()
            result = controller.on_twilio(json.loads(raw))
            if controller.started and not marked:
                marked = True
                try:
                    await store.mark_in_progress(controller.call_id)
                except Exception:
                    logger.exception("Failed to mark the call in progress.")
            await dispatch(result)
            if controller.started:
                stream_ready.set()
            if result.stop:
                if not result.hangup:
                    await finish(result.end_reason or controller.end_reason)
                return

    async def provider_reader() -> None:
        await stream_ready.wait()
        while True:
            event = await provider.recv()
            if event is None:
                if not controller.hung_up:
                    controller.end_reason = controller.end_reason or "provider_closed"
                    await finish(controller.end_reason)
                return
            result = controller.on_provider(event)
            await dispatch(result)
            if result.stop:
                return

    async def watchdog() -> None:
        while True:
            await asyncio.sleep(0.05)
            deadline = controller.hangup_deadline
            if (
                deadline is not None
                and not controller.hung_up
                and controller.clock() >= deadline
            ):
                logger.info("Hangup watchdog fired for call %s", controller.call_id)
                controller.hangup_deadline = None
                await finish(controller.end_reason or "end_call")
                return

    tasks = [
        asyncio.create_task(provider_writer(), name="provider-writer"),
        asyncio.create_task(twilio_writer(), name="twilio-writer"),
        asyncio.create_task(twilio_reader(), name="twilio-reader"),
        asyncio.create_task(provider_reader(), name="provider-reader"),
        asyncio.create_task(watchdog(), name="hangup-watchdog"),
    ]
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if isinstance(exc, WebSocketDisconnect):
                if not finished:
                    reason = controller.end_reason
                    if reason == "completed":
                        reason = "caller_hangup"
                    await finish(reason)
                continue
            if exc:
                logger.exception("Relay task failed", exc_info=exc)
                if not finished:
                    await finish("relay_error")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await provider.close()
        except Exception:
            logger.exception("Failed to close the provider socket.")
