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
from app.providers import (
    NormalizedEvent,
    SessionPlan,
    append_audio,
    function_calls_in_response,
    function_output,
    normalize_event,
    response_create,
)

logger = logging.getLogger("tack45.relay")

_MULAW_RATE = 8000


class CallStore(Protocol):
    async def append_transcript(self, call_id: str, lines: list[dict]) -> None: ...

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
            return StepResult(stop=True, end_reason=self.end_reason, hangup=False)
        return StepResult()

    def on_provider(self, event: dict) -> StepResult:
        norm = normalize_event(event)
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
            for item in function_calls_in_response(event):
                self._remember_tool(item["name"], item["arguments"], item["call_id"])
            return self._on_response_done()
        if norm.kind == "error":
            logger.error("Provider error: %s", norm.text)
            return StepResult()
        return StepResult()

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
            return StepResult(hangup=True, stop=True, end_reason=self.end_reason)
        return StepResult()

    def _on_audio(self, norm: NormalizedEvent) -> StepResult:
        if not norm.audio_b64 or not self.stream_sid:
            return StepResult()
        if self.interrupted:
            return StepResult()
        payload = passthrough_audio(norm.audio_b64)
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

        if self.hangup_requested:
            if interrupted or not had_audio:
                result.hangup = True
                result.stop = True
                result.end_reason = self.end_reason
            else:
                self._queue_mark(result, "hangup")
                self.hangup_deadline = self.clock() + remaining + self.hangup_grace_seconds
        elif provider_messages:
            if interrupted or not had_audio:
                result.provider_messages.append(response_create())
            else:
                self._queue_mark(result, "continue")
        elif self.protecting_opener and had_audio:
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
                function_output(
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
                function_output(
                    call_id,
                    {
                        "ok": True,
                        "detail": "The call will end after the current audio finishes. Say nothing further.",
                    },
                ),
                None,
            )
        return (
            function_output(call_id, {"ok": False, "error": f"Unknown tool {name}"}),
            None,
        )

    def _remember_tool(self, name: str | None, arguments: str | dict | None, call_id: str | None) -> None:
        if not name or not call_id or call_id in self.seen_tool_ids:
            return
        self.seen_tool_ids.add(call_id)
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
