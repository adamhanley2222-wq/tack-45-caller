"""Local simulation of the Gemini cascade.

Scripted caller audio, a fake text model, and fake μ-law. Nothing here dials
Twilio or calls Deepgram or Gemini. ``scripts/simulate_gemini.py`` prints the
same run.
"""

from __future__ import annotations

import asyncio
import base64
import time

from app.config import Settings
from app.gemini_cascade import ClipCache, SpeechClip, run_gemini_cascade
from app.gemini_turn import (
    BACKCHANNELS,
    POLITE_END,
    GeminiController,
    ModelEvent,
    clip_key,
)
from app.providers import SessionPlan, build_session_plan


class MemoryTwilio:
    def __init__(self) -> None:
        self.inbound: asyncio.Queue = asyncio.Queue()
        self.outbound: asyncio.Queue = asyncio.Queue()

    async def receive_text(self) -> str:
        import json

        item = await self.inbound.get()
        if item is None:
            from starlette.websockets import WebSocketDisconnect

            raise WebSocketDisconnect()
        return json.dumps(item)

    async def send_json(self, data: dict) -> None:
        await self.outbound.put(data)

    async def inject(self, message: dict | None) -> None:
        await self.inbound.put(message)


class MemoryStore:
    def __init__(self, order: list[str] | None = None) -> None:
        self.order = order if order is not None else []
        self.transcripts: list[dict] = []
        self.opt_outs: list[dict] = []
        self.finished = None
        self.in_progress = False
        self.voice = None

    async def append_transcript(self, call_id: str, lines: list[dict]) -> None:
        self.transcripts.extend(lines)

    async def record_opt_out(self, **kwargs) -> None:
        self.order.append("opt_out")
        self.opt_outs.append(kwargs)

    async def finish_call(self, call_id: str, *, end_reason: str, transcript: list[dict], status: str = "completed") -> None:
        self.order.append(f"finish:{end_reason}")
        self.finished = {"end_reason": end_reason, "transcript": transcript, "status": status}

    async def mark_in_progress(self, call_id: str) -> None:
        self.in_progress = True

    async def record_provider_session(self, call_id: str, *, voice: str | None, session_id: str | None) -> None:
        self.voice = voice


class MemoryGateway:
    def __init__(self) -> None:
        self.hungup: list[str] = []

    async def hangup(self, call_sid: str) -> None:
        self.hungup.append(call_sid)


class FakeListener:
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.events: asyncio.Queue = asyncio.Queue()
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def send_audio(self, payload: bytes) -> None:
        self.audio.append(payload)

    async def recv(self):
        return await self.events.get()

    async def close(self) -> None:
        self.closed = True

    async def push(self, event: dict | None) -> None:
        await self.events.put(event)


class ScriptModel:
    def __init__(self, events: list[ModelEvent], gate: asyncio.Event | None = None) -> None:
        self._events = events
        self.gate = gate
        self.calls = 0

    async def stream_reply(self, *, instructions: str, transcript: list[dict]):
        self.calls += 1
        yielded = 0
        for event in self._events:
            if self.gate is not None and yielded == 1:
                await self.gate.wait()
            yielded += 1
            yield event


class TimingSynth:
    def __init__(self, order: list[str], delay: float = 0.05, rate_limit: bool = False) -> None:
        self.order = order
        self.delay = delay
        self.rate_limit = rate_limit
        self.events: list[tuple] = []

    async def synthesize(self, text: str) -> SpeechClip:
        self.order.append(f"synth:{text}")
        self.events.append(("start", text, time.monotonic()))
        await asyncio.sleep(self.delay)
        self.events.append(("end", text, time.monotonic()))
        if self.rate_limit:
            return SpeechClip(audio=b"", error="HTTP 429: resource exhausted", status=429, rate_limited=True)
        token = text.encode("utf-8")[:24]
        return SpeechClip(audio=token + (b"\xff" * 180), error=None, status=200, rate_limited=False)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        voice_provider="gemini",
        gemini_api_key="gem-test",
        deepgram_api_key="dg-test",
        hangup_grace_seconds=1.0,
    )


def _plan(settings: Settings) -> SessionPlan:
    return build_session_plan(settings, "be brief", "Hi, is this Sam?")


def _cache(settings: Settings) -> ClipCache:
    cache = ClipCache()
    cache.remember(clip_key("Hi, is this Sam?", settings), b"\x01" * 160)
    cache.remember(clip_key(POLITE_END, settings), b"\x03" * 160)
    for line in BACKCHANNELS:
        cache.remember(clip_key(line, settings), b"\x02" * 160)
    return cache


def _controller(settings: Settings, plan: SessionPlan) -> GeminiController:
    message = plan.setup_messages[0]
    return GeminiController(
        call_id="call-1",
        call_sid="CA1",
        phone="+61400000000",
        opener=message["opener"],
        instructions=message["instructions"],
        hangup_grace_seconds=settings.hangup_grace_seconds,
    )


async def _echo(twilio: MemoryTwilio, sent: list[dict]) -> None:
    while True:
        message = await twilio.outbound.get()
        sent.append(message)
        if message.get("event") == "mark":
            await twilio.inject(
                {"event": "mark", "streamSid": "MZ1", "mark": message["mark"]}
            )


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for the cascade")


def _speech_final(text: str) -> dict:
    return {
        "type": "Results",
        "is_final": True,
        "speech_final": True,
        "channel": {"alternatives": [{"transcript": text}]},
    }


def _payloads(sent: list[dict]) -> list[bytes]:
    found = []
    for message in sent:
        if message.get("event") == "media":
            found.append(base64.b64decode(message["media"]["payload"]))
    return found


async def _drive(settings, plan, listener, model, synth, store, *, speak: str) -> tuple[asyncio.Task, list[dict], MemoryTwilio, MemoryGateway]:
    twilio = MemoryTwilio()
    gateway = MemoryGateway()
    sent: list[dict] = []
    cascade = asyncio.create_task(
        run_gemini_cascade(
            twilio_ws=twilio,
            settings=settings,
            controller=_controller(settings, plan),
            store=store,
            gateway=gateway,
            cache=_cache(settings),
            listener=listener,
            text_model=model,
            synth=synth,
        )
    )
    echo = asyncio.create_task(_echo(twilio, sent))
    cascade.add_done_callback(lambda _task: echo.cancel())
    await twilio.inject({"event": "connected"})
    await twilio.inject(
        {
            "event": "start",
            "streamSid": "MZ1",
            "start": {"streamSid": "MZ1", "callSid": "CA1"},
        }
    )
    await _wait_until(lambda: any(message.get("event") == "media" for message in sent))
    await listener.push(_speech_final(speak))
    return cascade, sent, twilio, gateway


async def _overlap(settings, plan) -> str:
    order: list[str] = []
    synth = TimingSynth(order, delay=0.05)
    model = ScriptModel([ModelEvent(text="Thanks Sam. How are you?")])
    listener = FakeListener()
    store = MemoryStore(order)
    cascade, _sent, twilio, _gateway = await _drive(
        settings, plan, listener, model, synth, store, speak="Yes this is Sam."
    )
    await _wait_until(
        lambda: any(kind == "end" and text == "How are you?" for kind, text, _at in synth.events)
    )
    starts = {text: at for kind, text, at in synth.events if kind == "start"}
    ends = {text: at for kind, text, at in synth.events if kind == "end"}
    assert starts["How are you?"] < ends["Thanks Sam."]
    assert "synth:Thanks Sam." in order
    assert order.index("synth:Thanks Sam.") < order.index("synth:How are you?")
    await twilio.inject({"event": "stop"})
    await asyncio.wait_for(cascade, 2)
    return "overlap: second sentence started synthesis while the first was still rendering"


async def _barge(settings, plan) -> str:
    order: list[str] = []
    gate = asyncio.Event()
    synth = TimingSynth(order, delay=0.01)
    model = ScriptModel(
        [ModelEvent(text="Thanks Sam. "), ModelEvent(text="I can explain the chat. ")],
        gate=gate,
    )
    listener = FakeListener()
    store = MemoryStore(order)
    cascade, sent, twilio, _gateway = await _drive(
        settings, plan, listener, model, synth, store, speak="Yes."
    )
    await _wait_until(lambda: any(item[1] == "Thanks Sam." and item[0] == "end" for item in synth.events))
    await listener.push({"type": "SpeechStarted"})
    await _wait_until(lambda: any(message.get("event") == "clear" for message in sent))
    gate.set()
    await asyncio.sleep(0.05)
    spoken = b"".join(_payloads(sent))
    assert b"Thanks Sam." in spoken
    assert b"I can explain" not in spoken
    await twilio.inject({"event": "stop"})
    await asyncio.wait_for(cascade, 2)
    return "barge-in: Twilio clear sent and the rest of the reply was dropped"


async def _opt_out(settings, plan) -> str:
    order: list[str] = []
    synth = TimingSynth(order, delay=0.02)
    model = ScriptModel(
        [
            ModelEvent(tool_name="record_opt_out", tool_args={"reason": "stop calling"}),
            ModelEvent(tool_name="end_call", tool_args={"reason": "opt out"}),
            ModelEvent(text="I'll remove this number and won't call again."),
        ]
    )
    listener = FakeListener()
    store = MemoryStore(order)
    cascade, _sent, _twilio, gateway = await _drive(
        settings, plan, listener, model, synth, store, speak="Please stop calling me."
    )
    await asyncio.wait_for(cascade, 2)
    assert store.opt_outs[0]["reason"] == "stop calling"
    assert order.index("opt_out") < order.index("synth:I'll remove this number and won't call again.")
    assert store.finished["end_reason"] == "opt_out"
    assert gateway.hungup == ["CA1"]
    roles = [line["role"] for line in store.finished["transcript"]]
    assert "caller" in roles and "assistant" in roles
    return "opt-out: recorded before the confirmation audio, then the call hung up"


async def _limited(settings, plan) -> str:
    order: list[str] = []
    synth = TimingSynth(order, delay=0.01, rate_limit=True)
    model = ScriptModel([ModelEvent(text="Thanks Sam. Is now a good time?")])
    listener = FakeListener()
    store = MemoryStore(order)
    cascade, sent, _twilio, gateway = await _drive(
        settings, plan, listener, model, synth, store, speak="Yes."
    )
    await asyncio.wait_for(cascade, 2)
    assert store.finished["end_reason"] == "rate_limited"
    assert gateway.hungup == ["CA1"]
    spoken = b"".join(_payloads(sent))
    assert b"\x03" * 20 in spoken
    assert b"Thanks Sam." not in spoken
    return "rate-limit: 429 logged via fallback, cached goodbye played, call ended"


async def run_scenarios() -> dict:
    settings = _settings()
    plan = _plan(settings)
    lines = [
        await _overlap(settings, plan),
        await _barge(settings, plan),
        await _opt_out(settings, plan),
        await _limited(settings, plan),
    ]
    return {"ok": True, "lines": lines}
