import asyncio
import json

import pytest
from starlette.websockets import WebSocketDisconnect

from app.audio import passthrough_audio
from app.providers import SessionPlan
from app.relay import RelayController, drain_queued_audio, run_relay


class Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def controller(clock: Clock | None = None) -> RelayController:
    plan = SessionPlan(
        setup_messages=[{"type": "session.update", "session": {"marker": True}}],
        provider="grok",
    )
    return RelayController(
        call_id="call-1",
        call_sid="CA1",
        phone="+61400000000",
        opener="Hi, this is an AI assistant.",
        plan=plan,
        hangup_grace_seconds=5,
        clock=clock or Clock(),
    )


def start(relay: RelayController):
    return relay.on_twilio(
        {
            "event": "start",
            "streamSid": "MZ1",
            "start": {"streamSid": "MZ1", "callSid": "CA1", "customParameters": {"name": "Sam"}},
        }
    )


def test_start_does_not_send_the_session_twice():
    relay = controller()
    relay.setup_sent = True
    started = start(relay)
    assert started.provider_messages == []
    assert started.transcripts[0]["text"] == "Hi, this is an AI assistant."


def test_passthrough_forwards_mulaw_unchanged_both_ways():
    relay = controller()
    started = start(relay)
    assert started.provider_messages[0]["type"] == "session.update"
    assert started.transcripts[0]["role"] == "assistant"

    payload = "////qq=="
    inbound = relay.on_twilio(
        {"event": "media", "media": {"track": "inbound", "payload": payload}}
    )
    assert inbound.provider_messages == [
        {"type": "input_audio_buffer.append", "audio": payload}
    ]
    assert inbound.provider_messages[0]["audio"] is passthrough_audio(payload)

    outbound = relay.on_provider({"type": "response.audio.delta", "delta": payload})
    assert outbound.twilio_messages[0]["media"]["payload"] == payload
    assert outbound.twilio_messages[0]["media"]["payload"] is not None
    assert "payload" in outbound.twilio_messages[0]["media"]


def test_opener_is_protected_then_barge_in_clears_twilio():
    clock = Clock()
    relay = controller(clock)
    start(relay)
    relay.on_provider({"type": "response.output_audio.delta", "delta": "AAAA"})
    during = relay.on_provider({"type": "input_audio_buffer.speech_started"})
    assert during.twilio_messages == []
    assert during.drop_queued_audio is False

    done = relay.on_provider({"type": "response.done", "response": {}})
    assert done.twilio_messages[0]["event"] == "mark"
    assert done.twilio_messages[0]["mark"]["name"].startswith("opener-")

    clock.now = 10
    barge = relay.on_provider({"type": "input_audio_buffer.speech_started"})
    assert barge.drop_queued_audio is True
    assert barge.twilio_messages == [{"event": "clear", "streamSid": "MZ1"}]

    dropped = relay.on_provider({"type": "response.output_audio.delta", "delta": "BBBB"})
    assert dropped.twilio_messages == []


def test_end_call_waits_for_playback_mark_then_hangs_up():
    relay = controller()
    start(relay)
    relay.on_provider({"type": "response.done", "response": {}})
    relay.on_provider({"type": "response.created"})
    relay.on_provider({"type": "response.output_audio.delta", "delta": "CCCC"})
    relay.on_provider(
        {
            "type": "response.function_call_arguments.done",
            "name": "end_call",
            "call_id": "fn-1",
            "arguments": json.dumps({"reason": "goodbye"}),
        }
    )
    done = relay.on_provider({"type": "response.done", "response": {}})
    assert done.hangup is False
    assert done.provider_messages[0]["item"]["type"] == "function_call_output"
    mark = done.twilio_messages[0]
    assert mark["event"] == "mark"
    assert relay.hangup_deadline is not None
    echoed = relay.on_twilio(
        {"event": "mark", "mark": {"name": mark["mark"]["name"]}, "streamSid": "MZ1"}
    )
    assert echoed.hangup is True
    assert echoed.end_reason.startswith("end_call:goodbye")


def test_opt_out_is_recorded_and_ends_without_extra_audio():
    relay = controller()
    start(relay)
    relay.on_provider({"type": "response.done", "response": {}})
    relay.on_provider(
        {
            "type": "response.function_call_arguments.done",
            "name": "record_opt_out",
            "call_id": "fn-opt",
            "arguments": {"reason": "do not call"},
        }
    )
    relay.on_provider(
        {
            "type": "response.function_call_arguments.done",
            "name": "end_call",
            "call_id": "fn-end",
            "arguments": {"reason": "opt out"},
        }
    )
    done = relay.on_provider({"type": "response.done", "response": {}})
    assert done.opt_out_reason == "do not call"
    assert done.hangup is True
    assert done.end_reason == "opt_out"
    assert all(message["type"] != "response.create" for message in done.provider_messages)


def test_transcripts_record_both_sides_without_duplicating_the_opener():
    relay = controller()
    start(relay)
    relay.on_provider(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "No thanks, please don't call again.",
            "item_id": "u1",
        }
    )
    relay.on_provider(
        {
            "type": "response.output_audio_transcript.done",
            "transcript": "Hi, this is an AI assistant.",
        }
    )
    roles = [line["role"] for line in relay.transcript]
    assert roles == ["assistant", "caller"]
    assert relay.transcript[1]["text"].startswith("No thanks")


def test_drain_drops_media_and_keeps_marks():
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait({"event": "media", "media": {"payload": "AAAA"}})
    queue.put_nowait({"event": "mark", "mark": {"name": "hangup-1"}})
    queue.put_nowait({"event": "media", "media": {"payload": "BBBB"}})
    dropped = drain_queued_audio(queue)
    assert dropped == 2
    kept = queue.get_nowait()
    assert kept["event"] == "mark"
    assert queue.empty()


class MemoryTwilio:
    def __init__(self) -> None:
        self.inbound: asyncio.Queue = asyncio.Queue()
        self.outbound: asyncio.Queue = asyncio.Queue()

    async def receive_text(self) -> str:
        item = await self.inbound.get()
        if item is None:
            raise WebSocketDisconnect()
        return json.dumps(item)

    async def send_json(self, data: dict) -> None:
        await self.outbound.put(data)

    async def inject(self, message: dict | None) -> None:
        await self.inbound.put(message)


class MemoryProvider:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.inbound: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        return None

    async def send(self, payload: dict) -> None:
        self.sent.append(payload)

    async def recv(self):
        return await self.inbound.get()

    async def push(self, event) -> None:
        await self.inbound.put(event)

    async def close(self) -> None:
        return None


class MemoryStore:
    def __init__(self) -> None:
        self.transcripts: list[dict] = []
        self.opt_outs: list[dict] = []
        self.finished = None
        self.in_progress = False

    async def append_transcript(self, call_id: str, lines: list[dict]) -> None:
        self.transcripts.extend(lines)

    async def record_opt_out(self, **kwargs) -> None:
        self.opt_outs.append(kwargs)

    async def finish_call(self, call_id: str, *, end_reason: str, transcript: list[dict], status: str = "completed") -> None:
        self.finished = {"end_reason": end_reason, "transcript": transcript, "status": status}

    async def mark_in_progress(self, call_id: str) -> None:
        self.in_progress = True


class MemoryGateway:
    def __init__(self) -> None:
        self.hungup: list[str] = []

    async def hangup(self, call_sid: str) -> None:
        self.hungup.append(call_sid)


async def _wait_until(predicate, timeout: float = 1.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met")


@pytest.mark.asyncio
async def test_runner_forwards_audio_and_hangs_up_after_the_mark():
    clock = Clock()
    relay = controller(clock)
    twilio = MemoryTwilio()
    provider = MemoryProvider()
    store = MemoryStore()
    gateway = MemoryGateway()
    task = asyncio.create_task(
        run_relay(
            twilio_ws=twilio,
            provider=provider,
            controller=relay,
            store=store,
            gateway=gateway,
        )
    )
    await twilio.inject({"event": "connected", "protocol": "Call", "version": "1.0.0"})
    await twilio.inject(
        {
            "event": "start",
            "streamSid": "MZ1",
            "start": {"streamSid": "MZ1", "callSid": "CA1", "customParameters": {}},
        }
    )
    await _wait_until(lambda: any(item.get("type") == "session.update" for item in provider.sent))

    payload = "////AAAA"
    await provider.push({"type": "response.output_audio.delta", "delta": payload})
    media = await asyncio.wait_for(twilio.outbound.get(), 1)
    assert media["event"] == "media"
    assert media["media"]["payload"] == payload

    await provider.push({"type": "response.done", "response": {}})
    opener_mark = await asyncio.wait_for(twilio.outbound.get(), 1)
    assert opener_mark["event"] == "mark"
    await twilio.inject(
        {"event": "mark", "streamSid": "MZ1", "mark": {"name": opener_mark["mark"]["name"]}}
    )
    await _wait_until(lambda: relay.protecting_opener is False)

    await provider.push({"type": "response.created"})
    await provider.push({"type": "response.output_audio.delta", "delta": "BBBB"})
    spoken = await asyncio.wait_for(twilio.outbound.get(), 1)
    assert spoken["media"]["payload"] == "BBBB"
    await provider.push({"type": "input_audio_buffer.speech_started"})
    cleared = await asyncio.wait_for(twilio.outbound.get(), 1)
    assert cleared == {"event": "clear", "streamSid": "MZ1"}

    await provider.push({"type": "response.done", "response": {}})
    await provider.push({"type": "response.created"})
    await provider.push({"type": "response.output_audio.delta", "delta": "DDDD"})
    goodbye = await asyncio.wait_for(twilio.outbound.get(), 1)
    assert goodbye["media"]["payload"] == "DDDD"
    await provider.push(
        {
            "type": "response.function_call_arguments.done",
            "name": "end_call",
            "call_id": "fn-9",
            "arguments": '{"reason": "done"}',
        }
    )
    await provider.push({"type": "response.done", "response": {}})
    hang_mark = await asyncio.wait_for(twilio.outbound.get(), 1)
    assert hang_mark["event"] == "mark"
    await twilio.inject(
        {"event": "mark", "streamSid": "MZ1", "mark": {"name": hang_mark["mark"]["name"]}}
    )
    await asyncio.wait_for(task, 2)
    assert gateway.hungup == ["CA1"]
    assert store.finished["end_reason"].startswith("end_call:")
    assert store.in_progress is True
    assert any(line["role"] == "assistant" for line in store.transcripts)


@pytest.mark.asyncio
async def test_session_is_sent_before_stream_start_and_early_audio_is_kept():
    plan = SessionPlan(
        setup_messages=[
            {"type": "session.update", "session": {"marker": True}},
            {"type": "conversation.item.create", "item": {"type": "force_message"}},
        ],
        provider="grok",
    )
    relay = RelayController(
        call_id="call-1",
        call_sid="CA1",
        phone="+61400000000",
        opener="Hi, is this Sam?",
        plan=plan,
        clock=Clock(),
    )
    twilio = MemoryTwilio()
    provider = MemoryProvider()
    store = MemoryStore()
    gateway = MemoryGateway()
    task = asyncio.create_task(
        run_relay(
            twilio_ws=twilio,
            provider=provider,
            controller=relay,
            store=store,
            gateway=gateway,
        )
    )
    try:
        await _wait_until(lambda: [item.get("type") for item in provider.sent] == [
            "session.update",
            "conversation.item.create",
        ])
        assert relay.started is False
        await provider.push({"type": "response.output_audio.delta", "delta": "AAAA"})
        await twilio.inject(
            {
                "event": "start",
                "streamSid": "MZ1",
                "start": {"streamSid": "MZ1", "callSid": "CA1"},
            }
        )
        media = await asyncio.wait_for(twilio.outbound.get(), 1)
        assert media["event"] == "media"
        assert media["media"]["payload"] == "AAAA"
        assert relay.transcript[0]["text"] == "Hi, is this Sam?"
    finally:
        if not task.done():
            await twilio.inject({"event": "stop"})
            await asyncio.wait_for(task, 2)
