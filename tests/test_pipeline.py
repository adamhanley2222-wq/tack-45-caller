"""VOICE_PROVIDER=pipeline. Google clients are fakes. Nothing is dialled."""

from __future__ import annotations

import base64
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import APIError, prepare_script
from app.config import missing_dial_configuration
from app.gemini_turn import GeminiController, ModelEvent
from app.main import create_app
from app.pipeline import (
    PipelineController,
    PipelineTextModel,
    PipelineTurns,
    SttNotice,
    credential_json,
    hold_opener,
    notices_from_stt,
    project_id_from_credentials,
    recognize_requests,
    run_pipeline,
    speech_api_endpoint,
    stt_recognizer,
    stt_streaming_config,
    tts_config_request,
)
from app.providers import build_session_plan, provider_connection_target
from app.tts_probe import cloudtts_credentials
from tests.conftest import make_settings
from tests.test_api import _allowed_time, _auth

SPEECH = bytes([0xC8, 0x48, 0xB8, 0x38]) * 200
SILENCE = b"\xff" * 160
CLICK = bytes([0xC8, 0x48, 0xB8, 0x38]) * 40

START = {
    "event": "start",
    "streamSid": "MZ1",
    "start": {"streamSid": "MZ1", "callSid": "CA1"},
}


class Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _controller(clock: Clock, *, hold: bool) -> PipelineController:
    return PipelineController(
        call_id="c1",
        call_sid="CA1",
        phone="+61400000000",
        opener="Hi, is this Sam?",
        instructions="Be brief. I'm an AI assistant calling for Tack45.",
        clock=clock,
        hold_opener=hold,
        silence_ms=500,
        max_wait_ms=2500,
    )


def _media(payload: bytes, at: float, clock: Clock) -> dict:
    clock.now = at
    return {
        "event": "media",
        "media": {"track": "inbound", "payload": base64.b64encode(payload).decode("ascii")},
    }


def _pipeline_settings(tmp_path, **overrides):
    values = dict(
        voice_provider="pipeline",
        gemini_api_key="gem-test",
        google_tts_credentials_json='{"project_id":"tack45-test","client_email":"sa@test.iam"}',
    )
    values.update(overrides)
    return make_settings(tmp_path, **values)


def test_pipeline_plan_keeps_the_prompt_disclosure_and_tools(tmp_path):
    settings = _pipeline_settings(tmp_path, pipeline_llm_model="gemini-2.5-flash")
    opener, plan = prepare_script(
        settings,
        {
            "name": "Sam Lee",
            "business": "Harbour Roasters",
            "notes": "Asked about invoices",
            "to_number": "+61400000000",
            "brief": "Mention the workshop.",
        },
    )
    assert opener == "Hi, is this Sam?"
    message = plan.setup_messages[0]
    assert plan.provider == "pipeline"
    assert message["type"] == "voice.pipeline"
    assert message["opener"] == opener
    assert "I'm an AI" in message["instructions"]
    assert "Sam Lee" in message["instructions"]
    assert "Harbour Roasters" in message["instructions"]
    assert "Asked about invoices" in message["instructions"]
    assert "Mention the workshop." in message["instructions"]
    assert "record_opt_out" in message["instructions"]
    assert "end_call" in message["instructions"]
    assert "OFF-TOPIC" in message["instructions"]
    assert "twelve words" in message["instructions"]
    assert message["tools"] == ["end_call", "record_opt_out"]
    assert message["llm"] == {"model": "gemini-2.5-flash"}
    assert message["tts"]["voice"] == "en-AU-Chirp3-HD-Charon"
    assert message["tts"]["audio_encoding"] == "MULAW"
    assert message["tts"]["sample_rate_hertz"] == 8000
    assert message["stt"] == {
        "provider": "google",
        "model": "telephony",
        "language": "en-AU",
        "location": "global",
        "encoding": "MULAW",
        "sample_rate_hertz": 8000,
    }
    live = build_session_plan(
        make_settings(tmp_path, voice_provider="geminilive", gemini_api_key="gem-test"),
        "be brief",
        opener,
    )
    assert live.provider == "geminilive"
    gpt = build_session_plan(
        make_settings(tmp_path, voice_provider="gptlive"),
        "be brief",
        opener,
    )
    assert gpt.provider == "gptlive"
    assert make_settings(tmp_path).voice_provider == "grok"


def test_pipeline_needs_a_gemini_key_and_service_account_json(tmp_path):
    grok = make_settings(tmp_path)
    assert "GEMINI_API_KEY" not in missing_dial_configuration(grok)
    assert "GOOGLE_TTS_CREDENTIALS_JSON" not in missing_dial_configuration(grok)

    missing_key = _pipeline_settings(tmp_path, gemini_api_key="")
    assert missing_dial_configuration(missing_key) == ["GEMINI_API_KEY"]
    missing_creds = _pipeline_settings(tmp_path, google_tts_credentials_json="")
    assert missing_dial_configuration(missing_creds) == ["GOOGLE_TTS_CREDENTIALS_JSON"]
    fallback = _pipeline_settings(
        tmp_path,
        google_tts_credentials_json="",
        google_cloud_credentials_json='{"project_id":"other"}',
    )
    assert missing_dial_configuration(fallback) == []
    ready = _pipeline_settings(tmp_path)
    assert missing_dial_configuration(ready) == []
    with pytest.raises(ValueError, match="VOICE_PROVIDER=pipeline"):
        provider_connection_target(ready)
    assert credential_json(fallback) == '{"project_id":"other"}'
    assert credential_json(ready).startswith('{"project_id":"tack45-test"')
    assert project_id_from_credentials("not-json") == ""
    assert project_id_from_credentials(credential_json(ready)) == "tack45-test"


def test_stt_config_is_mulaw_8k_en_au_with_interim_and_endpointing():
    config = stt_streaming_config("en-AU", "telephony", 500)
    decoding = config.config.explicit_decoding_config
    assert decoding.encoding == 2  # MULAW
    assert decoding.sample_rate_hertz == 8000
    assert decoding.audio_channel_count == 1
    assert list(config.config.language_codes) == ["en-AU"]
    assert config.config.model == "telephony"
    assert config.streaming_features.interim_results is True
    assert config.streaming_features.enable_voice_activity_events is True
    timeout = config.streaming_features.voice_activity_timeout.speech_end_timeout
    assert timeout == timedelta(milliseconds=500)
    assert stt_recognizer("tack45-test", "global") == (
        "projects/tack45-test/locations/global/recognizers/_"
    )
    assert speech_api_endpoint("global") is None
    assert speech_api_endpoint("us") == "us-speech.googleapis.com"
    requests = list(
        recognize_requests("tack45-test", "global", "en-AU", "telephony", 500, iter([b"\xff\xff", b""]))
    )
    assert requests[0].recognizer.endswith("/recognizers/_")
    assert requests[1].audio == b"\xff\xff"
    assert len(requests) == 2


def test_tts_config_is_chirp3_mulaw_8k():
    from google.cloud import texttospeech

    request = tts_config_request("en-AU-Chirp3-HD-Charon")
    voice = request.streaming_config.voice
    audio = request.streaming_config.streaming_audio_config
    assert voice.language_code == "en-AU"
    assert voice.name == "en-AU-Chirp3-HD-Charon"
    assert audio.audio_encoding == texttospeech.AudioEncoding.MULAW
    assert audio.sample_rate_hertz == 8000


def test_stt_notices_map_activity_and_finals():
    from google.cloud.speech_v2.types import cloud_speech

    begin = cloud_speech.StreamingRecognizeResponse(
        speech_event_type=cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN
    )
    end = cloud_speech.StreamingRecognizeResponse(
        speech_event_type=cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END
    )
    final = cloud_speech.StreamingRecognizeResponse(
        results=[
            cloud_speech.StreamingRecognitionResult(
                alternatives=[cloud_speech.SpeechRecognitionAlternative(transcript="Yes this is Sam.")],
                is_final=True,
            )
        ]
    )
    assert notices_from_stt(begin) == [SttNotice(speech_started=True)]
    assert notices_from_stt(end) == [SttNotice(speech_ended=True)]
    assert notices_from_stt(final) == [SttNotice(text="Yes this is Sam.", is_final=True, interim=False)]


def test_speech_end_waits_for_the_final_and_does_not_double_reply():
    turns = PipelineTurns()
    assert turns.push(SttNotice(speech_ended=True)) == (False, None)
    assert turns.push(SttNotice(text="Yes", interim=True)) == (False, None)
    started, text = turns.push(SttNotice(text="Yes this is Sam.", is_final=True))
    assert started is False
    assert text == "Yes this is Sam."
    assert turns.push(SttNotice(speech_ended=True)) == (False, None)
    started, _text = turns.push(SttNotice(speech_started=True))
    assert started is True
    assert turns.push(SttNotice(is_final=True)) == (False, None)
    started, _text = turns.push(SttNotice(speech_started=True))
    assert started is True


def test_outbound_opener_waits_for_hello_then_silence(caplog):
    caplog.set_level(logging.INFO, logger="tack45.pipeline")
    clock = Clock()
    controller = _controller(clock, hold=True)
    step = controller.on_twilio(START)
    assert step.plays == []
    assert controller.opener_pending is True
    assert controller.on_twilio(_media(SILENCE, 0.2, clock)).plays == []
    assert controller.on_twilio(_media(CLICK, 0.4, clock)).plays == []
    assert controller.poll_opener() is None
    assert controller.on_twilio(_media(SPEECH, 0.5, clock)).plays == []
    clock.now = 1.0
    released = controller.poll_opener()
    assert released is not None
    assert released.plays[0].text == "Hi, is this Sam?"
    assert released.plays[0].purpose == "opener"
    assert controller.opener_pending is False
    assert "Opener triggered by speech after 1000 ms" in caplog.text
    assert released.caller_audio is None


def test_opener_timeout_and_inbound_does_not_wait(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="tack45.pipeline")
    clock = Clock()
    waiting = _controller(clock, hold=True)
    waiting.on_twilio(START)
    clock.now = 2.5
    released = waiting.poll_opener()
    assert released is not None
    assert released.plays[0].purpose == "opener"
    assert "Opener triggered by the timeout after 2500 ms" in caplog.text

    inbound = _controller(Clock(), hold=False)
    immediate = inbound.on_twilio(START)
    assert immediate.plays[0].purpose == "opener"
    assert inbound.opener_pending is False
    settings = make_settings(tmp_path, opener_wait_for_speech=True)
    assert hold_opener(settings, {"direction": "inbound"}) is False
    assert hold_opener(settings, {"direction": "outbound"}) is True
    settings.opener_wait_for_speech = False
    assert hold_opener(settings, {"direction": "outbound"}) is False


def test_pipeline_utterance_ids_do_not_change_the_gemini_prefix():
    gemini = GeminiController(
        call_id="c1",
        call_sid="CA1",
        phone="+61400000000",
        opener="Hi, is this Sam?",
        instructions="system",
    )
    gemini_line = gemini.on_twilio(START).transcripts[0]
    assert gemini_line["utterance_id"].startswith("gemini-")
    pipeline = _controller(Clock(), hold=False)
    pipeline_line = pipeline.on_twilio(START).transcripts[0]
    assert pipeline_line["utterance_id"].startswith("pipeline-")


def test_rate_limit_log_points_at_ai_studio(caplog):
    caplog.set_level(logging.WARNING, logger="tack45.pipeline")
    controller = _controller(Clock(), hold=False)
    controller.started = True
    controller.stream_sid = "MZ1"
    step = controller.on_rate_limited("llm")
    assert step.plays[0].purpose == "fallback"
    assert "AI Studio" in caplog.text
    assert "10 requests/minute" not in caplog.text


class MemoryTwilio:
    def __init__(self) -> None:
        import asyncio

        self.inbound: asyncio.Queue = asyncio.Queue()
        self.outbound: asyncio.Queue = asyncio.Queue()

    async def receive_text(self) -> str:
        from starlette.websockets import WebSocketDisconnect

        item = await self.inbound.get()
        if item is None:
            raise WebSocketDisconnect()
        return json.dumps(item)

    async def send_json(self, data: dict) -> None:
        await self.outbound.put(data)

    async def inject(self, message: dict | None) -> None:
        await self.inbound.put(message)


class MemoryStore:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.opt_outs: list[dict] = []
        self.finished = None
        self.voice = None
        self.in_progress = False

    async def append_transcript(self, call_id: str, lines: list[dict]) -> None:
        return None

    async def record_opt_out(self, **kwargs) -> None:
        self.events.append("opt_out")
        self.opt_outs.append(kwargs)

    async def finish_call(self, call_id: str, *, end_reason: str, transcript: list[dict], status: str) -> None:
        self.events.append("finish")
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


class FakeStt:
    def __init__(self, clock: Clock) -> None:
        import asyncio

        self.clock = clock
        self.audio: list[bytes] = []
        self.closed = False
        self._events: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        return None

    async def send_audio(self, payload: bytes) -> None:
        self.audio.append(payload)

    async def recv(self):
        item = await self._events.get()
        if item is None:
            return None
        stamp, notice = item
        if stamp is not None:
            self.clock.now = stamp
        return notice

    async def close(self) -> None:
        self.closed = True
        await self._events.put(None)

    async def push(self, notice: SttNotice, at: float | None) -> None:
        await self._events.put((at, notice))


class FakeSpeaker:
    def __init__(self, sink: "SpeakerSink") -> None:
        import asyncio

        self.sink = sink
        self.texts: list[str] = []
        self.cancelled = False
        self._audio: asyncio.Queue = asyncio.Queue()

    def start(self) -> None:
        self.sink.started += 1

    async def send_text(self, text: str) -> None:
        self.texts.append(text)
        self.sink.texts.append(text)
        if self.cancelled:
            return
        if text.startswith("Thanks for calling"):
            self.sink.clock.now = 1.8
        await self._audio.put(b"\xff" * 320)

    async def finish(self) -> None:
        await self._audio.put(None)

    def cancel(self) -> None:
        self.cancelled = True
        self.sink.cancelled += 1
        self._audio.put_nowait(None)

    async def chunks(self):
        while True:
            item = await self._audio.get()
            if item is None:
                return
            yield item


class SpeakerSink:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.texts: list[str] = []
        self.cancelled = 0
        self.started = 0


class ScriptModel:
    def __init__(self, clock: Clock) -> None:
        import asyncio

        self.clock = clock
        self.release = asyncio.Event()
        self.calls: list[dict] = []

    async def stream_reply(self, *, instructions: str, transcript: list[dict]):
        self.calls.append({"instructions": instructions, "transcript": list(transcript)})
        self.clock.now = 1.5
        yield ModelEvent(text="Thanks for calling, is now ")
        await self.release.wait()
        yield ModelEvent(text="a good time?")


class OptOutModel:
    async def stream_reply(self, *, instructions: str, transcript: list[dict]):
        yield ModelEvent(tool_name="record_opt_out", tool_args={"reason": "stop calling"})
        yield ModelEvent(text="I'll remove your number and will not call again.")


async def _next(twilio: MemoryTwilio, kind: str, timeout: float = 2.0) -> dict:
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        item = await asyncio.wait_for(twilio.outbound.get(), timeout)
        if item.get("event") == kind:
            return item
    raise AssertionError(kind)


async def _echo_mark(twilio: MemoryTwilio) -> dict:
    mark = await _next(twilio, "mark")
    await twilio.inject(
        {"event": "mark", "streamSid": "MZ1", "mark": {"name": mark["mark"]["name"]}}
    )
    return mark


def _run_settings(tmp_path):
    return _pipeline_settings(
        tmp_path,
        opener_wait_for_speech=False,
        gemini_api_key="gem-secret-do-not-log",
        google_tts_credentials_json=(
            '{"project_id":"tack45-test","private_key":"FAKE-CLOUD-TTS-PRIVATE-KEY-SECRET"}'
        ),
    )


async def test_turn_latency_clause_flush_and_barge_in(tmp_path, caplog):
    import asyncio

    caplog.set_level(logging.INFO, logger="tack45.pipeline")
    clock = Clock()
    controller = _controller(clock, hold=False)
    listener = FakeStt(clock)
    model = ScriptModel(clock)
    sink = SpeakerSink(clock)
    store = MemoryStore()
    gateway = MemoryGateway()
    twilio = MemoryTwilio()
    task = asyncio.create_task(
        run_pipeline(
            twilio_ws=twilio,
            settings=_run_settings(tmp_path),
            controller=controller,
            store=store,
            gateway=gateway,
            listener=listener,
            text_model=model,
            tts_factory=lambda: FakeSpeaker(sink),
        )
    )
    await twilio.inject({"event": "connected"})
    await twilio.inject(START)
    opener = await _echo_mark(twilio)
    assert opener["mark"]["name"].startswith("opener-")
    await _until(lambda: controller.protecting_opener is False)

    await listener.push(SttNotice(speech_ended=True), at=1.0)
    await listener.push(SttNotice(text="Yes", interim=True), at=1.1)
    await asyncio.sleep(0.05)
    assert model.calls == []
    await listener.push(SttNotice(text="Yes this is Sam.", is_final=True), at=1.2)
    await _until(lambda: any(text.startswith("Thanks for calling") for text in sink.texts))
    await _until(lambda: "tts_audio_ms=800" in caplog.text)

    assert sink.texts[0] == "Hi, is this Sam?"
    assert sink.texts[1] == "Thanks for calling,"
    assert "stt_final_ms=200" in caplog.text
    assert "llm_token_ms=500" in caplog.text
    assert "tts_audio_ms=800" in caplog.text
    assert caplog.text.count("Pipeline turn latency") == 1
    assert "Yes this is Sam" not in caplog.text
    assert "gem-secret-do-not-log" not in caplog.text
    assert "FAKE-CLOUD-TTS-PRIVATE-KEY-SECRET" not in caplog.text
    assert model.calls[0]["instructions"].startswith("Be brief.")
    assert any(line["text"] == "Yes this is Sam." for line in model.calls[0]["transcript"])

    await listener.push(SttNotice(speech_started=True), at=None)
    cleared = await _next(twilio, "clear")
    assert cleared["streamSid"] == "MZ1"
    await _until(lambda: sink.cancelled >= 1)

    await twilio.inject({"event": "stop", "streamSid": "MZ1"})
    await asyncio.wait_for(task, 2)
    assert store.finished["end_reason"] == "caller_hangup"
    assert store.voice == "en-AU-Chirp3-HD-Charon"
    assert store.in_progress is True
    assert gateway.hungup == []
    assert listener.closed is True


async def test_opt_out_is_recorded_before_the_goodbye_finishes(tmp_path):
    import asyncio

    clock = Clock()
    controller = _controller(clock, hold=False)
    listener = FakeStt(clock)
    store = MemoryStore()
    gateway = MemoryGateway()
    twilio = MemoryTwilio()
    sink = SpeakerSink(clock)
    task = asyncio.create_task(
        run_pipeline(
            twilio_ws=twilio,
            settings=_run_settings(tmp_path),
            controller=controller,
            store=store,
            gateway=gateway,
            listener=listener,
            text_model=OptOutModel(),
            tts_factory=lambda: FakeSpeaker(sink),
        )
    )
    await twilio.inject({"event": "connected"})
    await twilio.inject(START)
    await _echo_mark(twilio)
    await _until(lambda: controller.protecting_opener is False)
    await listener.push(SttNotice(text="Please stop calling.", is_final=True), at=1.0)
    hangup = await _echo_mark(twilio)
    assert hangup["mark"]["name"].startswith("hangup-")
    await asyncio.wait_for(task, 2)
    assert store.events == ["opt_out", "finish"]
    assert store.opt_outs[0]["reason"] == "stop calling"
    assert store.finished["end_reason"] == "opt_out"
    assert gateway.hungup == ["CA1"]
    spoken = [line["text"] for line in store.finished["transcript"]]
    assert "Please stop calling." in spoken
    assert "I'll remove your number and will not call again." in spoken


async def _until(predicate, timeout: float = 2.0) -> None:
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met")


async def test_bad_credentials_are_not_logged(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="tack45.pipeline")
    secret = "FAKE-CLOUD-TTS-PRIVATE-KEY-SECRET"
    settings = _pipeline_settings(tmp_path, google_tts_credentials_json="not-json " + secret)
    controller = _controller(Clock(), hold=False)
    store = MemoryStore()
    await run_pipeline(
        twilio_ws=MemoryTwilio(),
        settings=settings,
        controller=controller,
        store=store,
        gateway=MemoryGateway(),
        listener=None,
        text_model=None,
        tts_factory=None,
    )
    assert "not valid JSON" in caplog.text
    assert secret not in caplog.text
    assert store.finished["end_reason"] == "stt_unavailable"


async def test_cloud_credentials_env_is_the_fallback(tmp_path, monkeypatch):
    seen: dict = {}

    def fake(raw: str):
        seen["raw"] = raw
        raise APIError(400, "not_configured", "stopped")

    monkeypatch.setattr("app.pipeline.cloudtts_credentials", fake)
    raw = '{"project_id":"fallback-project","private_key":"FALLBACK-PRIVATE-KEY-VALUE"}'
    settings = _pipeline_settings(
        tmp_path,
        google_tts_credentials_json="",
        google_cloud_credentials_json=raw,
    )
    store = MemoryStore()
    await run_pipeline(
        twilio_ws=MemoryTwilio(),
        settings=settings,
        controller=_controller(Clock(), hold=False),
        store=store,
        gateway=MemoryGateway(),
    )
    assert seen["raw"] == raw
    assert store.finished["end_reason"] == "stt_unavailable"


def test_invalid_service_account_json_message_has_no_secret():
    secret = "FAKE-CLOUD-TTS-PRIVATE-KEY-SECRET"
    with pytest.raises(APIError) as caught:
        cloudtts_credentials("not-json " + secret)
    assert "not valid JSON" in caught.value.message
    assert secret not in caught.value.message


async def test_text_model_streams_and_redacts_the_key(tmp_path, monkeypatch):
    secret = "gem-secret-do-not-log"
    settings = _pipeline_settings(tmp_path, gemini_api_key=secret, pipeline_llm_model="gemini-2.5-flash")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(429, text=f"quota exceeded for {secret}")

    transport = httpx.MockTransport(handler)

    class _Client(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("app.pipeline.httpx.AsyncClient", _Client)
    events = []
    async for event in PipelineTextModel(settings).stream_reply(
        instructions="I'm an AI assistant calling for Tack45.",
        transcript=[{"role": "caller", "text": "Yes"}],
    ):
        events.append(event)
    assert seen["key"] == secret
    assert "gemini-2.5-flash:streamGenerateContent" in seen["url"]
    assert "alt=sse" in seen["url"]
    assert "I'm an AI assistant" in seen["body"]["systemInstruction"]["parts"][0]["text"]
    names = [item["name"] for item in seen["body"]["tools"][0]["functionDeclarations"]]
    assert "end_call" in names and "record_opt_out" in names
    assert events[0].rate_limited is True
    assert secret not in (events[0].error or "")


def test_pipeline_answer_does_not_open_a_realtime_socket(tmp_path, twilio_capture, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    warmed: list[bool] = []
    monkeypatch.setattr("app.pipeline.schedule_pipeline_auth", lambda *_args, **_kwargs: warmed.append(True))

    def factory(*_args):
        raise AssertionError("pipeline must not open a realtime socket")

    application = create_app(
        _pipeline_settings(tmp_path),
        http_transport=twilio_capture["transport"],
        provider_factory=factory,
    )
    with TestClient(application) as http:
        created = http.post(
            "/calls",
            json={"to": "+61280001111", "name": "Pat", "business": "Quay Books"},
            headers=_auth(),
        )
        assert created.status_code == 201, created.text
        assert created.json()["provider"] == "pipeline"
        voice = http.post(
            f"/twilio/voice?call_id={created.json()['id']}",
            data={"CallSid": created.json()["call_sid"]},
        )
        assert voice.status_code == 200
        assert "<Connect>" in voice.text
        assert "wss://caller.example.com/streams/media/" in voice.text
    assert warmed == [True]
    assert twilio_capture["requests"]


def test_pipeline_still_refuses_outside_hours_and_do_not_call(tmp_path, twilio_capture, monkeypatch):
    sunday = datetime(2026, 1, 18, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("app.guards.utcnow", lambda: sunday)
    scheduled: list[bool] = []
    monkeypatch.setattr(
        "app.pipeline.schedule_pipeline_auth",
        lambda *_args, **_kwargs: scheduled.append(True),
    )
    settings = _pipeline_settings(tmp_path)
    application = create_app(settings, http_transport=twilio_capture["transport"])
    with TestClient(application) as http:
        refused = http.post(
            "/calls",
            json={"to": "+61280001111", "name": "Pat"},
            headers=_auth(),
        )
        assert refused.status_code == 403
        assert refused.json()["code"] == "outside_calling_hours"
        monkeypatch.setattr("app.guards.utcnow", _allowed_time)
        with open(settings.do_not_call_path, "w", encoding="utf-8") as handle:
            handle.write("+61280001111\n")
        blocked = http.post(
            "/calls",
            json={"to": "+61280001111", "name": "Pat"},
            headers=_auth(),
        )
        assert blocked.status_code == 403
        assert blocked.json()["code"] == "do_not_call"
    assert scheduled == []
    assert twilio_capture["requests"] == []


def test_pipeline_media_stream_still_schedules_the_transcript_webhook(tmp_path, twilio_capture, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    monkeypatch.setattr("app.pipeline.schedule_pipeline_auth", lambda *_args, **_kwargs: None)
    posted: list[str] = []

    async def finish_immediately(**kwargs):
        await kwargs["store"].finish_call(
            kwargs["call"]["id"],
            end_reason="completed",
            transcript=[{"role": "assistant", "text": "Hi, is this Pat?"}],
            status="completed",
        )

    monkeypatch.setattr("app.pipeline.run_pipeline_call", finish_immediately)

    def schedule(app, call_id: str) -> None:
        posted.append(call_id)

    monkeypatch.setattr("app.api.schedule_transcript_webhook", schedule)
    application = create_app(
        _pipeline_settings(tmp_path),
        http_transport=twilio_capture["transport"],
        provider_factory=lambda *_args: (_ for _ in ()).throw(AssertionError("realtime")),
    )
    with TestClient(application) as http:
        created = http.post(
            "/calls",
            json={"to": "+61280001111", "name": "Pat"},
            headers=_auth(),
        )
        assert created.status_code == 201, created.text
        call_id = created.json()["id"]
        voice = http.post(
            f"/twilio/voice?call_id={call_id}",
            data={"CallSid": created.json()["call_sid"]},
        )
        token = voice.text.split("/streams/media/")[1].split('"')[0].split("'")[0]
        with http.websocket_connect(f"/streams/media/{token}"):
            deadline = time.time() + 2
            while time.time() < deadline and posted != [call_id]:
                time.sleep(0.02)
        stored = http.get(f"/calls/{call_id}", headers=_auth()).json()
        assert stored["status"] == "completed"
        assert stored["end_reason"] == "completed"
    assert posted == [call_id]
