"""Cascaded phone pipeline for VOICE_PROVIDER=pipeline.

Google Cloud Speech-to-Text v2 listens to Twilio's 8 kHz μ-law. A Gemini
text model writes the reply from the same system prompt and brief as the
other providers. Cloud Text-to-Speech Chirp 3 HD streams μ-law straight
back to Twilio, starting at the first sentence or clause.

Geminilive and GPT-Live do not enter this module. Credentials and access
tokens are not logged.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import json
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import httpx
from starlette.websockets import WebSocketDisconnect

from app.api import APIError
from app.config import Settings
from app.gemini_turn import (
    GeminiController,
    GeminiStep,
    ModelEvent,
    PlayRequest,
    events_from_gemini_payload,
    is_rate_limit,
    take_sentences,
    text_request,
)
from app.opener_vad import OpenerVad
from app.relay import drain_queued_audio
from app.tts_probe import (
    _credential_secrets,
    _redact,
    cloudtts_credentials,
    mulaw_from_cloud_audio,
)

logger = logging.getLogger("tack45.pipeline")

_FRAME = 160  # 20 ms of 8 kHz μ-law
_LANGUAGE = "en-AU"
_GENERATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
_SPEECH_BEGIN = 2
_SPEECH_END = 3


@dataclass
class SttNotice:
    text: str = ""
    is_final: bool = False
    interim: bool = False
    speech_started: bool = False
    speech_ended: bool = False


@dataclass
class SpeechPlay:
    generation: int
    text: str
    purpose: str


@dataclass
class ReplyEnd:
    generation: int


def credential_json(settings: Settings) -> str:
    """Service-account JSON. The TTS env wins, then GOOGLE_CLOUD_CREDENTIALS_JSON."""
    primary = (settings.google_tts_credentials_json or "").strip()
    if primary:
        return primary
    return (settings.google_cloud_credentials_json or "").strip()


def project_id_from_credentials(raw: str) -> str:
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(info, dict):
        return ""
    project = info.get("project_id")
    if not isinstance(project, str):
        return ""
    return project.strip()


def hold_opener(settings: Settings, call: dict) -> bool:
    """Outbound calls wait for hello. Inbound callers are already on the line."""
    if not settings.opener_wait_for_speech:
        return False
    return (call.get("direction") or "outbound") != "inbound"


def stt_recognizer(project_id: str, location: str) -> str:
    place = (location or "global").strip() or "global"
    return f"projects/{project_id}/locations/{place}/recognizers/_"


def speech_api_endpoint(location: str) -> str | None:
    place = (location or "global").strip() or "global"
    if place == "global":
        return None
    return f"{place}-speech.googleapis.com"


def stt_streaming_config(language: str, model: str, endpoint_ms: int) -> Any:
    """v2 streaming config: μ-law 8 kHz, interim results, and endpointing."""
    from google.cloud.speech_v2.types import cloud_speech

    decoding = cloud_speech.ExplicitDecodingConfig(
        encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.MULAW,
        sample_rate_hertz=8000,
        audio_channel_count=1,
    )
    recognition = cloud_speech.RecognitionConfig(
        explicit_decoding_config=decoding,
        model=model,
        language_codes=[language or _LANGUAGE],
    )
    features = cloud_speech.StreamingRecognitionFeatures(
        interim_results=True,
        enable_voice_activity_events=True,
        voice_activity_timeout=cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
            speech_end_timeout=timedelta(milliseconds=max(1, int(endpoint_ms))),
        ),
    )
    return cloud_speech.StreamingRecognitionConfig(
        config=recognition,
        streaming_features=features,
    )


def recognize_requests(
    project_id: str,
    location: str,
    language: str,
    model: str,
    endpoint_ms: int,
    audio: Iterator[bytes],
) -> Iterator[Any]:
    from google.cloud.speech_v2.types import cloud_speech

    yield cloud_speech.StreamingRecognizeRequest(
        recognizer=stt_recognizer(project_id, location),
        streaming_config=stt_streaming_config(language, model, endpoint_ms),
    )
    for chunk in audio:
        if chunk:
            yield cloud_speech.StreamingRecognizeRequest(audio=chunk)


def notices_from_stt(response: Any) -> list[SttNotice]:
    notices: list[SttNotice] = []
    event = getattr(response, "speech_event_type", 0) or 0
    name = getattr(event, "name", "")
    if name == "SPEECH_ACTIVITY_BEGIN" or event == _SPEECH_BEGIN:
        notices.append(SttNotice(speech_started=True))
    elif name == "SPEECH_ACTIVITY_END" or event == _SPEECH_END:
        notices.append(SttNotice(speech_ended=True))
    for result in getattr(response, "results", None) or []:
        alternatives = getattr(result, "alternatives", None) or []
        text = ""
        if alternatives:
            text = str(getattr(alternatives[0], "transcript", "") or "").strip()
        final = bool(getattr(result, "is_final", False))
        if text or final:
            notices.append(SttNotice(text=text, is_final=final, interim=bool(text) and not final))
    return notices


def tts_config_request(voice: str) -> Any:
    from google.cloud import texttospeech

    return texttospeech.StreamingSynthesizeRequest(
        streaming_config=texttospeech.StreamingSynthesizeConfig(
            voice=texttospeech.VoiceSelectionParams(language_code=_LANGUAGE, name=voice),
            streaming_audio_config=texttospeech.StreamingAudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MULAW,
                sample_rate_hertz=8000,
            ),
        )
    )


def tts_text_request(text: str) -> Any:
    from google.cloud import texttospeech

    return texttospeech.StreamingSynthesizeRequest(
        input=texttospeech.StreamingSynthesisInput(text=text)
    )


def _mulaw_bytes(payload: bytes) -> bytes:
    if not payload:
        return b""
    if payload[:4] != b"RIFF":
        return bytes(payload)
    audio, _ms, error = mulaw_from_cloud_audio(payload, time.monotonic)
    if error or not audio:
        return b""
    return audio


def _delta_ms(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return "-"
    return str(max(0, int(round((end - start) * 1000))))


class TurnLatency:
    """Milliseconds from the end of caller speech. Logged once per turn."""

    def __init__(self, clock: Callable[[], float], call_id: str) -> None:
        self.clock = clock
        self.call_id = call_id
        self.turns: list[dict[str, int | None]] = []
        self.reset()

    def reset(self) -> None:
        self.speech_ended_at: float | None = None
        self.stt_final_at: float | None = None
        self.llm_token_at: float | None = None
        self.tts_audio_at: float | None = None
        self.logged = False

    def speech_ended(self) -> None:
        if self.speech_ended_at is None:
            self.speech_ended_at = self.clock()

    def heard(self) -> None:
        """The transcript that actually started the reply, not a late final."""
        now = self.clock()
        if self.speech_ended_at is None:
            self.speech_ended_at = now
        if self.stt_final_at is None:
            self.stt_final_at = now

    def stt_final(self) -> None:
        now = self.clock()
        if self.speech_ended_at is None:
            self.speech_ended_at = now
        if self.stt_final_at is None:
            self.stt_final_at = now

    def llm_token(self) -> None:
        if self.llm_token_at is None:
            self.llm_token_at = self.clock()

    def tts_audio(self) -> None:
        if self.tts_audio_at is None:
            self.tts_audio_at = self.clock()
            self.log()

    def _ms(self, end: float | None) -> int | None:
        if self.speech_ended_at is None or end is None:
            return None
        return max(0, int(round((end - self.speech_ended_at) * 1000)))

    def summary(self) -> dict[str, int | None]:
        audio = self._ms(self.tts_audio_at)
        return {
            "speech_to_audio_ms": audio,
            "stt_ms": self._ms(self.stt_final_at),
            "llm_ms": self._ms(self.llm_token_at),
            "tts_ms": audio,
        }

    def log(self) -> None:
        if self.logged or self.speech_ended_at is None:
            return
        self.logged = True
        self.turns.append(self.summary())
        logger.info(
            "Pipeline turn latency call=%s stt_final_ms=%s llm_token_ms=%s tts_audio_ms=%s",
            self.call_id,
            _delta_ms(self.speech_ended_at, self.stt_final_at),
            _delta_ms(self.speech_ended_at, self.llm_token_at),
            _delta_ms(self.speech_ended_at, self.tts_audio_at),
        )


def _utterance_words(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return cleaned.split()


def _phrase_in(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1))


class PipelineTurns:
    """Interim text barges in. Voice-activity end starts the reply.

    A speech-end with interim text, or the next interim after that end,
    starts the reply. The later final for that same utterance does not.
    A final that arrives on its own still starts the reply. A tail the next
    Speech-to-Text stream transcribes from the same words is ignored.
    """

    def __init__(self) -> None:
        self._interim = ""
        self._final = ""
        self._spoken = ""
        self._talking = False
        self._closing = False
        self._emitted = False

    def push(self, notice: SttNotice) -> tuple[bool, str | None]:
        if notice.speech_started:
            if self._closing and not self._emitted:
                return False, None
            self._closing = False
            started = not self._talking
            self._talking = True
            self._arm()
            return started, None
        if notice.interim and notice.text:
            if self._emitted and not self._talking and (self._closing or self._repeats(notice.text)):
                return False, None
            if self._closing and not self._emitted:
                self._interim = notice.text
                return False, self._take()
            started = not self._talking
            self._talking = True
            self._arm()
            self._interim = notice.text
            return started, None
        if notice.is_final:
            if self._emitted and self._closing:
                self._closing = False
                self._talking = False
                self._interim = ""
                self._final = ""
                return False, None
            if self._emitted and not self._talking and self._repeats(notice.text):
                self._closing = False
                self._interim = ""
                self._final = ""
                return False, None
            self._closing = False
            self._talking = False
            if not notice.text:
                return False, None
            self._arm()
            self._final = notice.text
            self._interim = ""
            return False, self._take()
        if notice.speech_ended:
            # A late end after the reply started must not swallow the next utterance.
            if self._emitted and not self._talking and not self._interim:
                return False, None
            self._talking = False
            self._closing = True
            if self._interim and not self._emitted:
                return False, self._take()
        return False, None

    def _repeats(self, text: str) -> bool:
        new = _utterance_words(text)
        old = _utterance_words(self._spoken)
        if not new or not old:
            return False
        return new == old or _phrase_in(old, new) or _phrase_in(new, old)

    def _arm(self) -> None:
        if self._emitted:
            self._emitted = False
            self._final = ""
            self._interim = ""
            self._spoken = ""

    def _take(self) -> str | None:
        if self._emitted:
            return None
        text = (self._final or self._interim).strip()
        self._final = ""
        self._interim = ""
        if not text:
            return None
        self._emitted = True
        self._spoken = text
        return text


class PipelineController(GeminiController):
    """Gemini turn logic, plus the outbound wait for hello before the opener."""

    def __init__(
        self,
        *,
        hold_opener: bool,
        silence_ms: int,
        max_wait_ms: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(utterance_prefix="pipeline", **kwargs)
        self.hold_opener = hold_opener
        self.opener_pending = False
        self._vad = OpenerVad(silence_ms=silence_ms, max_wait_ms=max_wait_ms)
        self._stream_started_at: float | None = None

    def on_rate_limited(self, source: str) -> GeminiStep:
        logger.warning(
            "Pipeline model 429 from %s on call %s. Each caller turn is one "
            "generateContent request. Check that Gemini key's limits in AI Studio. "
            "A 429 ends the call.",
            source,
            self.call_id,
        )
        return self._begin_fallback("rate_limited")

    def poll_opener(self) -> GeminiStep | None:
        if not self.opener_pending:
            return None
        decision = self._vad.poll(self.clock())
        if decision is None:
            return None
        return self._release_opener(decision)

    def _on_start(self, message: dict) -> GeminiStep:
        start = message.get("start") or {}
        self.stream_sid = start.get("streamSid") or message.get("streamSid")
        self.call_sid = start.get("callSid") or self.call_sid
        self.started = True
        self._stream_started_at = self.clock()
        if self.hold_opener:
            self.opener_pending = True
            self.protecting_opener = True
            self._vad.start(self.clock())
            return GeminiStep()
        return self._speak_opener(GeminiStep())

    def _on_media(self, message: dict) -> GeminiStep:
        if not self.started or self.hangup_requested:
            return GeminiStep()
        media = message.get("media") or {}
        if media.get("track") not in (None, "", "inbound"):
            return GeminiStep()
        payload = media.get("payload") or ""
        if not payload:
            return GeminiStep()
        if self.opener_pending:
            try:
                mulaw = base64.b64decode(payload, validate=False)
            except Exception:
                mulaw = b""
            decision = self._vad.feed(mulaw, self.clock())
            if decision is None:
                return GeminiStep()
            return self._release_opener(decision)
        return GeminiStep(caller_audio=payload)

    def _speak_opener(self, step: GeminiStep) -> GeminiStep:
        self.opener_pending = False
        self.protecting_opener = True
        line = self._commit("assistant", self.opener)
        step.plays.append(PlayRequest(self.opener, "opener"))
        if line:
            step.transcripts.append(line)
        return step

    def _release_opener(self, reason: str) -> GeminiStep:
        if not self.opener_pending:
            return GeminiStep()
        elapsed = 0
        if self._stream_started_at is not None:
            elapsed = max(0, int(round((self.clock() - self._stream_started_at) * 1000)))
        if reason == "speech":
            logger.info("Opener triggered by speech after %d ms", elapsed)
        else:
            logger.info("Opener triggered by the timeout after %d ms", elapsed)
        return self._speak_opener(GeminiStep())


class PipelineTextModel:
    """Streaming generateContent. The model name is PIPELINE_LLM_MODEL."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def stream_reply(self, *, instructions: str, transcript: list[dict]) -> AsyncIterator[ModelEvent]:
        body = apply_pipeline_thinking(text_request(instructions, transcript, self.settings), self.settings)
        url = _GENERATE.format(model=self.settings.pipeline_llm_model)
        secret = self.settings.gemini_api_key
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                async with client.stream(
                    "POST",
                    url,
                    params={"alt": "sse"},
                    headers={"x-goog-api-key": secret, "Content-Type": "application/json"},
                    json=body,
                ) as response:
                    if response.status_code == 429:
                        raw = (await response.aread()).decode("utf-8", errors="replace")
                        yield ModelEvent(rate_limited=True, error=_redact(f"HTTP 429: {raw}", secret))
                        return
                    if response.status_code >= 400:
                        raw = (await response.aread()).decode("utf-8", errors="replace")
                        message = _redact(f"HTTP {response.status_code}: {raw}", secret)
                        yield ModelEvent(
                            rate_limited=is_rate_limit(response.status_code, message),
                            error=message,
                        )
                        return
                    async for line in response.aiter_lines():
                        payload = _sse_json(line)
                        if payload is None:
                            continue
                        for event in events_from_gemini_payload(payload):
                            if event.error:
                                event.error = _redact(event.error, secret)
                            yield event
            except httpx.HTTPError as exc:
                message = _redact(str(exc), secret)
                yield ModelEvent(rate_limited=is_rate_limit(None, message), error=message)


def apply_pipeline_thinking(body: dict, settings: Settings) -> dict:
    """Force the pipeline thinking budget. The cascade setting does not apply.

    gemini-2.5-flash thinks before the first token unless thinkingBudget is 0.
    A negative PIPELINE_LLM_THINKING_BUDGET omits the field.
    """
    budget = settings.pipeline_llm_thinking_budget
    generation = body.setdefault("generationConfig", {})
    if isinstance(budget, int) and not isinstance(budget, bool) and budget >= 0:
        generation["thinkingConfig"] = {"thinkingBudget": budget}
    else:
        generation.pop("thinkingConfig", None)
    return body


def _sse_json(line: str) -> Any:
    text = line.strip()
    if not text or text.startswith(":"):
        return None
    if text.startswith("data:"):
        text = text[5:].strip()
    if not text or text == "[DONE]":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


_client_lock = threading.Lock()
_speech_clients: dict[tuple[int, str], Any] = {}
_tts_clients: dict[int, Any] = {}


def _speech_client(credentials: Any, location: str) -> Any:
    """One Speech client per credentials object and location. The channel is reused."""
    place = (location or "global").strip() or "global"
    key = (id(credentials), place)
    with _client_lock:
        cached = _speech_clients.get(key)
        if cached is not None:
            return cached
    from google.api_core.client_options import ClientOptions
    from google.cloud.speech_v2 import SpeechClient

    endpoint = speech_api_endpoint(place)
    options = ClientOptions(api_endpoint=endpoint) if endpoint else None
    client = SpeechClient(credentials=credentials, client_options=options)
    with _client_lock:
        existing = _speech_clients.get(key)
        if existing is not None:
            return existing
        _speech_clients[key] = client
        return client


class _AudioBuffer:
    """μ-law waiting for a streaming recognize session.

    A session that is stopping does not take audio. Those chunks stay here
    and the next session sends them.
    """

    def __init__(self) -> None:
        self._items: collections.deque[bytes] = collections.deque()
        self._cond = threading.Condition()
        self.closed = False

    def add(self, payload: bytes) -> None:
        if not payload:
            return
        with self._cond:
            if self.closed:
                return
            self._items.append(payload)
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def waiting(self) -> int:
        with self._cond:
            return len(self._items)

    def take(self, stop: threading.Event) -> bytes | None:
        """Next chunk, or None when this session must stop. None leaves the audio queued."""
        with self._cond:
            while not self._items and not self.closed and not stop.is_set():
                self._cond.wait(timeout=0.2)
            if stop.is_set() or self.closed:
                return None
            return self._items.popleft()

    def wait_closed(self, timeout: float) -> None:
        with self._cond:
            if self.closed:
                return
            self._cond.wait(timeout=timeout)


class GoogleStt:
    """StreamingRecognize for the whole call.

    Cloud Speech closes a stream after a final result, on endpointing, at the
    roughly five-minute limit, or on a transient error. The next stream is
    opened when speech ends or a final arrives, while the old RPC is still
    up, so the gap is not paid at the start of the next turn. Audio stays
    queued across that handoff. ``recv`` returns None only after ``close``.
    Nothing from the service-account key is logged.
    """

    def __init__(
        self,
        credentials: Any,
        *,
        project_id: str,
        location: str,
        model: str,
        language: str,
        endpoint_ms: int,
        secrets: list[str],
        call_id: str = "",
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._credentials = credentials
        self._project_id = project_id
        self._location = location
        self._model = model
        self._language = language
        self._endpoint_ms = endpoint_ms
        self._secrets = secrets
        self._call_id = call_id
        self._client_factory = client_factory or _speech_client
        self._buffer = _AudioBuffer()
        self._events: queue.Queue[SttNotice | None] = queue.Queue()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._stop: threading.Event | None = None
        self._client: Any = None
        self._live = False
        self._generation = 0
        self._error_streak = 0
        self.closed = False
        self.restarts = 0

    async def connect(self) -> None:
        if not self._project_id:
            raise RuntimeError("Service account JSON has no project_id.")
        thread = threading.Thread(target=self._bootstrap, name="pipeline-stt", daemon=True)
        with self._lock:
            self._threads.append(thread)
        thread.start()

    async def send_audio(self, payload: bytes) -> None:
        self._buffer.add(payload)

    async def recv(self) -> SttNotice | None:
        return await asyncio.to_thread(self._events.get)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._buffer.close()
        with self._lock:
            stop = self._stop
            threads = list(self._threads)
        if stop is not None:
            stop.set()
        self._buffer.wake()
        for thread in threads:
            if thread is not threading.current_thread():
                await asyncio.to_thread(thread.join, 2.0)
        self._events.put(None)

    def _chunks(self, stop: threading.Event) -> Iterator[bytes]:
        while not stop.is_set() and not self._buffer.closed:
            chunk = self._buffer.take(stop)
            if chunk is None:
                return
            yield chunk

    def _log_restart(self, reason: str, exc: BaseException | None) -> None:
        self.restarts += 1
        if exc is None:
            logger.info(
                "Pipeline STT stream restarting call=%s restart=%d reason=%s buffered=%d",
                self._call_id,
                self.restarts,
                reason,
                self._buffer.waiting(),
            )
            return
        logger.warning(
            "Pipeline STT stream restarting call=%s restart=%d reason=%s buffered=%d error=%s",
            self._call_id,
            self.restarts,
            reason,
            self._buffer.waiting(),
            _redact(str(exc), *self._secrets),
        )

    def _backoff(self) -> float:
        self._error_streak += 1
        return min(2.0, 0.05 * (2 ** min(self._error_streak, 5)))

    def _bootstrap(self) -> None:
        while not self._buffer.closed:
            try:
                self._client = self._client_factory(self._credentials, self._location)
                break
            except Exception as exc:
                self._log_restart("client", exc)
                self._buffer.wait_closed(self._backoff())
        if self._buffer.closed or self._client is None:
            return
        self._start_session(None, None, 0.0)

    def _start_session(
        self,
        reason: str | None,
        exc: BaseException | None,
        backoff: float,
    ) -> bool:
        with self._lock:
            if self.closed or self._buffer.closed or self._live:
                return False
            self._live = True
            self._generation += 1
            generation = self._generation
        thread = threading.Thread(
            target=self._session,
            args=(generation, reason, exc, backoff),
            name="pipeline-stt",
            daemon=True,
        )
        with self._lock:
            self._threads.append(thread)
        thread.start()
        return True

    def _release(self, generation: int) -> None:
        with self._lock:
            if self._generation == generation:
                self._live = False

    def _session(
        self,
        generation: int,
        reason: str | None,
        exc: BaseException | None,
        backoff: float,
    ) -> None:
        if backoff > 0:
            self._buffer.wait_closed(backoff)
        if self._buffer.closed:
            self._release(generation)
            return
        if reason:
            self._log_restart(reason, exc)
        stop = threading.Event()
        with self._lock:
            if self._generation == generation:
                self._stop = stop
        handed_off = False
        got_response = False
        stream_exc: BaseException | None = None
        try:
            requests = recognize_requests(
                self._project_id,
                self._location,
                self._language,
                self._model,
                self._endpoint_ms,
                self._chunks(stop),
            )
            for response in self._client.streaming_recognize(requests=requests):
                if self._buffer.closed:
                    break
                got_response = True
                handoff = False
                for notice in notices_from_stt(response):
                    self._events.put(notice)
                    if notice.speech_ended or notice.is_final:
                        handoff = True
                if handoff and not handed_off:
                    # Stop feeding this stream. Keep reading it so a late final
                    # still arrives. The next stream is already opening.
                    stop.set()
                    self._buffer.wake()
                    self._release(generation)
                    if self._start_session("preopened", None, 0.0):
                        handed_off = True
                    else:
                        with self._lock:
                            if self._generation == generation and not self.closed:
                                self._live = True
        except Exception as caught:
            stream_exc = caught
        finally:
            stop.set()
            self._buffer.wake()
            with self._lock:
                if self._stop is stop:
                    self._stop = None
            if handed_off or self._buffer.closed:
                self._release(generation)
                return
            if got_response and stream_exc is None:
                self._error_streak = 0
                delay = 0.0
                next_reason = "ended"
            elif got_response:
                self._error_streak = 0
                delay = 0.0
                next_reason = "error"
            else:
                delay = self._backoff()
                next_reason = "error" if stream_exc is not None else "ended"
            self._release(generation)
            self._start_session(next_reason, stream_exc, delay)


def _tts_client(credentials: Any) -> Any:
    """One Text-to-Speech client per credentials object. The channel is reused."""
    key = id(credentials)
    with _client_lock:
        cached = _tts_clients.get(key)
        if cached is not None:
            return cached
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient(credentials=credentials)
    with _client_lock:
        existing = _tts_clients.get(key)
        if existing is not None:
            return existing
        _tts_clients[key] = client
        return client


def _warm_channel(client: Any, secrets: list[str]) -> None:
    transport = getattr(client, "transport", None)
    channel = getattr(transport, "grpc_channel", None)
    if channel is None:
        return
    import grpc

    try:
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as exc:
        logger.warning("Pipeline channel warm-up failed: %s", _redact(str(exc), *secrets))


def warm_pipeline_channels(credentials: Any, location: str, secrets: list[str] | None = None) -> None:
    """Open the Speech and Text-to-Speech channels before the first caller turn.

    This waits until each channel is ready. It does not call recognize or
    synthesize, so it is not a billed request. A failure is logged and ignored.
    """
    hidden = list(secrets or [])
    try:
        _warm_channel(_speech_client(credentials, location), hidden)
    except Exception as exc:
        logger.warning("Pipeline channel warm-up failed: %s", _redact(str(exc), *hidden))
    try:
        _warm_channel(_tts_client(credentials), hidden)
    except Exception as exc:
        logger.warning("Pipeline channel warm-up failed: %s", _redact(str(exc), *hidden))


class CloudStreamingSpeaker:
    """One StreamingSynthesize session. Clauses go in, μ-law chunks come out."""

    def __init__(self, credentials: Any, voice: str, *, secrets: list[str] | None = None) -> None:
        self._credentials = credentials
        self._voice = voice
        self._secrets = list(secrets or [])
        self._text: queue.Queue[str | None] = queue.Queue()
        self._audio: queue.Queue[bytes | Exception | None] = queue.Queue()
        self._cancelled = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="pipeline-tts", daemon=True)
        self._thread.start()

    async def send_text(self, text: str) -> None:
        if text and not self._cancelled:
            self._text.put(text)

    async def finish(self) -> None:
        self._text.put(None)

    def cancel(self) -> None:
        self._cancelled = True
        self._text.put(None)
        self._audio.put(None)

    async def chunks(self) -> AsyncIterator[bytes]:
        while True:
            item = await asyncio.to_thread(self._audio.get)
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def _requests(self) -> Iterator[Any]:
        yield tts_config_request(self._voice)
        while True:
            item = self._text.get()
            if item is None or self._cancelled:
                return
            yield tts_text_request(item)

    def _run(self) -> None:
        try:
            client = _tts_client(self._credentials)
            for response in client.streaming_synthesize(self._requests()):
                if self._cancelled:
                    break
                chunk = _mulaw_bytes(bytes(getattr(response, "audio_content", b"") or b""))
                if chunk:
                    self._audio.put(chunk)
        except Exception as exc:
            if not self._cancelled:
                self._audio.put(RuntimeError(_redact(str(exc), *self._secrets)))
        finally:
            self._audio.put(None)


def schedule_pipeline_auth(app, settings: Settings) -> None:
    """Mint the access token while TwiML is returned. The JSON is not logged."""
    raw = credential_json(settings)
    if not raw:
        return

    def _prepare() -> None:
        credentials = cloudtts_credentials(raw)
        warm_pipeline_channels(credentials, settings.pipeline_stt_location, _credential_secrets(raw))

    async def _warm() -> None:
        try:
            await asyncio.to_thread(_prepare)
        except APIError as exc:
            logger.error("Pipeline credential warm-up failed: %s", exc.message)
        except Exception as exc:
            logger.error(
                "Pipeline credential warm-up failed: %s",
                _redact(str(exc), *_credential_secrets(raw)),
            )

    task = asyncio.create_task(_warm(), name="pipeline-auth")
    tasks = getattr(app.state, "pipeline_auth_tasks", None)
    if tasks is None:
        tasks = set()
        app.state.pipeline_auth_tasks = tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)


async def run_pipeline_call(
    *,
    websocket,
    settings: Settings,
    call: dict,
    plan,
    store,
    gateway,
    phone: str,
    listener=None,
    text_model=None,
    tts_factory=None,
    clock: Callable[[], float] | None = None,
) -> None:
    message = plan.setup_messages[0]
    controller = PipelineController(
        call_id=call["id"],
        call_sid=call.get("call_sid"),
        phone=phone,
        opener=message["opener"],
        instructions=message["instructions"],
        hangup_grace_seconds=settings.hangup_grace_seconds,
        clock=clock or time.monotonic,
        hold_opener=hold_opener(settings, call),
        silence_ms=settings.opener_silence_ms,
        max_wait_ms=settings.opener_max_wait_ms,
    )
    await run_pipeline(
        twilio_ws=websocket,
        settings=settings,
        controller=controller,
        store=store,
        gateway=gateway,
        listener=listener,
        text_model=text_model,
        tts_factory=tts_factory,
    )


async def run_pipeline(
    *,
    twilio_ws,
    settings: Settings,
    controller: PipelineController,
    store,
    gateway,
    listener=None,
    text_model=None,
    tts_factory=None,
) -> None:
    raw = credential_json(settings)
    secrets = _credential_secrets(raw, settings.gemini_api_key)
    if listener is None or tts_factory is None:
        if not raw:
            logger.error("Pipeline credentials are not configured.")
            await _finish_now(controller, store, gateway, "stt_unavailable")
            return
        try:
            credentials = await asyncio.to_thread(cloudtts_credentials, raw)
        except APIError as exc:
            logger.error("Pipeline credentials are not usable: %s", exc.message)
            await _finish_now(controller, store, gateway, "stt_unavailable")
            return
        except Exception as exc:
            logger.error("Pipeline credentials are not usable: %s", _redact(str(exc), *secrets))
            await _finish_now(controller, store, gateway, "stt_unavailable")
            return
        project = project_id_from_credentials(raw)
        if listener is None:
            listener = GoogleStt(
                credentials,
                project_id=project,
                location=settings.pipeline_stt_location,
                model=settings.pipeline_stt_model,
                language=settings.pipeline_stt_language,
                endpoint_ms=settings.pipeline_stt_endpoint_ms,
                secrets=secrets,
                call_id=controller.call_id,
            )
        if tts_factory is None:
            tts_factory = lambda: CloudStreamingSpeaker(
                credentials,
                settings.pipeline_tts_voice,
                secrets=secrets,
            )
    text_model = text_model or PipelineTextModel(settings)
    turns = PipelineTurns()
    latency = TurnLatency(controller.clock, controller.call_id)
    twilio_out: asyncio.Queue = asyncio.Queue()
    speak_q: asyncio.Queue = asyncio.Queue()
    llm_task: asyncio.Task | None = None
    speaker: Any = None
    audio_task: asyncio.Task | None = None
    finished = False

    async def finish(reason: str) -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        controller.hung_up = True
        latency.log()
        try:
            await store.finish_call(
                controller.call_id,
                end_reason=reason,
                transcript=list(controller.transcript),
                status="completed",
                turn_latencies=list(latency.turns),
            )
        except Exception:
            logger.error("Failed to store the final call record for %s", controller.call_id)
        if gateway and controller.call_sid and reason != "caller_hangup":
            try:
                await gateway.hangup(controller.call_sid)
            except Exception:
                logger.error("Twilio hangup failed for %s", controller.call_sid)

    def cancel_speech() -> None:
        nonlocal llm_task, speaker
        current = asyncio.current_task()
        if llm_task is not None and llm_task is not current and not llm_task.done():
            llm_task.cancel()
            llm_task = None
        if speaker is not None:
            speaker.cancel()
        while True:
            try:
                speak_q.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def close_speaker() -> None:
        nonlocal speaker, audio_task
        current = speaker
        task = audio_task
        speaker = None
        audio_task = None
        if current is not None:
            await current.finish()
        if task is not None:
            await task

    def open_speaker(generation: int) -> None:
        nonlocal speaker, audio_task
        speaker = tts_factory()
        speaker.start()
        audio_task = asyncio.create_task(forward_audio(speaker, generation), name="pipeline-audio")

    async def forward_audio(active, generation: int) -> None:
        # 20 ms frames, sent as soon as they exist. Twilio plays on arrival,
        # so sleeping to pace them would not shorten the gap before the answer.
        # A first chunk shorter than one frame is sent immediately.
        pending = b""
        sent_any = False
        try:
            async for chunk in active.chunks():
                if generation != controller.generation or controller.interrupted:
                    return
                # The opener and the polite goodbye are not a caller turn.
                if controller.reply_active:
                    latency.tts_audio()
                pending += chunk
                if not sent_any and 0 < len(pending) < _FRAME:
                    sent_any = True
                    if not controller.note_playback(len(pending), generation) or not controller.stream_sid:
                        return
                    frame, pending = pending, b""
                    await twilio_out.put(
                        {
                            "event": "media",
                            "streamSid": controller.stream_sid,
                            "media": {"payload": base64.b64encode(frame).decode("ascii")},
                        }
                    )
                    continue
                while len(pending) >= _FRAME:
                    sent_any = True
                    if not controller.note_playback(_FRAME, generation) or not controller.stream_sid:
                        return
                    frame, pending = pending[:_FRAME], pending[_FRAME:]
                    await twilio_out.put(
                        {
                            "event": "media",
                            "streamSid": controller.stream_sid,
                            "media": {"payload": base64.b64encode(frame).decode("ascii")},
                        }
                    )
            if pending and generation == controller.generation and not controller.interrupted:
                if controller.note_playback(len(pending), generation) and controller.stream_sid:
                    await twilio_out.put(
                        {
                            "event": "media",
                            "streamSid": controller.stream_sid,
                            "media": {"payload": base64.b64encode(pending).decode("ascii")},
                        }
                    )
        except Exception as exc:
            message = _redact(str(exc), *secrets)
            logger.error("Pipeline TTS failed on call %s: %s", controller.call_id, message)
            await speak_q.put(RuntimeError(message))

    async def dispatch(step: GeminiStep) -> bool:
        if step.cancel_speech:
            cancel_speech()
        if step.drop_queued_audio:
            drain_queued_audio(twilio_out)
        for message in step.twilio_messages:
            await twilio_out.put(message)
        if step.transcripts:
            try:
                await store.append_transcript(controller.call_id, step.transcripts)
            except Exception:
                logger.error("Failed to append the transcript for %s", controller.call_id)
        if step.opt_out_reason is not None:
            try:
                await store.record_opt_out(
                    call_id=controller.call_id,
                    phone=controller.phone,
                    call_sid=controller.call_sid,
                    reason=step.opt_out_reason,
                )
            except Exception:
                logger.error("Failed to record opt-out for %s", controller.call_id)
        generation = controller.generation
        for play in step.plays:
            if play.cached_only:
                continue
            await speak_q.put(SpeechPlay(generation, play.text, play.purpose))
        if step.begin_reply:
            start_reply(generation)
        if step.follow_with_end:
            await speak_q.put(ReplyEnd(generation))
        if step.hangup:
            await finish(step.end_reason or controller.end_reason)
            return True
        return False

    def start_reply(generation: int) -> None:
        nonlocal llm_task
        current = asyncio.current_task()
        if llm_task is not None and llm_task is not current and not llm_task.done():
            llm_task.cancel()
        llm_task = asyncio.create_task(_reply(generation), name="pipeline-reply")

    async def _reply(generation: int) -> None:
        buffer = ""
        early = True
        try:
            async for event in text_model.stream_reply(
                instructions=controller.instructions,
                transcript=list(controller.transcript),
            ):
                if generation != controller.generation:
                    return
                if event.rate_limited:
                    await dispatch(controller.on_rate_limited("llm"))
                    return
                if event.error:
                    await dispatch(controller.on_model_error(_redact(event.error, *secrets)))
                    return
                if event.tool_name:
                    if await dispatch(controller.on_tool(event.tool_name, event.tool_args or {}, generation)):
                        return
                    if generation != controller.generation:
                        return
                if event.text:
                    latency.llm_token()
                    sentences, buffer = take_sentences(
                        buffer + event.text,
                        flush=False,
                        early_clause=early,
                        flush_trailing=True,
                        early_words=6,
                    )
                    if sentences:
                        early = False
                    for sentence in sentences:
                        if await dispatch(controller.on_assistant_sentence(sentence, generation)):
                            return
            if generation != controller.generation:
                return
            sentences, _rest = take_sentences(
                buffer,
                flush=True,
                early_clause=early,
                flush_trailing=True,
                early_words=6,
            )
            for sentence in sentences:
                if await dispatch(controller.on_assistant_sentence(sentence, generation)):
                    return
            await speak_q.put(ReplyEnd(generation))
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return

    async def player() -> None:
        while True:
            item = await speak_q.get()
            if item is None:
                await close_speaker()
                return
            if isinstance(item, RuntimeError):
                if await dispatch(controller.on_model_error(str(item))):
                    return
                continue
            if getattr(item, "generation", None) != controller.generation:
                continue
            if isinstance(item, ReplyEnd):
                await close_speaker()
                if item.generation != controller.generation:
                    continue
                if await dispatch(controller.on_reply_finished(item.generation)):
                    return
                continue
            if item.purpose in {"opener", "fallback"}:
                await close_speaker()
                if item.generation != controller.generation:
                    continue
                open_speaker(item.generation)
                await speaker.send_text(item.text)
                await close_speaker()
                if item.generation != controller.generation:
                    continue
                purpose = "opener" if item.purpose == "opener" else "hangup"
                if await dispatch(controller.queue_mark(purpose)):
                    return
                continue
            if speaker is None:
                open_speaker(item.generation)
            await speaker.send_text(item.text)

    async def twilio_writer() -> None:
        while True:
            message = await twilio_out.get()
            if message is None:
                return
            await twilio_ws.send_json(message)

    async def twilio_reader() -> None:
        marked = False
        try:
            while True:
                raw_message = await twilio_ws.receive_text()
                step = controller.on_twilio(json.loads(raw_message))
                if controller.started and not marked:
                    marked = True
                    try:
                        await store.mark_in_progress(controller.call_id)
                    except Exception:
                        logger.error("Failed to mark call %s in progress", controller.call_id)
                if step.caller_audio:
                    try:
                        await listener.send_audio(base64.b64decode(step.caller_audio))
                    except Exception as exc:
                        logger.error(
                            "Failed to forward caller audio on %s: %s",
                            controller.call_id,
                            _redact(str(exc), *secrets),
                        )
                if await dispatch(step):
                    return
                if step.stop:
                    await finish(step.end_reason or controller.end_reason)
                    return
        except WebSocketDisconnect:
            if not finished:
                reason = controller.end_reason if controller.end_reason != "completed" else "caller_hangup"
                await finish(reason)

    async def stt_reader() -> None:
        # A recognize stream ending is not a hangup. Twilio stop and the
        # end-call path are what finish the call.
        while not finished and not controller.hung_up:
            try:
                notice = await listener.recv()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "Pipeline speech notice failed on %s: %s. The call stays up.",
                    controller.call_id,
                    _redact(str(exc), *secrets),
                )
                await asyncio.sleep(0.2)
                continue
            if notice is None:
                if finished or controller.hung_up or getattr(listener, "closed", False):
                    return
                logger.error(
                    "Pipeline speech listener stopped on %s. The call stays up.",
                    controller.call_id,
                )
                continue
            started, text = turns.push(notice)
            if started:
                latency.log()
                latency.reset()
                if await dispatch(controller.on_caller_speech()):
                    return
            if notice.speech_ended:
                latency.speech_ended()
            if text:
                latency.heard()
                if await dispatch(controller.on_caller_turn(text)):
                    return
            elif notice.is_final and notice.text:
                latency.stt_final()

    async def watchdog() -> None:
        while True:
            await asyncio.sleep(0.05)
            opener = controller.poll_opener()
            if opener is not None and await dispatch(opener):
                return
            deadline = controller.hangup_deadline
            if deadline is not None and not controller.hung_up and controller.clock() >= deadline:
                logger.info("Hangup watchdog fired for call %s", controller.call_id)
                controller.hangup_deadline = None
                await finish(controller.end_reason or "end_call")
                return

    try:
        await listener.connect()
    except Exception as exc:
        logger.error(
            "Pipeline speech connection failed for %s: %s",
            controller.call_id,
            _redact(str(exc), *secrets),
        )
        await finish("stt_unavailable")
        try:
            await listener.close()
        except Exception:
            pass
        return

    try:
        await store.record_provider_session(
            controller.call_id,
            voice=settings.pipeline_tts_voice,
            session_id="pipeline",
        )
    except Exception:
        logger.error("Failed to store the pipeline voice for %s", controller.call_id)

    tasks = [
        asyncio.create_task(twilio_writer(), name="twilio-writer"),
        asyncio.create_task(twilio_reader(), name="twilio-reader"),
        asyncio.create_task(stt_reader(), name="stt-reader"),
        asyncio.create_task(player(), name="pipeline-player"),
        asyncio.create_task(watchdog(), name="hangup-watchdog"),
    ]
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc:
                logger.error("Pipeline task failed on %s: %s", controller.call_id, _redact(str(exc), *secrets))
                if not finished:
                    await finish("relay_error")
    finally:
        reply = llm_task
        try:
            await listener.close()
        except Exception:
            logger.error("Failed to close the speech stream for %s", controller.call_id)
        for task in tasks:
            task.cancel()
        if reply is not None and not reply.done():
            reply.cancel()
        cancel_speech()
        await asyncio.gather(*tasks, return_exceptions=True)
        if reply is not None:
            await asyncio.gather(reply, return_exceptions=True)


async def _finish_now(controller: PipelineController, store, gateway, reason: str) -> None:
    controller.hung_up = True
    try:
        await store.finish_call(
            controller.call_id,
            end_reason=reason,
            transcript=list(controller.transcript),
            status="completed",
        )
    except Exception:
        logger.error("Failed to store the final call record for %s", controller.call_id)
    if gateway and controller.call_sid:
        try:
            await gateway.hangup(controller.call_sid)
        except Exception:
            logger.error("Twilio hangup failed for %s", controller.call_sid)
