"""Cascaded full-duplex loop for VOICE_PROVIDER=gemini.

Deepgram listens to the caller's μ-law for the whole call. Gemini writes the
reply. Each sentence is spoken by gemini-3.8-flash-lite-tts as 8 kHz μ-law
and forwarded to Twilio. The next sentence is synthesised while the current
one plays. Grok, OpenAI, and GPT-Live never enter this module.

Gemini Tier 1 (10 requests/minute, 100/day) is too small for a live call.
A 429 is logged and the caller hears one cached goodbye, then the call ends.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass

import httpx
import websockets
from starlette.websockets import WebSocketDisconnect

from app.config import Settings
from app.gemini_turn import (
    GeminiController,
    GeminiStep,
    ModelEvent,
    PlayRequest,
    CallerTurns,
    clip_key,
    deepgram_headers,
    deepgram_listen_url,
    events_from_gemini_payload,
    is_rate_limit,
    prerender_lines,
    spoken_only,
    status_from_error,
    take_sentences,
    text_request,
)
from app.relay import drain_queued_audio
from app.tts_probe import interaction_body, synthesize

logger = logging.getLogger("tack45.gemini")

_FRAME = 160  # 20 ms of 8 kHz μ-law
_GENERATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"


@dataclass
class SpeechClip:
    audio: bytes
    error: str | None
    status: int | None
    rate_limited: bool


@dataclass
class SpeakItem:
    generation: int
    text: str
    purpose: str
    pending: asyncio.Task | None = None
    audio: bytes | None = None


@dataclass
class ReplyEnd:
    generation: int


class ClipCache:
    """Process-wide μ-law clips. One in-flight render per key, so a warm-up and
    the live call do not spend two requests on the opener."""

    def __init__(self) -> None:
        self._audio: dict[str, bytes] = {}
        self._inflight: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    def peek(self, key: str) -> bytes | None:
        return self._audio.get(key)

    def remember(self, key: str, audio: bytes) -> None:
        if audio:
            self._audio[key] = audio

    async def acquire(self, key: str, text: str, render) -> SpeechClip:
        async with self._lock:
            hit = self._audio.get(key)
            if hit is not None:
                return SpeechClip(audio=hit, error=None, status=200, rate_limited=False)
            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(render(text), name=f"gemini-tts-{len(key)}")
                self._inflight[key] = task
        try:
            clip = await task
        except asyncio.CancelledError:
            async with self._lock:
                if self._inflight.get(key) is task:
                    self._inflight.pop(key, None)
            raise
        except Exception:
            async with self._lock:
                if self._inflight.get(key) is task:
                    self._inflight.pop(key, None)
            raise
        async with self._lock:
            if self._inflight.get(key) is task:
                self._inflight.pop(key, None)
            if isinstance(clip, SpeechClip) and clip.audio and not clip.rate_limited:
                self._audio[key] = clip.audio
        return clip


def gemini_cache(app) -> ClipCache:
    cache = getattr(app.state, "gemini_clips", None)
    if cache is None:
        cache = ClipCache()
        app.state.gemini_clips = cache
    return cache


def schedule_gemini_prewarm(app, settings: Settings, opener: str) -> None:
    """Render the opener and the fixed lines without blocking the TwiML response."""
    cache = gemini_cache(app)
    task = asyncio.create_task(
        warm_clips(cache, settings, prerender_lines(opener)),
        name="gemini-clip-warm",
    )
    tasks = getattr(app.state, "gemini_warm_tasks", None)
    if tasks is None:
        tasks = set()
        app.state.gemini_warm_tasks = tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)


async def cancel_warm_tasks(tasks: set | None) -> None:
    if not tasks:
        return
    pending = [task for task in list(tasks) if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def warm_clips(cache: ClipCache, settings: Settings, texts: list[str]) -> None:
    for text in texts:
        key = clip_key(text, settings)
        if cache.peek(key) is not None:
            continue
        try:
            clip = await cache.acquire(key, text, lambda line: render_speech(settings, line))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Pre-render failed for a fixed line.")
            continue
        if clip.rate_limited:
            logger.warning(
                "Stopping Gemini clip pre-render after HTTP 429. "
                "Tier 1 is 10 requests/minute and 100/day. "
                "Raise the quota before placing calls."
            )
            return
        if clip.error or not clip.audio:
            logger.warning("Could not pre-render a fixed line: %s", clip.error)


async def render_speech(settings: Settings, text: str) -> SpeechClip:
    """μ-law from the same Interactions request the TTS probe uses."""
    spoken = spoken_only(text)
    style = settings.gemini_tts_style
    if not spoken or (style and style.lower() in spoken.lower()):
        logger.error("Refusing to synthesise a line that is not spoken text.")
        return SpeechClip(audio=b"", error="style leaked into text", status=None, rate_limited=False)
    body = interaction_body(
        spoken,
        model=settings.gemini_tts_model,
        style=style,
        voice=settings.gemini_tts_voice,
        language=settings.gemini_tts_language or None,
    )
    result = await synthesize(settings.gemini_api_key, body)
    error = result.get("error")
    status = status_from_error(error)
    audio = result.get("audio") or b""
    if audio and not error:
        return SpeechClip(audio=audio, error=None, status=status or 200, rate_limited=False)
    return SpeechClip(
        audio=audio,
        error=error or "Gemini stream contained no audio.",
        status=status,
        rate_limited=is_rate_limit(status, error),
    )


class DeepgramListener:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ws = None

    async def connect(self) -> None:
        url = deepgram_listen_url(
            base=self.settings.deepgram_url,
            model=self.settings.deepgram_model,
            language=self.settings.deepgram_language,
            endpointing_ms=self.settings.deepgram_endpointing_ms,
        )
        self._ws = await websockets.connect(
            url,
            additional_headers=deepgram_headers(self.settings.deepgram_api_key),
            max_size=4 * 1024 * 1024,
            open_timeout=15,
        )
        logger.info("Connected to Deepgram for the cascaded call.")

    async def send_audio(self, payload: bytes) -> None:
        if self._ws is None:
            return
        await self._ws.send(payload)

    async def recv(self) -> dict | None:
        if self._ws is None:
            return None
        message = await self._ws.recv()
        if isinstance(message, bytes):
            return None
        import json

        try:
            return json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Ignoring a non-JSON Deepgram message.")
            return {}

    async def close(self) -> None:
        if self._ws is None:
            return
        try:
            import json

            await self._ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:
            logger.debug("Deepgram close message failed.", exc_info=True)
        try:
            await self._ws.close()
        finally:
            self._ws = None


class GeminiTextModel:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def stream_reply(self, *, instructions: str, transcript: list[dict]):
        body = text_request(instructions, transcript, self.settings)
        url = _GENERATE.format(model=self.settings.gemini_text_model)
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                async with client.stream(
                    "POST",
                    url,
                    params={"alt": "sse"},
                    headers={
                        "x-goog-api-key": self.settings.gemini_api_key,
                        "Content-Type": "application/json",
                    },
                    json=body,
                ) as response:
                    if response.status_code == 429:
                        raw = (await response.aread()).decode("utf-8", errors="replace")
                        yield ModelEvent(rate_limited=True, error=f"HTTP 429: {raw[:500]}")
                        return
                    if response.status_code >= 400:
                        raw = (await response.aread()).decode("utf-8", errors="replace")
                        yield ModelEvent(error=f"HTTP {response.status_code}: {raw[:500]}")
                        return
                    async for line in response.aiter_lines():
                        payload = _sse_json(line)
                        if payload is None:
                            continue
                        for event in events_from_gemini_payload(payload):
                            yield event
            except httpx.HTTPError as exc:
                message = str(exc)[:500]
                yield ModelEvent(rate_limited=is_rate_limit(None, message), error=message)


class GeminiSynth:
    def __init__(self, settings: Settings, cache: ClipCache) -> None:
        self.settings = settings
        self.cache = cache

    async def synthesize(self, text: str) -> SpeechClip:
        key = clip_key(text, self.settings)
        return await self.cache.acquire(key, text, lambda line: render_speech(self.settings, line))


def _sse_json(line: str):
    import json

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


async def run_gemini_call(
    *,
    websocket,
    settings: Settings,
    call: dict,
    plan,
    store,
    gateway,
    phone: str,
    cache: ClipCache | None = None,
    listener=None,
    text_model=None,
    synth=None,
) -> None:
    message = plan.setup_messages[0]
    controller = GeminiController(
        call_id=call["id"],
        call_sid=call.get("call_sid"),
        phone=phone,
        opener=message["opener"],
        instructions=message["instructions"],
        hangup_grace_seconds=settings.hangup_grace_seconds,
    )
    await run_gemini_cascade(
        twilio_ws=websocket,
        settings=settings,
        controller=controller,
        store=store,
        gateway=gateway,
        cache=cache or ClipCache(),
        listener=listener,
        text_model=text_model,
        synth=synth,
    )


async def run_gemini_cascade(
    *,
    twilio_ws,
    settings: Settings,
    controller: GeminiController,
    store,
    gateway,
    cache: ClipCache | None = None,
    listener=None,
    text_model=None,
    synth=None,
) -> None:
    cache = cache or ClipCache()
    listener = listener or DeepgramListener(settings)
    text_model = text_model or GeminiTextModel(settings)
    synth = synth or GeminiSynth(settings, cache)
    turns = CallerTurns()
    twilio_out: asyncio.Queue = asyncio.Queue()
    speak_q: asyncio.Queue = asyncio.Queue()
    llm_task: asyncio.Task | None = None
    finished = False

    async def finish(reason: str) -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        controller.hung_up = True
        try:
            await store.finish_call(
                controller.call_id,
                end_reason=reason,
                transcript=list(controller.transcript),
                status="completed",
            )
        except Exception:
            logger.exception("Failed to store the final call record.")
        if gateway and controller.call_sid and reason != "caller_hangup":
            try:
                await gateway.hangup(controller.call_sid)
            except Exception:
                logger.exception("Twilio hangup failed for %s", controller.call_sid)

    def cancel_speech() -> None:
        nonlocal llm_task
        current = asyncio.current_task()
        if llm_task is not None and llm_task is not current and not llm_task.done():
            llm_task.cancel()
            llm_task = None
        while True:
            try:
                item = speak_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            pending = getattr(item, "pending", None)
            if pending is not None and not pending.done():
                pending.cancel()

    async def enqueue_play(play: PlayRequest, generation: int) -> None:
        key = clip_key(play.text, settings)
        cached = cache.peek(key)
        if play.cached_only and not cached:
            return
        item = SpeakItem(generation=generation, text=play.text, purpose=play.purpose)
        if cached:
            item.audio = cached
        else:
            item.pending = asyncio.create_task(synth.synthesize(play.text))
        await speak_q.put(item)

    async def dispatch(step: GeminiStep) -> bool:
        """Apply one controller step. Returns True when the caller should stop."""
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
                logger.exception("Failed to append transcript.")
        if step.opt_out_reason is not None:
            try:
                await store.record_opt_out(
                    call_id=controller.call_id,
                    phone=controller.phone,
                    call_sid=controller.call_sid,
                    reason=step.opt_out_reason,
                )
            except Exception:
                logger.exception("Failed to record opt-out.")
        generation = controller.generation
        for play in step.plays:
            await enqueue_play(play, generation)
        if step.begin_reply:
            start_reply(step.caller_text, generation)
        if step.follow_with_end:
            await speak_q.put(ReplyEnd(generation))
        if step.hangup:
            await finish(step.end_reason or controller.end_reason)
            return True
        return False

    def start_reply(text: str, generation: int) -> None:
        nonlocal llm_task
        current = asyncio.current_task()
        if llm_task is not None and llm_task is not current and not llm_task.done():
            llm_task.cancel()
        llm_task = asyncio.create_task(_reply(text, generation), name="gemini-reply")

    async def _reply(text: str, generation: int) -> None:
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
                    if await dispatch(controller.on_rate_limited("llm")):
                        return
                    return
                if event.error:
                    if await dispatch(controller.on_model_error(event.error)):
                        return
                    return
                if event.tool_name:
                    if await dispatch(controller.on_tool(event.tool_name, event.tool_args or {}, generation)):
                        return
                    if generation != controller.generation:
                        return
                if event.text:
                    sentences, buffer = take_sentences(
                        buffer + event.text,
                        flush=False,
                        early_clause=early,
                    )
                    if sentences:
                        early = False
                    for sentence in sentences:
                        if await dispatch(controller.on_assistant_sentence(sentence, generation)):
                            return
            if generation != controller.generation:
                return
            sentences, _rest = take_sentences(buffer, flush=True, early_clause=early)
            for sentence in sentences:
                if await dispatch(controller.on_assistant_sentence(sentence, generation)):
                    return
            await speak_q.put(ReplyEnd(generation))
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return

    async def send_frames(audio: bytes, generation: int) -> None:
        offset = 0
        while offset < len(audio):
            if generation != controller.generation or controller.interrupted:
                return
            frame = audio[offset : offset + _FRAME]
            offset += _FRAME
            if not controller.note_playback(len(frame), generation):
                return
            if not controller.stream_sid:
                return
            await twilio_out.put(
                {
                    "event": "media",
                    "streamSid": controller.stream_sid,
                    "media": {"payload": base64.b64encode(frame).decode("ascii")},
                }
            )

    async def player() -> None:
        while True:
            item = await speak_q.get()
            if item is None:
                return
            generation = item.generation
            if generation != controller.generation:
                pending = getattr(item, "pending", None)
                if pending is not None and not pending.done():
                    pending.cancel()
                continue
            if isinstance(item, ReplyEnd):
                step = controller.on_reply_finished(generation)
                if await dispatch(step):
                    return
                continue
            audio = item.audio
            if audio is None and item.pending is not None:
                try:
                    clip = await item.pending
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
                    continue
                except Exception as exc:
                    logger.exception("Gemini TTS failed")
                    if await dispatch(controller.on_model_error(str(exc)[:500])):
                        return
                    continue
                if generation != controller.generation or controller.interrupted:
                    continue
                if not isinstance(clip, SpeechClip):
                    continue
                if clip.rate_limited:
                    if await dispatch(controller.on_rate_limited("tts")):
                        return
                    continue
                if clip.error or not clip.audio:
                    if await dispatch(controller.on_model_error(clip.error or "no audio")):
                        return
                    continue
                audio = clip.audio
            if not audio or generation != controller.generation or controller.interrupted:
                continue
            await send_frames(audio, generation)
            if generation != controller.generation:
                continue
            if item.purpose == "opener":
                if await dispatch(controller.queue_mark("opener")):
                    return
            elif item.purpose == "fallback":
                if await dispatch(controller.queue_mark("hangup")):
                    return

    async def twilio_writer() -> None:
        while True:
            message = await twilio_out.get()
            if message is None:
                return
            await twilio_ws.send_json(message)

    async def twilio_reader() -> None:
        import json

        marked = False
        try:
            while True:
                raw = await twilio_ws.receive_text()
                step = controller.on_twilio(json.loads(raw))
                if controller.started and not marked:
                    marked = True
                    try:
                        await store.mark_in_progress(controller.call_id)
                    except Exception:
                        logger.exception("Failed to mark the call in progress.")
                if step.caller_audio:
                    try:
                        await listener.send_audio(base64.b64decode(step.caller_audio))
                    except Exception:
                        logger.exception("Failed to forward caller audio to Deepgram.")
                if await dispatch(step):
                    return
                if step.stop:
                    await finish(step.end_reason or controller.end_reason)
                    return
        except WebSocketDisconnect:
            if not finished:
                reason = controller.end_reason
                if reason == "completed":
                    reason = "caller_hangup"
                await finish(reason)

    async def stt_reader() -> None:
        while True:
            try:
                event = await listener.recv()
            except Exception:
                logger.exception("Deepgram stream ended.")
                event = None
            if event is None:
                if not controller.hung_up and not finished:
                    reason = controller.end_reason
                    if reason == "completed":
                        reason = "stt_closed"
                    await finish(reason)
                return
            notice = turns.push(event)
            if notice.speech_started:
                if await dispatch(controller.on_caller_speech()):
                    return
            if notice.text:
                if await dispatch(controller.on_caller_turn(notice.text)):
                    return

    async def watchdog() -> None:
        while True:
            await asyncio.sleep(0.05)
            deadline = controller.hangup_deadline
            if deadline is not None and not controller.hung_up and controller.clock() >= deadline:
                logger.info("Hangup watchdog fired for call %s", controller.call_id)
                controller.hangup_deadline = None
                await finish(controller.end_reason or "end_call")
                return

    try:
        await listener.connect()
    except Exception:
        logger.exception("Deepgram connection failed for call %s", controller.call_id)
        await finish("stt_unavailable")
        try:
            await listener.close()
        except Exception:
            logger.debug("Deepgram close after a failed connect.", exc_info=True)
        return

    try:
        await store.record_provider_session(
            controller.call_id,
            voice=settings.gemini_tts_voice,
            session_id="gemini-cascade",
        )
    except Exception:
        logger.exception("Failed to store the Gemini voice.")

    tasks = [
        asyncio.create_task(twilio_writer(), name="twilio-writer"),
        asyncio.create_task(twilio_reader(), name="twilio-reader"),
        asyncio.create_task(stt_reader(), name="stt-reader"),
        asyncio.create_task(player(), name="gemini-player"),
        asyncio.create_task(watchdog(), name="hangup-watchdog"),
    ]
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc:
                logger.exception("Gemini cascade task failed", exc_info=exc)
                if not finished:
                    await finish("relay_error")
    finally:
        reply = llm_task
        for task in tasks:
            task.cancel()
        if reply is not None and not reply.done():
            reply.cancel()
        cancel_speech()
        await asyncio.gather(*tasks, return_exceptions=True)
        if reply is not None:
            await asyncio.gather(reply, return_exceptions=True)
        try:
            await listener.close()
        except Exception:
            logger.debug("Deepgram close failed.", exc_info=True)
