"""POST a finished call, including its transcript, to an optional webhook.

The request runs in the background. A failure is logged and retried, and it
never changes the outcome of the call itself.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.config import Settings
from app.db import FINAL_TWILIO_STATUSES

logger = logging.getLogger("tack45.transcript")

# How long to wait for a Connect TwiML stream to attach after the call is already over.
STREAM_ATTACH_GRACE_SECONDS = 3.0
# How long to wait for an open stream to finish writing the transcript.
STREAM_CLOSE_WAIT_SECONDS = 15.0
# Sleeps before retry 1, 2, and 3. The first POST is immediate.
RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0)
WEBHOOK_TIMEOUT_SECONDS = 10.0


def webhook_headers(settings: Settings) -> dict[str, str]:
    """Auth header for the transcript webhook.

    ``TRANSCRIPT_WEBHOOK_AUTH_HEADER`` is a header name. ``Authorization``
    sends ``Bearer <secret>``. Any other name sends the secret as the value.

    A full header line is accepted too (``Authorization: Bearer <token>``).
    The text before the first colon is the name and the rest is the value.
    Using that whole line as the name makes httpx reject the request.
    """
    secret = (settings.transcript_webhook_secret or "").strip()
    raw = (settings.transcript_webhook_auth_header or "").strip()
    if not raw:
        return {}
    name, inline = _split_header_line(raw)
    if not name:
        return {}
    if inline:
        if name.lower() == "authorization" and inline.lower() == "bearer":
            if not secret:
                return {}
            return {name: f"Bearer {secret}"}
        return {name: inline}
    if not secret:
        return {}
    if name.lower() == "authorization":
        return {name: f"Bearer {secret}"}
    return {name: secret}


def _split_header_line(raw: str) -> tuple[str, str]:
    if ":" not in raw:
        return raw.strip(), ""
    name, value = raw.split(":", 1)
    return name.strip(), value.strip()


def _redact(message: str, secret: str) -> str:
    if secret and secret in message:
        return message.replace(secret, "[redacted]")
    return message


def schedule_transcript_webhook(app, call_id: str) -> None:
    """Queue a delivery. Returns immediately, and does nothing when the URL is unset."""
    try:
        url = (app.state.settings.transcript_webhook_url or "").strip()
        if not url or not call_id:
            return
        task = asyncio.create_task(
            _guarded_deliver(app, call_id),
            name=f"transcript-webhook-{call_id}",
        )
        tasks: set[asyncio.Task] = app.state.webhook_tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    except Exception:
        logger.exception("Could not schedule the transcript webhook for %s", call_id)


async def _guarded_deliver(app, call_id: str) -> None:
    try:
        await deliver_transcript(app, call_id)
    except Exception:
        logger.exception("Transcript webhook failed for %s", call_id)


async def deliver_transcript(
    app,
    call_id: str,
    *,
    attach_grace: float = STREAM_ATTACH_GRACE_SECONDS,
    close_wait: float = STREAM_CLOSE_WAIT_SECONDS,
    retry_delays: tuple[float, ...] = RETRY_DELAYS_SECONDS,
    sleep=asyncio.sleep,
) -> None:
    """Wait until the transcript is final, then POST it once."""
    settings: Settings = app.state.settings
    url = (settings.transcript_webhook_url or "").strip()
    if not url:
        return
    started = time.monotonic()
    close_deadline: float | None = None
    allow_open = False
    while True:
        call = await app.state.db.get(call_id)
        if call is None or call.get("transcript_posted"):
            return
        if call.get("status") not in FINAL_TWILIO_STATUSES:
            return
        now = time.monotonic()
        if call.get("stream_open"):
            if close_deadline is None:
                close_deadline = now + close_wait
            if now < close_deadline:
                await sleep(0.1)
                continue
            allow_open = True
            logger.warning(
                "Posting the transcript for %s before the media stream closed.",
                call_id,
            )
            break
        if call.get("stream_pending") and now < started + attach_grace:
            await sleep(0.05)
            continue
        break

    payload = await app.state.db.claim_transcript_webhook(call_id, allow_open=allow_open)
    if payload is None:
        return
    await post_with_retries(
        app.state.http,
        settings,
        payload,
        retry_delays=retry_delays,
        sleep=sleep,
    )


async def post_with_retries(
    client,
    settings: Settings,
    payload: dict[str, Any],
    *,
    retry_delays: tuple[float, ...] = RETRY_DELAYS_SECONDS,
    sleep=asyncio.sleep,
) -> bool:
    """POST the payload. Retry up to three times. Never raise."""
    url = (settings.transcript_webhook_url or "").strip()
    headers = webhook_headers(settings)
    secret = settings.transcript_webhook_secret or ""
    call_id = payload.get("id")
    attempts = 1 + len(retry_delays)
    for attempt in range(1, attempts + 1):
        try:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=WEBHOOK_TIMEOUT_SECONDS,
            )
            if response.status_code < 400:
                logger.info("Posted transcript for call %s", call_id)
                return True
            logger.warning(
                "Transcript webhook for %s returned %s (attempt %s of %s).",
                call_id,
                response.status_code,
                attempt,
                attempts,
            )
        except Exception as exc:
            logger.warning(
                "Transcript webhook for %s failed (attempt %s of %s): %s",
                call_id,
                attempt,
                attempts,
                _redact(f"{type(exc).__name__}: {exc}", secret),
            )
        if attempt <= len(retry_delays):
            try:
                await sleep(retry_delays[attempt - 1])
            except Exception:
                logger.exception("Transcript webhook retry delay failed for %s", call_id)
                return False
    logger.warning("Gave up posting the transcript for call %s.", call_id)
    return False


async def cancel_webhook_tasks(tasks: set[asyncio.Task]) -> None:
    """Stop background posts during shutdown without raising."""
    pending = [task for task in list(tasks) if not task.done()]
    for task in pending:
        task.cancel()
    for task in pending:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
