"""Open the realtime session when the call is answered, not while it is ringing.

The answer webhook schedules this and returns Connect TwiML without waiting
for the handshake. The media relay claims the socket if it is ready. AMD is
not consulted here: a later machine result discards the socket and hangs up.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger("tack45.prewarm")


@dataclass
class PrewarmedCall:
    provider: object
    ready: asyncio.Event
    setup_sent: bool = False
    error: BaseException | None = None
    abandoned: bool = False
    task: asyncio.Task | None = None


class PrewarmPool:
    def __init__(self) -> None:
        self._items: dict[str, PrewarmedCall] = {}
        self._lock = asyncio.Lock()

    async def start(self, call_id: str, provider, messages: list[dict]) -> PrewarmedCall:
        """Connect and send the session in the background.

        A second start for the same call keeps the session already in flight
        so answer and AMD cannot open two billed sockets.
        """
        async with self._lock:
            existing = self._items.get(call_id)
            if existing is not None and not existing.abandoned:
                return existing
            entry = PrewarmedCall(provider=provider, ready=asyncio.Event())
            self._items[call_id] = entry
        entry.task = asyncio.create_task(
            self._open(entry, list(messages)),
            name=f"prewarm-{call_id}",
        )
        return entry

    async def claim(self, call_id: str) -> PrewarmedCall | None:
        async with self._lock:
            return self._items.pop(call_id, None)

    async def discard(self, call_id: str) -> None:
        entry = await self.claim(call_id)
        if entry is not None:
            await self._abandon(entry)

    async def close_entry(self, entry: PrewarmedCall) -> None:
        await self._abandon(entry)

    async def aclose(self) -> None:
        async with self._lock:
            entries = list(self._items.values())
            self._items.clear()
        for entry in entries:
            await self._abandon(entry)

    async def _open(self, entry: PrewarmedCall, messages: list[dict]) -> None:
        try:
            await entry.provider.connect()
            if entry.abandoned:
                await _safe_close(entry.provider)
                return
            for message in messages:
                if entry.abandoned:
                    await _safe_close(entry.provider)
                    return
                await entry.provider.send(message)
            entry.setup_sent = True
        except asyncio.CancelledError:
            await _safe_close(entry.provider)
            raise
        except Exception as exc:
            entry.error = exc
            logger.warning("Voice session prewarm failed: %s", exc)
            await _safe_close(entry.provider)
        finally:
            entry.ready.set()

    async def _abandon(self, entry: PrewarmedCall) -> None:
        entry.abandoned = True
        task = entry.task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await _safe_close(entry.provider)


async def _safe_close(provider) -> None:
    close = getattr(provider, "close", None)
    if close is None:
        return
    try:
        await close()
    except Exception:
        logger.debug("Ignoring provider close error.", exc_info=True)
