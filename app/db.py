"""Call log, opt-outs, and the do-not-call table.

SQLite by default. When ``DATABASE_URL`` is a Postgres URL (including the
``postgres://`` form Railway provides) the same schema is created there.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Integer, String, Text, delete, inspect, select, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import Settings
from app.phones import PhoneError, normalize_phone

logger = logging.getLogger("tack45.db")

# Twilio status callbacks can arrive out of order. A late "ringing" or
# "answered" must not undo a stream that is already up.
_STATUS_RANK = {
    "created": 0,
    "dialing": 1,
    "initiated": 1,
    "queued": 1,
    "ringing": 2,
    "answered": 3,
    "detecting": 3,
    "streaming": 4,
    "in-progress": 5,
    "machine": 6,
    "completed": 7,
    "busy": 7,
    "failed": 7,
    "no-answer": 7,
    "canceled": 7,
    "refused": 7,
}
# Twilio's terminal call statuses. A transcript webhook waits for one of these.
FINAL_TWILIO_STATUSES = {"completed", "busy", "failed", "no-answer", "canceled"}
_TERMINAL_TWILIO = FINAL_TWILIO_STATUSES
_TERMINAL_CALL = _TERMINAL_TWILIO | {"machine", "refused"}


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class CallRow(Base):
    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    call_sid: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)
    stream_token: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    direction: Mapped[str] = mapped_column(
        String(16), default="outbound", server_default=text("'outbound'")
    )
    to_number: Mapped[str] = mapped_column(String(20), index=True)
    from_number: Mapped[str] = mapped_column(String(20), default="")
    name: Mapped[str] = mapped_column(String(200), default="")
    business: Mapped[str] = mapped_column(String(200), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    brief: Mapped[str] = mapped_column(Text, default="", server_default=text("''"))
    timezone: Mapped[str] = mapped_column(String(64), default="Australia/Sydney")
    provider: Mapped[str] = mapped_column(String(16), default="")
    # Resolved by the voice provider after session start. Empty until then.
    provider_voice: Mapped[str | None] = mapped_column(String(128))
    provider_session_id: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    amd_result: Mapped[str | None] = mapped_column(String(64))
    end_reason: Mapped[str | None] = mapped_column(String(64))
    opted_out: Mapped[int] = mapped_column(Integer, default=0)
    opt_out_reason: Mapped[str | None] = mapped_column(Text)
    transcript_json: Mapped[str] = mapped_column(Text, default="[]")
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Armed Connect TwiML has not attached a media socket yet.
    stream_pending: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    # The media socket is open. The transcript is not final until this clears.
    stream_open: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    # One successful claim per call, so the transcript webhook is posted once.
    transcript_posted: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))


class OptOutRow(Base):
    __tablename__ = "opt_outs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    phone: Mapped[str] = mapped_column(String(20), index=True)
    call_id: Mapped[str | None] = mapped_column(String(36), index=True)
    call_sid: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DoNotCallRow(Base):
    __tablename__ = "do_not_call"

    phone: Mapped[str] = mapped_column(String(20), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), default="file")
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def build_engine_url(settings: Settings) -> tuple[URL, dict[str, Any]]:
    """Return a SQLAlchemy URL and connect args.

    ``sslmode`` in a libpq URL is translated to asyncpg's ``ssl`` connect
    argument so a Railway ``DATABASE_URL`` works without a rewrite.
    """
    raw = (settings.database_url or "").strip()
    connect_args: dict[str, Any] = {}
    if not raw:
        path = Path(settings.sqlite_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return make_url("sqlite+aiosqlite:///" + path.as_posix()), connect_args

    url = make_url(raw)
    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+asyncpg")
    elif url.drivername == "sqlite":
        url = url.set(drivername="sqlite+aiosqlite")
    query = dict(url.query)
    if "sslmode" in query:
        mode = str(query.pop("sslmode"))
        if mode.lower() not in {"disable", "allow", "prefer"}:
            connect_args["ssl"] = True
        url = url.set(query=query)
    return url, connect_args


def _iso(value: datetime | None) -> str | None:
    aware = as_utc(value)
    if aware is None:
        return None
    return aware.isoformat()


def call_to_dict(row: CallRow) -> dict[str, Any]:
    try:
        transcript = json.loads(row.transcript_json or "[]")
    except json.JSONDecodeError:
        transcript = []
    transcript = [
        {key: value for key, value in line.items() if key != "utterance_id"}
        if isinstance(line, dict)
        else line
        for line in transcript
    ]
    return {
        "id": row.id,
        "call_sid": row.call_sid,
        "stream_token": row.stream_token,
        "direction": row.direction or "outbound",
        "to_number": row.to_number,
        "from_number": row.from_number,
        "name": row.name,
        "business": row.business,
        "notes": row.notes,
        "brief": row.brief or "",
        "timezone": row.timezone,
        "provider": row.provider,
        "provider_voice": row.provider_voice,
        "provider_session_id": row.provider_session_id,
        "status": row.status,
        "amd_result": row.amd_result,
        "end_reason": row.end_reason,
        "opted_out": bool(row.opted_out),
        "opt_out_reason": row.opt_out_reason,
        "transcript": transcript,
        "duration_seconds": row.duration_seconds,
        "created_at": _iso(row.created_at),
        "answered_at": _iso(row.answered_at),
        "ended_at": _iso(row.ended_at),
        "stream_pending": bool(row.stream_pending),
        "stream_open": bool(row.stream_open),
        "transcript_posted": bool(row.transcript_posted),
    }


def public_call(row: dict[str, Any], *, include_stream_token: bool = False) -> dict[str, Any]:
    """JSON shape returned by the API. The stream token stays off list views."""
    payload = {
        "id": row["id"],
        "call_sid": row["call_sid"],
        "direction": row.get("direction") or "outbound",
        "to": row["to_number"],
        "from": row.get("from_number") or "",
        "name": row["name"],
        "business": row["business"],
        "notes": row["notes"],
        "timezone": row["timezone"],
        "provider": row["provider"],
        "provider_voice": row.get("provider_voice"),
        "provider_session_id": row.get("provider_session_id"),
        "status": row["status"],
        "amd_result": row["amd_result"],
        "end_reason": row["end_reason"],
        "opted_out": row["opted_out"],
        "opt_out_reason": row["opt_out_reason"],
        "transcript": row["transcript"],
        "duration_seconds": row["duration_seconds"],
        "created_at": row["created_at"],
        "answered_at": row["answered_at"],
        "ended_at": row["ended_at"],
    }
    if include_stream_token:
        payload["stream_token"] = row["stream_token"]
    return payload


def _merge_transcript_line(current: list, line: dict[str, Any]) -> None:
    """Keep one row per utterance. A longer final replaces its interim text."""
    utterance_id = line.get("utterance_id")
    if utterance_id:
        for existing in current:
            if existing.get("utterance_id") == utterance_id and existing.get("role") == line.get("role"):
                if existing.get("text") == line.get("text"):
                    return
                existing["text"] = line.get("text")
                if line.get("at"):
                    existing["at"] = line["at"]
                return
    if current:
        last = current[-1]
        text = line.get("text") or ""
        if last.get("role") == line.get("role") and last.get("text") == text:
            return
        if (
            last.get("role") == line.get("role")
            and last.get("utterance_id")
            and text.startswith(last.get("text") or "")
        ):
            last["text"] = text
            if line.get("at"):
                last["at"] = line["at"]
            if utterance_id:
                last["utterance_id"] = utterance_id
            return
    current.append(line)


def webhook_payload(row: dict[str, Any]) -> dict[str, Any]:
    """JSON body posted when a call is finished and its transcript is final."""
    return {
        "id": row["id"],
        "call_sid": row["call_sid"],
        "direction": row.get("direction") or "outbound",
        "to": row["to_number"],
        "from": row.get("from_number") or "",
        "name": row["name"],
        "business": row["business"],
        "notes": row["notes"],
        "provider": row["provider"],
        "status": row["status"],
        "amd_result": row["amd_result"],
        "end_reason": row["end_reason"],
        "opted_out": row["opted_out"],
        "opt_out_reason": row["opt_out_reason"],
        "duration_seconds": row["duration_seconds"],
        "created_at": row["created_at"],
        "answered_at": row["answered_at"],
        "ended_at": row["ended_at"],
        "transcript": row["transcript"],
    }


# Columns added after the first deploy. create_all does not alter an existing table.
_ADDED_CALL_COLUMNS = {
    "direction": "VARCHAR(16) NOT NULL DEFAULT 'outbound'",
    "stream_pending": "INTEGER NOT NULL DEFAULT 0",
    "stream_open": "INTEGER NOT NULL DEFAULT 0",
    "transcript_posted": "INTEGER NOT NULL DEFAULT 0",
    "brief": "TEXT NOT NULL DEFAULT ''",
    "provider_voice": "VARCHAR(128)",
    "provider_session_id": "VARCHAR(128)",
}


def ensure_call_columns(sync_conn) -> None:
    """Add columns that existing Railway databases will not get from create_all."""
    inspector = inspect(sync_conn)
    if not inspector.has_table("calls"):
        return
    columns = {column["name"] for column in inspector.get_columns("calls")}
    for name, ddl in _ADDED_CALL_COLUMNS.items():
        if name in columns:
            continue
        sync_conn.execute(text(f"ALTER TABLE calls ADD COLUMN {name} {ddl}"))


def load_dnc_file(path: str) -> set[str]:
    file = Path(path)
    if not file.is_file():
        logger.warning("Do-not-call file not found at %s; treating it as empty.", path)
        return set()
    numbers: set[str] = set()
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            numbers.add(normalize_phone(stripped))
        except PhoneError:
            logger.warning("Skipping unreadable do-not-call entry: %s", stripped)
    return numbers


class Database:
    def __init__(self, settings: Settings) -> None:
        url, connect_args = build_engine_url(settings)
        self.url = url
        pool_kwargs: dict[str, Any] = {}
        # The default SQLite pool shares one connection. The relay and the AMD
        # webhook write at the same time, and that single connection deadlocks.
        if url.drivername.startswith("sqlite"):
            connect_args = {**connect_args, "timeout": 30}
            pool_kwargs["poolclass"] = NullPool
        self.engine: AsyncEngine = create_async_engine(
            url,
            connect_args=connect_args,
            pool_pre_ping=True,
            **pool_kwargs,
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.dnc_path = settings.do_not_call_path
        # One write at a time. Overlapping read-modify-write commits would
        # drop an AMD result or a transcript line.
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def _begin(self):
        async with self._lock:
            async with self.sessions() as session:
                yield session

    async def connect(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(ensure_call_columns)
        await self.sync_dnc_file()

    async def dispose(self) -> None:
        await self.engine.dispose()

    async def sync_dnc_file(self) -> int:
        """Replace file-sourced rows with the current file. Opt-outs are kept."""
        numbers = load_dnc_file(self.dnc_path)
        now = _utcnow()
        async with self._begin() as session:
            existing = (
                await session.execute(
                    select(DoNotCallRow).where(DoNotCallRow.source == "file")
                )
            ).scalars().all()
            for row in existing:
                if row.phone not in numbers:
                    await session.delete(row)
            present = set(
                (
                    await session.execute(select(DoNotCallRow.phone))
                ).scalars().all()
            )
            for phone in numbers:
                if phone not in present:
                    session.add(
                        DoNotCallRow(
                            phone=phone,
                            source="file",
                            note=None,
                            created_at=now,
                        )
                    )
            await session.commit()
        return len(numbers)

    async def blocked_numbers(self) -> set[str]:
        async with self._begin() as session:
            rows = (await session.execute(select(DoNotCallRow.phone))).scalars().all()
        return set(rows)

    async def create_call(self, fields: dict[str, Any]) -> dict[str, Any]:
        row = CallRow(
            id=fields["id"],
            call_sid=fields.get("call_sid"),
            stream_token=fields["stream_token"],
            direction=fields.get("direction") or "outbound",
            to_number=fields["to_number"],
            from_number=fields.get("from_number") or "",
            name=fields.get("name") or "",
            business=fields.get("business") or "",
            notes=fields.get("notes") or "",
            brief=fields.get("brief") or "",
            timezone=fields.get("timezone") or "Australia/Sydney",
            provider=fields.get("provider") or "",
            status=fields.get("status") or "created",
            amd_result=fields.get("amd_result"),
            end_reason=fields.get("end_reason"),
            opted_out=1 if fields.get("opted_out") else 0,
            opt_out_reason=fields.get("opt_out_reason"),
            transcript_json="[]",
            duration_seconds=None,
            created_at=fields.get("created_at") or _utcnow(),
            answered_at=None,
            ended_at=fields.get("ended_at"),
        )
        if fields.get("status") == "refused":
            row.ended_at = row.ended_at or _utcnow()
        async with self._begin() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return call_to_dict(row)

    async def get(self, call_id: str) -> dict[str, Any] | None:
        async with self._begin() as session:
            row = await session.get(CallRow, call_id)
            return call_to_dict(row) if row else None

    async def get_by_sid(self, call_sid: str) -> dict[str, Any] | None:
        async with self._begin() as session:
            row = (
                await session.execute(select(CallRow).where(CallRow.call_sid == call_sid))
            ).scalar_one_or_none()
            return call_to_dict(row) if row else None

    async def get_by_stream_token(self, token: str) -> dict[str, Any] | None:
        if not token:
            return None
        async with self._begin() as session:
            row = (
                await session.execute(
                    select(CallRow).where(CallRow.stream_token == token)
                )
            ).scalar_one_or_none()
            return call_to_dict(row) if row else None

    async def latest_outbound_to(
        self,
        number: str,
        *,
        within_days: int = 90,
    ) -> dict[str, Any] | None:
        """Newest outbound call we actually placed to this number.

        Refused attempts and inbound rows are ignored. ``within_days`` of 0
        keeps every placed call.
        """
        if not number:
            return None
        async with self._begin() as session:
            query = (
                select(CallRow)
                .where(CallRow.to_number == number)
                .where(CallRow.direction == "outbound")
                .where(CallRow.call_sid.is_not(None))
                .where(CallRow.status != "refused")
            )
            if within_days > 0:
                cutoff = _utcnow() - timedelta(days=within_days)
                query = query.where(CallRow.created_at >= cutoff)
            row = (
                await session.execute(
                    query.order_by(CallRow.created_at.desc()).limit(1)
                )
            ).scalar_one_or_none()
            return call_to_dict(row) if row else None

    async def list_calls(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        async with self._begin() as session:
            rows = (
                await session.execute(
                    select(CallRow).order_by(CallRow.created_at.desc()).limit(limit)
                )
            ).scalars().all()
            return [call_to_dict(row) for row in rows]

    async def attach_sid(self, call_id: str, call_sid: str, status: str) -> None:
        async with self._begin() as session:
            row = await session.get(CallRow, call_id)
            if row is None:
                return
            row.call_sid = call_sid
            row.status = status
            await session.commit()

    async def mark_failed(self, call_id: str, reason: str) -> None:
        async with self._begin() as session:
            row = await session.get(CallRow, call_id)
            if row is None:
                return
            row.status = "failed"
            row.end_reason = reason[:64]
            row.ended_at = row.ended_at or _utcnow()
            await session.commit()

    async def record_amd(
        self,
        call_id: str,
        *,
        amd_result: str | None,
        end_reason: str | None,
        status: str | None,
    ) -> None:
        async with self._begin() as session:
            row = await session.get(CallRow, call_id)
            if row is None:
                return
            if amd_result is not None:
                row.amd_result = amd_result
            if end_reason and not row.end_reason:
                row.end_reason = end_reason
            if status:
                row.status = status
            if status in {"completed", "machine"} and row.ended_at is None and status == "completed":
                row.ended_at = _utcnow()
            if row.answered_at is None and (amd_result == "human" or status == "answered"):
                row.answered_at = _utcnow()
            await session.commit()

    async def mark_streaming(self, call_id: str) -> None:
        """The answer webhook has connected the media stream."""
        async with self._begin() as session:
            row = await session.get(CallRow, call_id)
            if row is None:
                return
            if row.status not in _TERMINAL_CALL and _STATUS_RANK.get(row.status, 0) <= _STATUS_RANK["streaming"]:
                row.status = "streaming"
            if row.answered_at is None:
                row.answered_at = _utcnow()
            row.stream_pending = 1
            await session.commit()

    async def record_provider_session(
        self,
        call_id: str,
        *,
        voice: str | None,
        session_id: str | None,
    ) -> None:
        """Store the voice and session id the provider resolved at startup."""
        async with self._begin() as session:
            row = await session.get(CallRow, call_id)
            if row is None:
                return
            if voice:
                row.provider_voice = voice[:128]
            if session_id:
                row.provider_session_id = session_id[:128]
            await session.commit()

    async def mark_in_progress(self, call_id: str) -> None:
        async with self._begin() as session:
            row = await session.get(CallRow, call_id)
            if row is None:
                return
            if row.status in _TERMINAL_CALL or (row.end_reason or "").startswith("amd_"):
                return
            row.status = "in-progress"
            if row.answered_at is None:
                row.answered_at = _utcnow()
            await session.commit()

    async def append_transcript(self, call_id: str, lines: list[dict[str, Any]]) -> None:
        if not lines:
            return
        async with self._begin() as session:
            row = await session.get(CallRow, call_id)
            if row is None:
                return
            try:
                current = json.loads(row.transcript_json or "[]")
            except json.JSONDecodeError:
                current = []
            for line in lines:
                _merge_transcript_line(current, line)
            row.transcript_json = json.dumps(current)
            await session.commit()

    async def record_opt_out(
        self,
        *,
        call_id: str,
        phone: str,
        call_sid: str | None,
        reason: str,
    ) -> None:
        from uuid import uuid4

        now = _utcnow()
        async with self._begin() as session:
            session.add(
                OptOutRow(
                    id=str(uuid4()),
                    phone=phone,
                    call_id=call_id,
                    call_sid=call_sid,
                    reason=reason,
                    created_at=now,
                )
            )
            existing = await session.get(DoNotCallRow, phone)
            if existing is None:
                session.add(
                    DoNotCallRow(
                        phone=phone,
                        source="opt_out",
                        note=reason,
                        created_at=now,
                    )
                )
            row = await session.get(CallRow, call_id)
            if row is not None:
                row.opted_out = 1
                row.opt_out_reason = reason
            await session.commit()

    async def finish_call(
        self,
        call_id: str,
        *,
        end_reason: str,
        transcript: list[dict[str, Any]],
        status: str = "completed",
    ) -> None:
        async with self._begin() as session:
            row = await session.get(CallRow, call_id)
            if row is None:
                return
            if (row.end_reason or "").startswith("amd_") and end_reason in {
                "caller_hangup",
                "completed",
                "relay_error",
            }:
                end_reason = row.end_reason
            row.end_reason = end_reason or row.end_reason
            row.transcript_json = json.dumps(transcript)
            row.ended_at = as_utc(row.ended_at) or _utcnow()
            answered = as_utc(row.answered_at)
            if answered and row.duration_seconds is None:
                row.duration_seconds = max(
                    0, int((as_utc(row.ended_at) - answered).total_seconds())
                )
            if row.status == "machine" and status == "completed":
                status = "machine"
            if row.status not in {"busy", "failed", "no-answer", "canceled"}:
                row.status = status
            await session.commit()

    async def apply_twilio_status(
        self,
        call_sid: str,
        status: str,
        duration: str | None,
    ) -> bool:
        async with self._begin() as session:
            row = (
                await session.execute(select(CallRow).where(CallRow.call_sid == call_sid))
            ).scalar_one_or_none()
            if row is None:
                return False
            now = _utcnow()
            if status in {"answered", "in-progress", "streaming"} and row.answered_at is None:
                row.answered_at = now
            if status not in _TERMINAL_TWILIO and _STATUS_RANK.get(status, 0) < _STATUS_RANK.get(
                row.status, 0
            ):
                await session.commit()
                return True
            row.status = status
            terminal = _TERMINAL_TWILIO
            if status in terminal:
                row.ended_at = as_utc(row.ended_at) or now
                if not row.end_reason:
                    row.end_reason = status
                if duration not in (None, ""):
                    row.duration_seconds = int(float(duration))
                else:
                    answered = as_utc(row.answered_at)
                    ended = as_utc(row.ended_at)
                    if answered and ended and row.duration_seconds is None:
                        row.duration_seconds = max(0, int((ended - answered).total_seconds()))
            await session.commit()
            return True

    async def mark_stream_open(self, call_id: str) -> None:
        """The media socket is attached. The transcript is still being written."""
        async with self._begin() as session:
            row = await session.get(CallRow, call_id)
            if row is None:
                return
            row.stream_open = 1
            row.stream_pending = 0
            await session.commit()

    async def mark_stream_closed(self, call_id: str) -> None:
        """The media socket is gone, so the stored transcript will not grow."""
        async with self._begin() as session:
            row = await session.get(CallRow, call_id)
            if row is None:
                return
            row.stream_open = 0
            row.stream_pending = 0
            await session.commit()

    async def claim_transcript_webhook(self, call_id: str, *, allow_open: bool = False) -> dict[str, Any] | None:
        """Reserve the one webhook for this call and return its JSON body.

        A second claim returns None. An open media stream also returns None
        unless the caller has given up waiting for it to close.
        """
        async with self._begin() as session:
            row = await session.get(CallRow, call_id)
            if row is None or row.transcript_posted:
                return None
            if row.status not in FINAL_TWILIO_STATUSES:
                return None
            if row.stream_open and not allow_open:
                return None
            row.transcript_posted = 1
            await session.commit()
            await session.refresh(row)
            return webhook_payload(call_to_dict(row))

    async def delete_file_sourced_for_tests(self) -> None:
        async with self._begin() as session:
            await session.execute(delete(DoNotCallRow).where(DoNotCallRow.source == "file"))
            await session.commit()
