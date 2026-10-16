"""HTTP and WebSocket routes."""

from __future__ import annotations

import asyncio
import logging
import secrets
from uuid import uuid4
from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from sqlalchemy.exc import IntegrityError

from app.amd import amd_end_reason, decide_amd
from app.config import Settings, missing_dial_configuration
from app.db import FINAL_TWILIO_STATUSES, public_call
from app import guards
from app.phones import PhoneError, infer_timezone, normalize_phone, validate_timezone
from app.prompts import first_name_from, inbound_opener, load_text, render_instructions, render_opener
from app.providers import SessionPlan, build_session_plan
from app.realtime_ws import default_provider_factory
from app.relay import RelayController, run_relay
from app.signatures import signature_is_valid
from app.transcript_webhook import schedule_transcript_webhook
from app.twiml import (
    connect_stream_twiml,
    hangup_twiml,
    say_and_hangup_twiml,
    stream_parameters,
)

logger = logging.getLogger("tack45.api")

router = APIRouter()

class APIError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status = status
        self.code = code
        self.message = message


def require_bearer(request: Request) -> None:
    settings: Settings = request.app.state.settings
    if not settings.api_bearer_token:
        raise APIError(503, "not_configured", "API_BEARER_TOKEN is not set.")
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    expected = settings.api_bearer_token
    if scheme.lower() != "bearer" or not hmac_equal(token, expected):
        raise APIError(401, "unauthorized", "Missing or invalid bearer token.")


def hmac_equal(presented: str, expected: str) -> bool:
    import hmac

    if not presented or not expected:
        return False
    return hmac.compare_digest(presented, expected)


def base_url(settings: Settings) -> str:
    return settings.public_base_url.rstrip("/")


def ws_base(settings: Settings) -> str:
    base = base_url(settings)
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :]
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :]
    return base


def stream_url(settings: Settings, token: str) -> str:
    return f"{ws_base(settings)}/streams/media/{token}"


_LIVE = {"streaming", "in-progress"}


def prepare_script(settings: Settings, call: dict) -> tuple[str, SessionPlan]:
    """Render the short opener and the session that speaks it."""
    opener = render_opener(load_text(settings.opener_path), call.get("name") or "")
    template = load_text(settings.prompt_path)
    instructions = render_instructions(
        template,
        opener=opener,
        name=call.get("name") or "",
        business=call.get("business") or "",
        notes=call.get("notes") or "",
        number=call.get("to_number") or "",
        first_name=first_name_from(call.get("name") or ""),
    )
    return opener, build_session_plan(settings, instructions, opener)


def prepare_inbound_script(
    settings: Settings,
    call: dict,
    match: dict | None,
) -> tuple[str, SessionPlan]:
    """Opener and session for a call the person placed to us."""
    source = match or {}
    name = source.get("name") or ""
    opener = inbound_opener(name, known=match is not None)
    template = load_text(settings.inbound_prompt_path)
    instructions = render_instructions(
        template,
        opener=opener,
        name=name,
        business=source.get("business") or "",
        notes=source.get("notes") or "",
        number=call.get("from_number") or "",
        first_name=first_name_from(name),
        callback="yes" if match is not None else "no",
    )
    return opener, build_session_plan(settings, instructions, opener)


async def script_for_call(settings: Settings, db, call: dict) -> tuple[str, SessionPlan]:
    """Outbound uses the compliance opener. Inbound uses the callback prompt."""
    if call.get("direction") == "inbound":
        caller = call.get("from_number") or ""
        match = None
        if caller.startswith("+"):
            match = await db.latest_outbound_to(
                caller,
                within_days=settings.inbound_lookup_days,
            )
        return prepare_inbound_script(settings, call, match)
    return prepare_script(settings, call)


def relay_phone(call: dict) -> str:
    """Number an opt-out should block. Inbound that is the caller, not us."""
    if call.get("direction") == "inbound":
        return call.get("from_number") or ""
    return call.get("to_number") or ""


def _stream_is_live(call: dict) -> bool:
    if call.get("status") in _LIVE:
        return True
    return False


def external_url(request: Request, settings: Settings) -> str:
    url = f"{base_url(settings)}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    return url


async def read_form(request: Request) -> dict[str, str]:
    form = await request.form()
    return {str(key): str(value) for key, value in form.multi_items()}


def xml(body: str) -> Response:
    return Response(content=body, media_type="text/xml")


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "tack45-caller"}


@router.get("/")
async def root() -> dict:
    return {
        "service": "tack45-caller",
        "health": "/health",
        "docs": "See the README for Railway deploy, env vars, and the test call script.",
    }


@router.post("/calls")
async def create_call(request: Request) -> JSONResponse:
    require_bearer(request)
    settings: Settings = request.app.state.settings
    missing = [item for item in missing_dial_configuration(settings) if item != "API_BEARER_TOKEN"]
    if missing:
        raise APIError(503, "not_configured", "Missing configuration: " + ", ".join(missing) + ".")

    try:
        payload = await request.json()
    except Exception as exc:
        raise APIError(422, "invalid_body", "Body must be JSON.") from exc
    if not isinstance(payload, dict):
        raise APIError(422, "invalid_body", "Body must be a JSON object.")

    raw_to = str(payload.get("to") or "")
    name = str(payload.get("name") or "")
    business = str(payload.get("business") or "")
    notes = str(payload.get("notes") or "")
    timezone_field = payload.get("timezone")
    if len(name) > 200 or len(business) > 200 or len(notes) > 2000:
        raise APIError(422, "invalid_body", "name, business, or notes is too long.")

    try:
        number = normalize_phone(raw_to)
    except PhoneError as exc:
        raise APIError(422, "invalid_number", str(exc)) from exc

    warning = None
    if timezone_field:
        try:
            timezone_name = validate_timezone(str(timezone_field))
        except PhoneError as exc:
            raise APIError(422, "invalid_timezone", str(exc)) from exc
    else:
        try:
            timezone_name, warning = infer_timezone(number, settings.default_timezone)
        except PhoneError as exc:
            raise APIError(422, "invalid_timezone", str(exc)) from exc

    await request.app.state.db.sync_dnc_file()
    blocked = await request.app.state.db.blocked_numbers()
    try:
        from_number = normalize_phone(settings.twilio_from_number)
    except PhoneError:
        from_number = settings.twilio_from_number
    decision = guards.evaluate_dial(
        number=number,
        timezone_name=timezone_name,
        moment=guards.utcnow(),
        blocked_numbers=blocked,
        from_number=from_number,
    )
    call_id = str(uuid4())
    token = secrets.token_urlsafe(32)
    provider = settings.voice_provider.strip().lower()
    if not decision.allowed:
        row = await request.app.state.db.create_call(
            {
                "id": call_id,
                "stream_token": token,
                "to_number": number,
                "from_number": settings.twilio_from_number,
                "name": name,
                "business": business,
                "notes": notes,
                "timezone": timezone_name,
                "provider": provider,
                "status": "refused",
                "end_reason": decision.code,
            }
        )
        logger.info("Refused call %s to %s: %s", call_id, number, decision.message)
        return JSONResponse(
            status_code=403,
            content={
                "id": row["id"],
                "to": number,
                "status": "refused",
                "code": decision.code,
                "detail": decision.message,
                "timezone": timezone_name,
            },
        )

    row = await request.app.state.db.create_call(
        {
            "id": call_id,
            "stream_token": token,
            "to_number": number,
            "from_number": settings.twilio_from_number,
            "name": name,
            "business": business,
            "notes": notes,
            "timezone": timezone_name,
            "provider": provider,
            "status": "dialing",
        }
    )
    voice = f"{base_url(settings)}/twilio/voice?call_id={call_id}"
    amd = f"{base_url(settings)}/twilio/amd?call_id={call_id}"
    status = f"{base_url(settings)}/twilio/status?call_id={call_id}"
    try:
        call_sid = await request.app.state.gateway.create_call(
            to=number,
            voice_url=voice,
            status_url=status,
            amd_url=amd,
        )
    except Exception as exc:
        await request.app.state.db.mark_failed(call_id, "twilio_error")
        schedule_transcript_webhook(request.app, call_id)
        logger.exception("Twilio create failed for %s", call_id)
        raise APIError(502, "twilio_error", str(exc)) from exc

    await request.app.state.db.attach_sid(call_id, call_sid, "initiated")
    body = public_call({**row, "call_sid": call_sid, "status": "initiated"}, include_stream_token=True)
    body["status"] = "initiated"
    body["call_sid"] = call_sid
    body["stream_url"] = stream_url(settings, token)
    if warning:
        body["timezone_warning"] = warning
    logger.info("Dialled %s call %s sid %s", number, call_id, call_sid)
    return JSONResponse(status_code=201, content=body)


@router.get("/calls/{call_id}")
async def get_call(call_id: str, request: Request) -> JSONResponse:
    require_bearer(request)
    row = await request.app.state.db.get(call_id)
    if row is None:
        raise APIError(404, "not_found", "Call not found.")
    return JSONResponse(public_call(row))


@router.get("/calls")
async def list_calls(request: Request) -> JSONResponse:
    require_bearer(request)
    try:
        limit = int(request.query_params.get("limit", "50"))
    except ValueError:
        limit = 50
    rows = await request.app.state.db.list_calls(limit)
    return JSONResponse([public_call(row) for row in rows])


@router.post("/twilio/voice")
async def twilio_voice(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    form = await read_form(request)
    if not _signed(request, settings, form):
        return JSONResponse(status_code=403, content={"detail": "Invalid Twilio signature."})
    call = await _find_call(request, form)
    if call is None:
        logger.warning("Voice webhook for an unknown call.")
        return xml(hangup_twiml())
    answered = form.get("AnsweredBy")
    decision = decide_amd(answered)
    # Speak as soon as the call is answered. Async AMD has no AnsweredBy yet;
    # a later machine, fax, or unknown result hangs up. Do not open a provider
    # session before this webhook: that would bill while the phone is ringing.
    if decision in {"hold", "connect"}:
        if decision == "connect":
            await request.app.state.db.record_amd(
                call["id"],
                amd_result="human",
                end_reason=None,
                status=None,
            )
        if _stream_is_live(call):
            return xml(_connect_twiml(settings, call))
        twiml = await _arm_stream(request, call)
        return xml(twiml)
    await request.app.state.prewarm.discard(call["id"])
    twiml = await _apply_answered_by(request, call, answered or "", respond_with_twiml=True)
    return xml(twiml or hangup_twiml())


@router.post("/twilio/amd")
async def twilio_amd(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    form = await read_form(request)
    if not _signed(request, settings, form):
        return JSONResponse(status_code=403, content={"detail": "Invalid Twilio signature."})
    call = await _find_call(request, form)
    if call is None:
        logger.warning("AMD webhook for an unknown call: %s", form.get("CallSid"))
        return Response(status_code=404)
    answered = form.get("AnsweredBy") or "unknown"
    logger.info("AMD %s for call %s: %s", call["id"], call.get("call_sid"), answered)
    await _apply_answered_by(request, call, answered, respond_with_twiml=False)
    return Response(status_code=204)


@router.post("/twilio/status")
async def twilio_status(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    form = await read_form(request)
    if not _signed(request, settings, form):
        return JSONResponse(status_code=403, content={"detail": "Invalid Twilio signature."})
    call_sid = form.get("CallSid") or ""
    status = form.get("CallStatus") or ""
    if not call_sid:
        return Response(status_code=400)
    found = await request.app.state.db.apply_twilio_status(
        call_sid,
        status,
        form.get("CallDuration"),
    )
    if not found:
        logger.warning("Status webhook for unknown CallSid %s", call_sid)
        return Response(status_code=404)
    if status in FINAL_TWILIO_STATUSES:
        call = await request.app.state.db.get_by_sid(call_sid)
        if call is not None:
            schedule_transcript_webhook(request.app, call["id"])
    return Response(status_code=204)


@router.post("/twilio/inbound")
async def twilio_inbound(request: Request) -> Response:
    """Answer a call to our number. No AMD and no calling-hours check."""
    settings: Settings = request.app.state.settings
    form = await read_form(request)
    if not _signed(request, settings, form):
        return JSONResponse(status_code=403, content={"detail": "Invalid Twilio signature."})
    call_sid = (form.get("CallSid") or "").strip()
    if not call_sid:
        return xml(hangup_twiml())

    db = request.app.state.db
    existing = await db.get_by_sid(call_sid)
    if existing is not None:
        if _call_finished(existing):
            return xml(hangup_twiml())
        return xml(_connect_twiml(settings, existing))

    caller = _stored_caller(form.get("From") or "")
    ours = _stored_caller(form.get("To") or "") or settings.twilio_from_number
    match = None
    if caller.startswith("+"):
        match = await db.latest_outbound_to(caller, within_days=settings.inbound_lookup_days)
    if match:
        name = match.get("name") or ""
        business = match.get("business") or ""
        notes = match.get("notes") or ""
        timezone_name = match.get("timezone") or settings.default_timezone
    else:
        name = ""
        business = ""
        notes = ""
        timezone_name = _caller_timezone(settings, caller)

    call_id = str(uuid4())
    token = secrets.token_urlsafe(32)
    try:
        call = await db.create_call(
            {
                "id": call_id,
                "call_sid": call_sid,
                "stream_token": token,
                "direction": "inbound",
                "to_number": ours,
                "from_number": caller,
                "name": name,
                "business": business,
                "notes": notes,
                "timezone": timezone_name,
                "provider": settings.voice_provider.strip().lower(),
                "status": "answered",
            }
        )
    except IntegrityError:
        existing = await db.get_by_sid(call_sid)
        if existing is None or _call_finished(existing):
            return xml(hangup_twiml())
        return xml(_connect_twiml(settings, existing))

    logger.info(
        "Inbound call %s from %s sid %s callback=%s",
        call_id,
        caller,
        call_sid,
        "yes" if match else "no",
    )
    return xml(await _arm_stream(request, call))


@router.websocket("/streams/media/{token}")
async def media_stream(websocket: WebSocket, token: str) -> None:
    # Token travels in the path: Twilio strips query strings from <Stream url>.
    await websocket.accept()
    call = await websocket.app.state.db.get_by_stream_token(token)
    if call is None:
        await websocket.close(code=1008)
        return
    settings: Settings = websocket.app.state.settings
    if _call_finished(call):
        await websocket.app.state.prewarm.discard(call["id"])
        await _release_stream(websocket.app, call["id"])
        await websocket.close(code=1000)
        return
    try:
        opener, plan = await script_for_call(settings, websocket.app.state.db, call)
    except FileNotFoundError:
        logger.exception("Prompt file missing.")
        await _release_stream(websocket.app, call["id"])
        await websocket.close(code=1011)
        return
    provider, setup_sent = await _take_provider(websocket.app, settings, call, plan)
    controller = RelayController(
        call_id=call["id"],
        call_sid=call["call_sid"],
        phone=relay_phone(call),
        opener=opener,
        plan=plan,
        hangup_grace_seconds=settings.hangup_grace_seconds,
        setup_sent=setup_sent,
    )
    try:
        await websocket.app.state.db.mark_stream_open(call["id"])
        await run_relay(
            twilio_ws=websocket,
            provider=provider,
            controller=controller,
            store=websocket.app.state.db,
            gateway=websocket.app.state.gateway,
        )
    except Exception:
        logger.exception("Media relay ended with an error for %s", call["id"])
        await websocket.close(code=1011)
    finally:
        await _release_stream(websocket.app, call["id"])


async def _release_stream(app, call_id: str) -> None:
    """The media socket is done. Post the transcript if the call is already final."""
    try:
        await app.state.db.mark_stream_closed(call_id)
    except Exception:
        logger.exception("Failed to mark the media stream closed for %s", call_id)
    schedule_transcript_webhook(app, call_id)


def _signed(request: Request, settings: Settings, form: dict[str, str]) -> bool:
    if not settings.twilio_auth_token:
        return True
    header = request.headers.get("x-twilio-signature")
    return signature_is_valid(
        settings.twilio_auth_token,
        external_url(request, settings),
        form,
        header,
    )


async def _find_call(request: Request, form: dict[str, str]) -> dict | None:
    call_id = request.query_params.get("call_id") or ""
    if call_id:
        row = await request.app.state.db.get(call_id)
        if row:
            return row
    call_sid = form.get("CallSid") or ""
    if call_sid:
        return await request.app.state.db.get_by_sid(call_sid)
    return None


async def _apply_answered_by(
    request: Request,
    call: dict,
    answered_by: str,
    *,
    respond_with_twiml: bool,
) -> str | None:
    settings: Settings = request.app.state.settings
    decision = decide_amd(answered_by)
    db = request.app.state.db
    gateway = request.app.state.gateway
    fresh = await db.get(call["id"]) or call
    if decision == "connect":
        live = _stream_is_live(fresh)
        await db.record_amd(
            call["id"],
            amd_result="human",
            end_reason=None,
            status=None if live else "answered",
        )
        if live:
            return None
        twiml = await _arm_stream(request, call)
        if respond_with_twiml:
            return twiml
        if call.get("call_sid"):
            await gateway.set_twiml(call["call_sid"], twiml)
        return None

    await request.app.state.prewarm.discard(call["id"])
    reason = amd_end_reason(answered_by)
    if decision == "machine" and settings.machine_message.strip():
        await db.record_amd(
            call["id"],
            amd_result=answered_by,
            end_reason=reason,
            status="machine",
        )
        twiml = say_and_hangup_twiml(settings.machine_message)
        if respond_with_twiml:
            return twiml
        if call.get("call_sid"):
            await gateway.set_twiml(call["call_sid"], twiml)
        return None

    await db.record_amd(
        call["id"],
        amd_result=answered_by,
        end_reason=reason,
        status="completed",
    )
    if respond_with_twiml:
        return hangup_twiml()
    if call.get("call_sid"):
        await gateway.hangup(call["call_sid"])
    return None


def _connect_twiml(settings: Settings, call: dict) -> str:
    return connect_stream_twiml(stream_url(settings, call["stream_token"]), stream_parameters(call))


def _call_finished(call: dict) -> bool:
    if call.get("status") in {"completed", "machine", "failed", "refused", "busy", "no-answer", "canceled"}:
        return True
    return (call.get("end_reason") or "").startswith("amd_")


def _stored_caller(raw: str) -> str:
    """E.164 when we can read it. Otherwise a short raw value, never our lookup key."""
    try:
        return normalize_phone(raw)
    except PhoneError:
        return raw.strip()[:20]


def _caller_timezone(settings: Settings, caller: str) -> str:
    if not caller.startswith("+"):
        return settings.default_timezone
    try:
        zone, _warning = infer_timezone(caller, settings.default_timezone)
    except PhoneError:
        return settings.default_timezone
    return zone


async def _arm_stream(request: Request, call: dict) -> str:
    """Connect media now and start the realtime session before TwiML returns."""
    settings: Settings = request.app.state.settings
    await request.app.state.db.mark_streaming(call["id"])
    await _prewarm_session(request, call)
    logger.info("Armed media stream for %s", call["id"])
    return _connect_twiml(settings, call)


async def _prewarm_session(request: Request, call: dict) -> None:
    """Start the provider handshake. Do not await it: TwiML must return now."""
    settings: Settings = request.app.state.settings
    try:
        _opener, plan = await script_for_call(settings, request.app.state.db, call)
    except FileNotFoundError:
        logger.exception("Prompt file missing; the stream will open the session itself.")
        return
    try:
        factory = getattr(request.app.state, "provider_factory", default_provider_factory)
        provider = factory(settings, call)
        await request.app.state.prewarm.start(call["id"], provider, plan.setup_messages)
        # Let the handshake task reach its first await. Do not wait for it.
        await asyncio.sleep(0)
    except Exception:
        logger.exception("Could not schedule a voice session for %s", call["id"])


async def _take_provider(app, settings: Settings, call: dict, plan: SessionPlan):
    """Prefer the session opened at answer. Otherwise connect when the stream starts."""
    factory = getattr(app.state, "provider_factory", default_provider_factory)
    entry = await app.state.prewarm.claim(call["id"])
    if entry is None:
        return factory(settings, call), False
    try:
        await asyncio.wait_for(entry.ready.wait(), timeout=8)
    except asyncio.TimeoutError:
        logger.warning("Prewarm timed out for %s; opening a new session.", call["id"])
        await app.state.prewarm.close_entry(entry)
        return factory(settings, call), False
    if entry.setup_sent and entry.error is None and not entry.abandoned:
        return entry.provider, True
    logger.warning("Prewarmed session unusable for %s: %s", call["id"], entry.error)
    await app.state.prewarm.close_entry(entry)
    return factory(settings, call), False
