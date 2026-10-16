"""HTTP and WebSocket routes."""

from __future__ import annotations

import logging
import secrets
from uuid import uuid4
from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import JSONResponse, Response

from app.amd import amd_end_reason, decide_amd
from app.config import Settings, missing_dial_configuration
from app.db import public_call
from app import guards
from app.phones import PhoneError, infer_timezone, normalize_phone, validate_timezone
from app.prompts import load_text, render_instructions
from app.providers import build_session_plan
from app.realtime_ws import default_provider_factory
from app.relay import RelayController, run_relay
from app.signatures import signature_is_valid
from app.twiml import (
    connect_stream_twiml,
    hangup_twiml,
    hold_twiml,
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
    return f"{ws_base(settings)}/streams/media?token={token}"


def hold_seconds(settings: Settings) -> int:
    return max(int(settings.amd_hold_seconds), int(settings.machine_detection_timeout) + 15)


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
    if decision == "hold":
        await request.app.state.db.record_amd(
            call["id"],
            amd_result=None,
            end_reason=None,
            status="detecting",
        )
        return xml(hold_twiml(hold_seconds(settings)))
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
    return Response(status_code=204)


@router.websocket("/streams/media")
async def media_stream(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token", "")
    await websocket.accept()
    call = await websocket.app.state.db.get_by_stream_token(token)
    if call is None:
        await websocket.close(code=1008)
        return
    settings: Settings = websocket.app.state.settings
    try:
        opener = load_text(settings.opener_path)
        template = load_text(settings.prompt_path)
    except FileNotFoundError:
        logger.exception("Prompt file missing.")
        await websocket.close(code=1011)
        return
    instructions = render_instructions(
        template,
        opener=opener,
        name=call["name"],
        business=call["business"],
        notes=call["notes"],
        number=call["to_number"],
    )
    plan = build_session_plan(settings, instructions, opener)
    controller = RelayController(
        call_id=call["id"],
        call_sid=call["call_sid"],
        phone=call["to_number"],
        opener=opener,
        plan=plan,
        hangup_grace_seconds=settings.hangup_grace_seconds,
    )
    factory = getattr(websocket.app.state, "provider_factory", default_provider_factory)
    provider = factory(settings, call)
    try:
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
    if decision == "connect":
        await db.record_amd(call["id"], amd_result="human", end_reason=None, status="answered")
        twiml = connect_stream_twiml(stream_url(settings, call["stream_token"]), stream_parameters(call))
        if respond_with_twiml:
            return twiml
        if call.get("call_sid"):
            await gateway.set_twiml(call["call_sid"], twiml)
        return None

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
