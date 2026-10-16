"""Twilio REST calls using API key basic auth. No call is placed from tests."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger("tack45.twilio")

_API = "https://api.twilio.com/2010-04-01/Accounts"


class TwilioRestError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class TwilioGateway:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    def _url(self, path: str) -> str:
        return f"{_API}/{self.settings.twilio_account_sid}{path}"

    def _auth(self) -> tuple[str, str]:
        return (self.settings.twilio_api_key_sid, self.settings.twilio_api_key_secret)

    async def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.post(
            self._url(path),
            data=data,
            auth=self._auth(),
            timeout=30,
        )
        if response.status_code >= 400:
            message = _error_message(response)
            logger.error("Twilio %s failed: %s %s", path, response.status_code, message)
            raise TwilioRestError(response.status_code, message)
        return response.json()

    async def create_call(
        self,
        *,
        to: str,
        voice_url: str,
        status_url: str,
        amd_url: str,
    ) -> str:
        fields = {
            "To": to,
            "From": self.settings.twilio_from_number,
            "Url": voice_url,
            "Method": "POST",
            "StatusCallback": status_url,
            "StatusCallbackMethod": "POST",
            "StatusCallbackEvent": ["initiated", "ringing", "answered", "completed"],
            "MachineDetection": "DetectMessageEnd",
            "AsyncAmd": "true",
            "AsyncAmdStatusCallback": amd_url,
            "AsyncAmdStatusCallbackMethod": "POST",
            "MachineDetectionTimeout": str(self.settings.machine_detection_timeout),
        }
        body = await self._post("/Calls.json", fields)
        sid = body.get("sid")
        if not sid:
            raise TwilioRestError(502, "Twilio did not return a call sid.")
        return str(sid)

    async def set_twiml(self, call_sid: str, twiml: str) -> None:
        await self._post(f"/Calls/{call_sid}.json", {"Twiml": twiml})

    async def hangup(self, call_sid: str) -> None:
        try:
            await self._post(f"/Calls/{call_sid}.json", {"Status": "completed"})
        except TwilioRestError as exc:
            if exc.status_code == 404:
                logger.info("Call %s was already gone when hanging up.", call_sid)
                return
            raise


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:
        return response.text[:300]
    if isinstance(body, dict):
        return str(body.get("message") or body)[:300]
    return str(body)[:300]
