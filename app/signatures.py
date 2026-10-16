"""Twilio webhook signature check.

Validation runs only when ``TWILIO_AUTH_TOKEN`` is set. API-key call control
does not require the auth token; the token is solely for signing webhooks.
"""

from __future__ import annotations

import base64
import hashlib
import hmac


def compute_twilio_signature(auth_token: str, url: str, params: dict[str, str]) -> str:
    payload = url
    for key in sorted(params):
        payload += key + params[key]
    digest = hmac.new(auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


def signature_is_valid(
    auth_token: str,
    url: str,
    params: dict[str, str],
    header: str | None,
) -> bool:
    if not auth_token:
        return True
    if not header:
        return False
    expected = compute_twilio_signature(auth_token, url, params)
    return hmac.compare_digest(expected, header)
