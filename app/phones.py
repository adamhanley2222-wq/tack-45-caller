"""Phone number normalisation and callee timezone inference."""

from __future__ import annotations

import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_E164 = re.compile(r"^\+[1-9]\d{7,14}$")

# Geographic Australian area codes that map to a single timezone.
# 08 is deliberately absent: it covers Adelaide, Darwin, and Perth.
_AREA_TIMEZONES = {
    "2": "Australia/Sydney",
    "3": "Australia/Melbourne",
    "7": "Australia/Brisbane",
}


class PhoneError(ValueError):
    """The number could not be normalised to E.164."""


def normalize_phone(raw: str) -> str:
    """Return E.164. Accepts +61…, 61…, and Australian national numbers (0…)."""
    text = (raw or "").strip()
    if not text:
        raise PhoneError("Phone number is empty.")
    leading_plus = text.startswith("+")
    digits = re.sub(r"\D", "", text)
    if not digits:
        raise PhoneError("Phone number has no digits.")

    if leading_plus:
        e164 = "+" + digits
    elif digits.startswith("61"):
        e164 = "+" + digits
    elif digits.startswith("0") and len(digits) == 10:
        e164 = "+61" + digits[1:]
    elif len(digits) == 10 and (digits.startswith("1300") or digits.startswith("1800")):
        e164 = "+61" + digits
    else:
        raise PhoneError(
            "Use E.164 (+61…) or an Australian national number such as 04… or 02…."
        )

    if not _E164.fullmatch(e164):
        raise PhoneError("Phone number is not a valid E.164 length.")
    return e164


def validate_timezone(name: str) -> str:
    try:
        ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise PhoneError(f"Unknown timezone: {name}") from exc
    return name


def infer_timezone(e164: str, default: str) -> tuple[str, str | None]:
    """Pick a timezone from an Australian area code.

    Returns ``(timezone, warning)``. Mobiles and 08 numbers stay on the default
    because they are not a single timezone. Pass an explicit timezone to override.
    """
    default = validate_timezone(default)
    if not e164.startswith("+61") or len(e164) < 4:
        return default, None
    national = e164[3:]
    if not national:
        return default, None
    area = national[0]
    if area == "4":
        return default, "Mobile numbers are not tied to a state; using the default timezone."
    if area in _AREA_TIMEZONES:
        return _AREA_TIMEZONES[area], None
    if area == "8":
        return (
            default,
            "08 numbers cover Adelaide, Darwin, and Perth. Pass timezone to override the default.",
        )
    return default, None
