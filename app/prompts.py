"""Load the system prompt and the short identity-check opener."""

from __future__ import annotations

import re
from pathlib import Path

_TOKEN = re.compile(
    r"\{(opener|name|business|notes|number|first_name|named_thanks|callback)\}"
)

# Spoken when the call has no usable name. The named form lives in the opener file.
NO_NAME_OPENER = "Hi, is that the business owner?"

# Inbound openers already include the AI disclosure. The known form is for a
# caller we dialled recently; the unknown form asks how we can help.
KNOWN_CALLBACK_OPENER = (
    "Hi {first_name}, thanks for calling back Tack45. I'm an AI assistant. "
    "We called earlier to see if a short chat about practical AI for your business would be useful."
)
KNOWN_CALLBACK_NO_NAME = (
    "Hi, thanks for calling back Tack45. I'm an AI assistant. "
    "We called earlier to see if a short chat about practical AI for your business would be useful."
)
UNKNOWN_INBOUND_OPENER = (
    "Thanks for calling Tack45, I'm an AI assistant. "
    "How can I help, or would you like me to take a message?"
)


def load_text(path: str) -> str:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return file.read_text(encoding="utf-8").strip()


def first_name_from(name: str) -> str:
    """First word of the callee's name, or empty when there is nothing usable."""
    for part in (name or "").replace(",", " ").split():
        word = part.strip(".,;:!?\"'`()[]{}")
        if word and any(character.isalpha() for character in word):
            return word
    return ""


def render_opener(template: str, name: str) -> str:
    """The verbatim first turn. No name means we ask for the business owner."""
    first = first_name_from(name)
    if not first:
        return NO_NAME_OPENER
    if "{first_name}" in template:
        return template.replace("{first_name}", first)
    return template


def inbound_opener(name: str, *, known: bool) -> str:
    """Verbatim first turn for a call the person placed to us."""
    if not known:
        return UNKNOWN_INBOUND_OPENER
    first = first_name_from(name)
    if not first:
        return KNOWN_CALLBACK_NO_NAME
    return KNOWN_CALLBACK_OPENER.replace("{first_name}", first)


def render_instructions(
    template: str,
    *,
    opener: str,
    name: str,
    business: str,
    notes: str,
    number: str,
    first_name: str = "",
    callback: str = "no",
) -> str:
    """Fill prompt tokens. Values are inserted once and are not re-scanned."""
    first = first_name.strip()
    mapping = {
        "opener": opener,
        "name": name.strip() or "unknown",
        "business": business.strip() or "unknown",
        "notes": notes.strip() or "none",
        "number": number,
        "first_name": first or "unknown",
        "named_thanks": f"Thanks {first}." if first else "Thanks.",
        "callback": "yes" if callback.strip().lower() == "yes" else "no",
    }
    return _TOKEN.sub(lambda match: mapping[match.group(1)], template)
