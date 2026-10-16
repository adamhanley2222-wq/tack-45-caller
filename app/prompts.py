"""Load the system prompt and the short identity-check opener."""

from __future__ import annotations

import re
from pathlib import Path

_TOKEN = re.compile(
    r"\{(opener|name|business|notes|number|first_name|named_thanks)\}"
)

# Spoken when the call has no usable name. The named form lives in the opener file.
NO_NAME_OPENER = "Hi, is that the business owner?"


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


def render_instructions(
    template: str,
    *,
    opener: str,
    name: str,
    business: str,
    notes: str,
    number: str,
    first_name: str = "",
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
    }
    return _TOKEN.sub(lambda match: mapping[match.group(1)], template)
