"""Per-target call briefs appended to the outbound system prompt."""

from __future__ import annotations

import re
from pathlib import Path

# Spoken after the person confirms the identity check, unless a brief replaces it.
GENERIC_CONFIRMATION = (
    "{named_thanks} I'm an AI assistant calling for Tack45, a Sydney-based AI consultancy. "
    "Would a short chat about practical AI for your business be useful?"
)

BRIEF_HEADING = (
    "TARGET-SPECIFIC CALL BRIEF (overrides the generic pitch where they differ; "
    "all disclosure, opt-out, honesty and no-claims rules still apply)"
)

_BRIEF_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_CONFIRMATION_LINE = re.compile(
    r'(?m)^2\.\s+(?:Once confirmed:\s*)?(?:"([^"\n]+)"|“([^”\n]+)”)[ \t]*$'
)
MAX_BRIEF_CHARS = 20000


class BriefError(Exception):
    def __init__(self, message: str, *, unknown: bool = False) -> None:
        self.message = message
        self.unknown = unknown


def resolve_brief(directory: str, brief: object, brief_id: object) -> str:
    """Return the brief text for this call. Empty when neither field is set."""
    if brief is not None and not isinstance(brief, str):
        raise BriefError("brief must be a string.")
    if brief_id is not None and not isinstance(brief_id, str):
        raise BriefError("brief_id must be a string.")
    text = (brief or "").strip()
    ident = (brief_id or "").strip()
    if text and ident:
        raise BriefError("Provide brief or brief_id, not both.")
    if ident:
        text = _read_brief_file(directory, ident)
    if len(text) > MAX_BRIEF_CHARS:
        raise BriefError("brief is too long.")
    return text


def _read_brief_file(directory: str, brief_id: str) -> str:
    if not _BRIEF_ID.fullmatch(brief_id):
        raise BriefError("Unknown brief_id.", unknown=True)
    root = Path(directory).expanduser().resolve()
    path = (root / f"{brief_id}.md").resolve()
    if path.parent != root or not path.is_file():
        raise BriefError("Unknown brief_id.", unknown=True)
    return path.read_text(encoding="utf-8").strip()


def confirmation_line(brief: str) -> str | None:
    """The brief's opener line 2, which replaces the generic post-confirmation line."""
    match = _CONFIRMATION_LINE.search(brief or "")
    if match is None:
        return None
    line = (match.group(1) or match.group(2) or "").strip()
    return line or None


def apply_brief(instructions: str, brief: str, *, named_thanks: str) -> str:
    """Append the brief, and use its opener line 2 when it has one."""
    text = (brief or "").strip()
    if not text:
        return instructions
    updated = instructions
    replacement = confirmation_line(text)
    if replacement:
        generic = GENERIC_CONFIRMATION.replace("{named_thanks}", named_thanks)
        if generic in updated:
            updated = updated.replace(generic, replacement, 1)
    return f"{updated.rstrip()}\n\n{BRIEF_HEADING}\n{text}\n"
