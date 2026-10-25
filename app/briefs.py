"""Per-target call briefs appended to the outbound system prompt."""

from __future__ import annotations

import re
from pathlib import Path

# Spoken after the person confirms the identity check, unless a brief replaces it.
GENERIC_CONFIRMATION = (
    "{named_thanks} I'm an AI assistant calling for Tack45, a Sydney-based AI consultancy. "
    "Is now a good time for a quick chat?"
)

# Front matter `base_prompt` may name one of these files, next to prompts/system.txt.
# Anything else is rejected so a brief cannot point at an arbitrary path.
BASE_PROMPTS = {"personal": "personal_system.txt"}

BRIEF_HEADING = (
    "TARGET-SPECIFIC CALL BRIEF (overrides the generic pitch where they differ; "
    "all disclosure, opt-out, honesty and no-claims rules still apply)"
)

_BRIEF_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_CONFIRMATION_LINE = re.compile(
    r'(?m)^2\.\s+(?:Once confirmed:\s*)?(?:"([^"\n]+)"|“([^”\n]+)”)[ \t]*$'
)
_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.DOTALL)
_META_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):\s*(.*)$")
_ALLOWED_META = {"base_prompt", "opener"}
MAX_BRIEF_CHARS = 20000
MAX_OPENER_CHARS = 300


class BriefError(Exception):
    def __init__(self, message: str, *, unknown: bool = False) -> None:
        self.message = message
        self.unknown = unknown


def resolve_brief(
    directory: str,
    brief: object,
    brief_id: object,
    *,
    prompt_path: str | None = None,
) -> str:
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
    _validate_front_matter(text, prompt_path)
    return text


def _read_brief_file(directory: str, brief_id: str) -> str:
    if not _BRIEF_ID.fullmatch(brief_id):
        raise BriefError("Unknown brief_id.", unknown=True)
    root = Path(directory).expanduser().resolve()
    path = (root / f"{brief_id}.md").resolve()
    if path.parent != root or not path.is_file():
        raise BriefError("Unknown brief_id.", unknown=True)
    return path.read_text(encoding="utf-8").strip()


def split_front_matter(brief: str) -> tuple[dict[str, str], str]:
    """Return front-matter keys and the brief body. No front matter returns the text."""
    text = (brief or "").strip()
    match = _FRONT_MATTER.match(text)
    if match is None:
        return {}, text
    return _parse_meta(match.group(1)), match.group(2).strip()


def spoken_opener(brief: str) -> str | None:
    """Front-matter opener, when this brief replaces the spoken first line."""
    opener = split_front_matter(brief)[0].get("opener", "").strip()
    return opener or None


def base_prompt_file(prompt_path: str, brief: str) -> str | None:
    """Path of the replacement base prompt, or None to keep prompts/system.txt."""
    name = split_front_matter(brief)[0].get("base_prompt", "").strip()
    if not name:
        return None
    filename = BASE_PROMPTS.get(name)
    if filename is None:
        raise BriefError("Unknown brief base_prompt.")
    root = Path(prompt_path).expanduser().resolve().parent
    path = (root / filename).resolve()
    if path.parent != root or not path.is_file():
        raise BriefError("Unknown brief base_prompt.")
    return str(path)


def _parse_meta(block: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _META_LINE.match(line)
        if match is None or match.group(1) in meta:
            raise BriefError("Brief front matter is invalid.")
        meta[match.group(1)] = _unquote(match.group(2).strip())
    return meta


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        inner = value[1:-1]
        if value[0] == '"':
            return inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner.replace("''", "'")
    return value


def _validate_front_matter(brief: str, prompt_path: str | None) -> None:
    meta, _body = split_front_matter(brief)
    unknown = set(meta) - _ALLOWED_META
    if unknown:
        raise BriefError("Unknown brief setting.")
    if "opener" in meta:
        opener = meta["opener"].strip()
        if not opener or "\n" in opener or len(opener) > MAX_OPENER_CHARS:
            raise BriefError("brief opener is invalid.")
    if "base_prompt" not in meta:
        return
    if prompt_path is None:
        if meta["base_prompt"].strip() not in BASE_PROMPTS:
            raise BriefError("Unknown brief base_prompt.")
        return
    base_prompt_file(prompt_path, brief)


def confirmation_line(brief: str) -> str | None:
    """The brief's opener line 2, which replaces the generic post-confirmation line."""
    match = _CONFIRMATION_LINE.search(brief or "")
    if match is None:
        return None
    line = (match.group(1) or match.group(2) or "").strip()
    return line or None


def apply_brief(instructions: str, brief: str, *, named_thanks: str) -> str:
    """Append the brief body, and use its opener line 2 when it has one.

    Front matter is not appended. It only selects the base prompt and opener.
    """
    _meta, text = split_front_matter(brief)
    if not text:
        return instructions
    updated = instructions
    replacement = confirmation_line(text)
    if replacement:
        generic = GENERIC_CONFIRMATION.replace("{named_thanks}", named_thanks)
        if generic in updated:
            updated = updated.replace(generic, replacement, 1)
    return f"{updated.rstrip()}\n\n{BRIEF_HEADING}\n{text}\n"
