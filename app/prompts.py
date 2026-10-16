"""Load the system prompt and the fixed compliance opener."""

from __future__ import annotations

import re
from pathlib import Path

_TOKEN = re.compile(r"\{(opener|name|business|notes|number)\}")


def load_text(path: str) -> str:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return file.read_text(encoding="utf-8").strip()


def render_instructions(
    template: str,
    *,
    opener: str,
    name: str,
    business: str,
    notes: str,
    number: str,
) -> str:
    """Fill prompt tokens. Values are inserted once and are not re-scanned."""
    mapping = {
        "opener": opener,
        "name": name.strip() or "unknown",
        "business": business.strip() or "unknown",
        "notes": notes.strip() or "none",
        "number": number,
    }
    return _TOKEN.sub(lambda match: mapping[match.group(1)], template)
