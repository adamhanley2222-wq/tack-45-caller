"""Answered-by classification for Twilio DetectMessageEnd."""

from __future__ import annotations

_MACHINE = frozenset(
    {
        "machine",
        "machine_end_beep",
        "machine_end_silence",
        "machine_end_other",
    }
)


def decide_amd(answered_by: str | None) -> str:
    """Return ``hold``, ``connect``, ``machine``, or ``hangup``.

    An empty result means async detection has not finished, so the answer
    webhook should keep the line open. ``unknown`` and ``fax`` hang up.
    """
    value = (answered_by or "").strip().lower()
    if value == "":
        return "hold"
    if value == "human":
        return "connect"
    if value in _MACHINE:
        return "machine"
    return "hangup"


def amd_end_reason(answered_by: str | None) -> str:
    value = (answered_by or "").strip().lower()
    if value in _MACHINE:
        return "amd_machine"
    if value == "fax":
        return "amd_fax"
    if value in {"", "unknown"}:
        return "amd_unknown"
    safe = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
    return f"amd_{safe or 'unknown'}"
