"""TwiML documents for the answer, machine, and media-stream legs."""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr


def _document(body: str) -> str:
    return f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>'


def hold_twiml(seconds: int) -> str:
    """Keep the line up while async answering-machine detection finishes."""
    length = max(1, int(seconds))
    return _document(f"<Pause length={quoteattr(str(length))}/>")


def hangup_twiml() -> str:
    return _document("<Hangup/>")


def say_and_hangup_twiml(message: str) -> str:
    text = escape(message.strip())
    return _document(
        "<Say language=\"en-AU\" voice=\"Polly.Olivia-Neural\">"
        f"{text}</Say><Hangup/>"
    )


def connect_stream_twiml(stream_url: str, parameters: dict[str, str]) -> str:
    """Bidirectional Media Stream. Audio is inbound μ-law; we send μ-law back."""
    parts = [
        "<Connect><Stream",
        f" url={quoteattr(stream_url)}",
        ' track="inbound_track">',
    ]
    for name, value in parameters.items():
        parts.append(
            f"<Parameter name={quoteattr(name)} value={quoteattr(value)}/>"
        )
    parts.append("</Stream></Connect>")
    return _document("".join(parts))


def stream_parameters(call: dict) -> dict[str, str]:
    """Context Twilio will echo on the stream ``start`` event."""
    notes = (call.get("notes") or "")[:500]
    return {
        "call_id": str(call.get("id") or ""),
        "name": str(call.get("name") or ""),
        "business": str(call.get("business") or ""),
        "notes": notes,
        "to": str(call.get("to_number") or ""),
        "timezone": str(call.get("timezone") or ""),
    }
