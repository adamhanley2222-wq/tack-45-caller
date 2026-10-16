"""μ-law payload handling.

Twilio Media Streams and the realtime providers are both configured for
8 kHz G.711 μ-law (``audio/pcmu``). Payloads are base64 text and are forwarded
unchanged. This module does not decode, resample, or transcode them.
"""

from __future__ import annotations


def passthrough_audio(payload: str) -> str:
    """Return the same base64 payload that arrived."""
    if not isinstance(payload, str) or payload == "":
        raise ValueError("audio payload must be a non-empty base64 string")
    return payload


def payload_nbytes(payload: str) -> int:
    """Byte length of a base64 payload, without decoding it.

    G.711 μ-law at 8 kHz is one byte per sample, so this is also the duration
    in samples. Divide by 8000 for seconds.
    """
    padding = payload.count("=")
    return (len(payload) * 3) // 4 - padding
