"""Energy voice-activity check for the outbound opener.

Twilio sends 8 kHz G.711 μ-law. Quiet line noise and a single click stay
below the speech gate. A short stretch of speech, then silence, releases
the opener. The clock is the caller's monotonic clock, measured from when
the media stream starts.
"""

from __future__ import annotations

_RATE = 8000
# Codec noise and line hiss measure well under 500 RMS. A quiet spoken
# frame is above 2000. 1000 sits between them so noise does not count.
SPEECH_RMS = 1000
# One 20 ms click is not a person saying hello.
MIN_SPEECH_MS = 80


def mulaw_rms(payload: bytes) -> int:
    """Root-mean-square of decoded 16-bit samples. Empty input is 0."""
    if not payload:
        return 0
    total = 0
    for byte in payload:
        sample = _MULAW[byte]
        total += sample * sample
    return int((total / len(payload)) ** 0.5)


class OpenerVad:
    """Speech-then-silence, or a timeout when nobody speaks."""

    def __init__(self, *, silence_ms: int, max_wait_ms: int) -> None:
        self.silence_ms = max(1, int(silence_ms))
        self.max_wait_ms = max(1, int(max_wait_ms))
        self.started_at: float | None = None
        self.speech_ms = 0.0
        self.heard = False
        self.last_voice_at: float | None = None

    def start(self, now: float) -> None:
        self.started_at = now

    def feed(self, mulaw: bytes, now: float) -> str | None:
        """One inbound frame. ``speech`` or ``timeout`` means play the opener."""
        if self.started_at is None:
            self.started_at = now
        if mulaw:
            frame_ms = len(mulaw) * 1000 / _RATE
            if mulaw_rms(mulaw) >= SPEECH_RMS:
                self.speech_ms += frame_ms
                self.last_voice_at = now
                if self.speech_ms >= MIN_SPEECH_MS:
                    self.heard = True
            elif not self.heard:
                self.speech_ms = 0.0
        return self.poll(now)

    def poll(self, now: float) -> str | None:
        """Timeout and trailing silence, including when no new frame arrived."""
        if self.started_at is None:
            return None
        if self.heard and self.last_voice_at is not None:
            if (now - self.last_voice_at) * 1000 >= self.silence_ms:
                return "speech"
            return None
        if (now - self.started_at) * 1000 >= self.max_wait_ms:
            return "timeout"
        return None


def _decode_mulaw(byte: int) -> int:
    value = (~byte) & 0xFF
    sign = value & 0x80
    exponent = (value >> 4) & 0x07
    mantissa = value & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    if sign:
        sample = -sample
    return sample


_MULAW = tuple(_decode_mulaw(index) for index in range(256))
