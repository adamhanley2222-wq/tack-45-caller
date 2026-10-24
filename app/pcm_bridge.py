"""Twilio μ-law ↔ Gemini Live PCM.

Twilio Media Streams are 8 kHz G.711 μ-law. The Gemini Live API takes raw
little-endian 16-bit PCM at 16 kHz and speaks raw little-endian 16-bit PCM at
24 kHz. Grok, OpenAI, GPT-Live, and the Gemini cascade never use this module;
they stay on μ-law passthrough.

``audioop`` does the G.711 conversion and the rate change. It is still in the
Python 3.12 standard library this service runs on. Chunk boundaries keep the
``ratecv`` state so a split frame matches one continuous conversion.
"""

from __future__ import annotations

import sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop

_PCM_WIDTH = 2


class PcmBridge:
    """Stateful resampler for one call.

    Inbound: 8 kHz μ-law → 16 kHz PCM. Outbound: 24 kHz PCM → 8 kHz μ-law.
    A rate other than 24 kHz is converted on its own, without touching the
    24 kHz state used for the model's normal audio.
    """

    def __init__(self) -> None:
        self._up_state = None
        self._down_state = None
        self._down_odd = b""

    def mulaw8k_to_pcm16k(self, mulaw: bytes) -> bytes:
        if not mulaw:
            return b""
        pcm8 = audioop.ulaw2lin(mulaw, _PCM_WIDTH)
        pcm8 = _to_native(pcm8)
        pcm16, self._up_state = audioop.ratecv(
            pcm8, _PCM_WIDTH, 1, 8000, 16000, self._up_state
        )
        return _to_little(pcm16)

    def pcm24k_to_mulaw8k(self, pcm: bytes, rate: int = 24000) -> bytes:
        if rate <= 0:
            raise ValueError("pcm sample rate must be positive")
        data = pcm
        if rate == 24000:
            data = self._down_odd + data
            if len(data) % _PCM_WIDTH:
                self._down_odd = data[-1:]
                data = data[:-1]
            else:
                self._down_odd = b""
        elif len(data) % _PCM_WIDTH:
            data = data[:-1]
        if not data:
            return b""
        native = _to_native(data)
        if rate == 24000:
            pcm8, self._down_state = audioop.ratecv(
                native, _PCM_WIDTH, 1, 24000, 8000, self._down_state
            )
        else:
            pcm8, _state = audioop.ratecv(native, _PCM_WIDTH, 1, rate, 8000, None)
        if not pcm8:
            return b""
        return audioop.lin2ulaw(pcm8, _PCM_WIDTH)


def linear16_to_mulaw8k(pcm: bytes, rate: int) -> bytes:
    """Headerless little-endian 16-bit PCM to 8 kHz μ-law.

    The TTS probe uses this when Cloud TTS returns LINEAR16. Call audio does
    not. ``rate`` is the PCM sample rate.
    """
    if rate <= 0:
        raise ValueError("pcm sample rate must be positive")
    if len(pcm) % _PCM_WIDTH:
        pcm = pcm[: len(pcm) - (len(pcm) % _PCM_WIDTH)]
    if not pcm:
        return b""
    native = _to_native(pcm)
    if rate != 8000:
        native, _state = audioop.ratecv(native, _PCM_WIDTH, 1, rate, 8000, None)
    if not native:
        return b""
    return audioop.lin2ulaw(native, _PCM_WIDTH)


def mulaw_to_8k(payload: bytes, rate: int) -> bytes:
    """μ-law at ``rate`` to 8 kHz μ-law. 8 kHz input is returned unchanged."""
    if rate == 8000:
        return payload
    return linear16_to_mulaw8k(audioop.ulaw2lin(payload, _PCM_WIDTH), rate)


def _to_native(pcm: bytes) -> bytes:
    if sys.byteorder == "little":
        return pcm
    return audioop.byteswap(pcm, _PCM_WIDTH)


def _to_little(pcm: bytes) -> bytes:
    if sys.byteorder == "little":
        return pcm
    return audioop.byteswap(pcm, _PCM_WIDTH)
