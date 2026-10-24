"""μ-law and sample-rate conversion for Gemini Live.

Grok, OpenAI, GPT-Live, and the cascade never call this. These tests lock
the Twilio 8 kHz μ-law ↔ Live PCM bridge: 16 kHz in, 24 kHz out.
"""

from __future__ import annotations

import struct
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop

from app.pcm_bridge import PcmBridge


def test_mulaw_round_trip_matches_the_standard_codec():
    raw = bytes(range(256))
    once = audioop.lin2ulaw(audioop.ulaw2lin(raw, 2), 2)
    # 0x7F and 0xFF are both digital silence, so the first pass collapses them.
    # A second pass is stable, and a loud code survives unchanged.
    assert audioop.lin2ulaw(audioop.ulaw2lin(once, 2), 2) == once
    assert audioop.ulaw2lin(b"\xff", 2) == b"\x00\x00"
    assert audioop.ulaw2lin(b"\x7f", 2) == b"\x00\x00"
    assert audioop.lin2ulaw(audioop.ulaw2lin(b"\x00", 2), 2) == b"\x00"


def test_upsampling_doubles_the_rate_and_chunking_matches_one_shot():
    mulaw = bytes([0x10, 0x7F, 0xFF, 0x00]) * 400
    whole = PcmBridge().mulaw8k_to_pcm16k(mulaw)
    # 1600 samples at 8 kHz become 3200 at 16 kHz, within the resampler's hold.
    assert abs(len(whole) // 2 - len(mulaw) * 2) <= 2
    split = PcmBridge()
    chunked = b""
    for offset in range(0, len(mulaw), 160):
        chunked += split.mulaw8k_to_pcm16k(mulaw[offset : offset + 160])
    assert chunked == whole


def test_silence_stays_quiet_at_16khz():
    pcm = PcmBridge().mulaw8k_to_pcm16k(b"\xff" * 160)
    samples = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
    assert max(abs(sample) for sample in samples) == 0


def test_downsampling_thirds_the_rate_and_odd_bytes_carry_across_chunks():
    samples = [1000, -1000, 0] * 800
    pcm = struct.pack("<" + "h" * len(samples), *samples)
    whole = PcmBridge().pcm24k_to_mulaw8k(pcm)
    assert len(whole) == len(samples) // 3

    split = PcmBridge()
    chunked = b""
    offset = 0
    first = True
    while offset < len(pcm):
        take = 5 if first else 90
        first = False
        chunked += split.pcm24k_to_mulaw8k(pcm[offset : offset + take])
        offset += take
    assert chunked == whole


def test_a_tone_survives_24khz_to_mulaw_and_is_not_silence():
    samples = []
    for index in range(2400):
        samples.append(8000 if (index // 12) % 2 == 0 else -8000)
    pcm = struct.pack("<" + "h" * len(samples), *samples)
    mulaw = PcmBridge().pcm24k_to_mulaw8k(pcm)
    decoded = audioop.ulaw2lin(mulaw, 2)
    heard = struct.unpack("<" + "h" * (len(decoded) // 2), decoded)
    assert max(abs(sample) for sample in heard) > 1000


def test_empty_input_and_a_declared_rate_other_than_24khz():
    bridge = PcmBridge()
    assert bridge.mulaw8k_to_pcm16k(b"") == b""
    assert bridge.pcm24k_to_mulaw8k(b"") == b""
    pcm16 = struct.pack("<" + "h" * 160, *([2000, -2000] * 80))
    mulaw = bridge.pcm24k_to_mulaw8k(pcm16, rate=16000)
    assert mulaw
    assert len(mulaw) == pytest.approx(80, abs=2)
