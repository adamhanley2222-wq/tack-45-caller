from app.amd import amd_end_reason, decide_amd
from app.twiml import connect_stream_twiml, hangup_twiml, hold_twiml, say_and_hangup_twiml


def test_amd_classification():
    assert decide_amd(None) == "hold"
    assert decide_amd("") == "hold"
    assert decide_amd("human") == "connect"
    assert decide_amd("machine_end_beep") == "machine"
    assert decide_amd("machine_end_silence") == "machine"
    assert decide_amd("machine_end_other") == "machine"
    assert decide_amd("fax") == "hangup"
    assert decide_amd("unknown") == "hangup"
    assert amd_end_reason("fax") == "amd_fax"
    assert amd_end_reason("unknown") == "amd_unknown"
    assert amd_end_reason("machine_end_beep") == "amd_machine"


def test_twiml_documents():
    pause = hold_twiml(45)
    assert "<Pause" in pause
    assert 'length="45"' in pause

    hangup = hangup_twiml()
    assert "<Hangup/>" in hangup

    spoken = say_and_hangup_twiml("Please call us back & ask for Pat.")
    assert "Polly.Olivia-Neural" in spoken
    assert "en-AU" in spoken
    assert "Please call us back &amp; ask for Pat." in spoken
    assert "<Hangup/>" in spoken

    stream = connect_stream_twiml(
        "wss://caller.example.com/streams/media?token=abc",
        {"call_id": "1", "business": 'Fish & "Chips"'},
    )
    assert 'track="inbound_track"' in stream
    assert "wss://caller.example.com/streams/media?token=abc" in stream
    assert "Fish &amp;" in stream
    assert "Chips" in stream
    assert "<Connect>" in stream
    assert "<Stream" in stream
