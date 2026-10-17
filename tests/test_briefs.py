"""Per-target briefs are optional and never change a call that has none."""

from pathlib import Path

from app.api import prepare_script
from app.briefs import BRIEF_HEADING
from tests.conftest import ROOT, make_settings
from tests.test_api import _allowed_time, _auth


GENERIC = (
    "Thanks {name}. I'm an AI assistant calling for Tack45, a Sydney-based AI consultancy. "
    "Would a short chat about practical AI for your business be useful?"
)
MARC_LINE = (
    "Thanks Marc. I'm an AI assistant calling for Tack45, Adam's AI business. "
    "Adam asked me to give you a call so you could hear what one of these sounds like. "
    "Have you got a couple of minutes?"
)


def _instructions(plan) -> str:
    return plan.setup_messages[0]["session"]["instructions"]


def test_absent_brief_leaves_the_prompt_unchanged(tmp_path):
    settings = make_settings(tmp_path)
    call = {
        "name": "Sam Lee",
        "business": "Harbour Roasters",
        "notes": "left a message",
        "to_number": "+61412345678",
    }
    opener, plain = prepare_script(settings, call)
    same_opener, empty = prepare_script(settings, {**call, "brief": "   "})
    assert opener == same_opener == "Hi, is this Sam?"
    assert _instructions(plain) == _instructions(empty)
    text = _instructions(plain)
    assert BRIEF_HEADING not in text
    assert GENERIC.format(name="Sam") in text
    assert plain.setup_messages[1]["item"]["content"][0]["text"] == "Hi, is this Sam?"


def test_brief_id_loads_the_file_and_replaces_the_confirmation_line(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, _settings = client
    created = http.post(
        "/calls",
        json={
            "to": "+61412345678",
            "name": "Marc Hanley",
            "business": "Barrenjoey Inspections",
            "notes": "",
            "brief_id": "barrenjoey-marc",
        },
        headers=_auth(),
    )
    assert created.status_code == 201, created.text
    body = created.json()
    voice = http.post(
        f"/twilio/voice?call_id={body['id']}",
        data={"CallSid": body["call_sid"]},
    )
    assert voice.status_code == 200
    provider = seen["providers"][0]
    assert provider.sent[1]["item"]["content"][0]["text"] == "Hi, is this Marc?"
    instructions = provider.sent[0]["session"]["instructions"]
    brief = (ROOT / "prompts" / "briefs" / "barrenjoey-marc.md").read_text(encoding="utf-8").strip()
    assert instructions.rstrip().endswith(brief)
    assert BRIEF_HEADING in instructions
    assert MARC_LINE in instructions
    assert GENERIC.format(name="Marc") not in instructions
    assert "record_opt_out" in instructions
    assert "Never claim to be human" in instructions
    assert "Barrenjoey Inspections" in instructions


def test_inline_brief_is_appended(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, _settings = client
    created = http.post(
        "/calls",
        json={
            "to": "+61280001111",
            "name": "Sam Lee",
            "business": "Harbour Roasters",
            "notes": "",
            "brief": "Ask about the roastery inbox, then the meeting ask.",
        },
        headers=_auth(),
    )
    assert created.status_code == 201, created.text
    body = created.json()
    voice = http.post(
        f"/twilio/voice?call_id={body['id']}",
        data={"CallSid": body["call_sid"]},
    )
    assert voice.status_code == 200
    provider = seen["providers"][0]
    assert provider.sent[1]["item"]["content"][0]["text"] == "Hi, is this Sam?"
    instructions = provider.sent[0]["session"]["instructions"]
    assert BRIEF_HEADING in instructions
    assert "Ask about the roastery inbox, then the meeting ask." in instructions
    assert GENERIC.format(name="Sam") in instructions


def test_unknown_brief_id_returns_400_and_does_not_dial(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, _settings = client
    missing = http.post(
        "/calls",
        json={"to": "+61412345678", "name": "Sam Lee", "brief_id": "no-such-brief"},
        headers=_auth(),
    )
    assert missing.status_code == 400
    assert missing.json()["code"] == "unknown_brief"
    escaped = http.post(
        "/calls",
        json={"to": "+61412345678", "name": "Sam Lee", "brief_id": "../system"},
        headers=_auth(),
    )
    assert escaped.status_code == 400
    assert escaped.json()["code"] == "unknown_brief"
    assert seen["requests"] == []
    assert Path(_settings.briefs_dir, "barrenjoey-marc.md").is_file()
