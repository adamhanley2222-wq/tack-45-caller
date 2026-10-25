"""Per-target briefs are optional and never change a call that has none."""

from pathlib import Path

import pytest

from app.api import prepare_script
from app.briefs import BRIEF_HEADING, BriefError, resolve_brief
from tests.conftest import ROOT, make_settings
from tests.test_api import _allowed_time, _auth

MIKE_OPENER = "Hi Mike, this is Adam Hanley's digital assistant calling."


GENERIC = (
    "Thanks {name}. I'm an AI assistant calling for Tack45, a Sydney-based AI consultancy. "
    "Is now a good time for a quick chat?"
)
MARC_LINE = (
    "Thanks Marc. I'm an AI assistant calling for Tack45, Adam's AI business. "
    "Adam asked me to give you a call so you could hear what one of these sounds like. "
    "Is now a good time for a quick chat?"
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
    assert "Ha, I'll leave the cooking to you." in instructions
    assert "Have you got a couple of minutes?" not in instructions


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


def test_mike_brief_swaps_out_the_sales_prompt(tmp_path):
    settings = make_settings(tmp_path)
    text = (ROOT / "prompts" / "briefs" / "mike-charter-scheduling.md").read_text(encoding="utf-8")
    opener, plan = prepare_script(
        settings,
        {
            "name": "Mike",
            "business": "Avaaler",
            "notes": "",
            "to_number": "+61400000000",
            "brief": text,
        },
    )
    instructions = _instructions(plan)
    assert opener == MIKE_OPENER
    assert plan.setup_messages[1]["item"]["content"][0]["text"] == MIKE_OPENER
    assert "AI assistant" not in opener
    assert "base_prompt:" not in instructions
    assert "SALES SCRIPT" not in instructions
    assert "Sydney-based AI consultancy" not in instructions
    assert "Would you be open to a chat with Adam" not in instructions
    assert "practical AI" not in instructions
    assert "Ha, I'll leave the cooking to you." not in instructions
    assert "Teams" not in instructions
    assert "Do not pitch Tack45" in instructions
    assert "Do not ask for a sales meeting" in instructions
    assert "say plainly that you are an AI" in instructions
    assert "Never claim to be human" in instructions
    assert "Avaaler" in instructions
    assert "Wednesday 28 October at 4pm" in instructions
    assert "Thursday 29 October, any time from 1pm to 5pm" in instructions
    assert "Friday 30 October, from 3pm" in instructions
    assert "Saturday 31 October, from 9am" in instructions
    assert "about 45 minutes" in instructions
    assert "Adam's house in Forestville" in instructions
    assert "Never give a street address" in instructions
    assert "Adam will confirm the location" in instructions
    assert "Adam will send a calendar invite" in instructions
    assert "Adam will call him back" in instructions
    assert "record_opt_out" in instructions
    assert BRIEF_HEADING in instructions


def test_mike_brief_id_is_loaded_for_the_call(client, monkeypatch):
    monkeypatch.setattr("app.guards.utcnow", _allowed_time)
    http, seen, _settings = client
    created = http.post(
        "/calls",
        json={
            "to": "+61412345678",
            "name": "Mike",
            "business": "Avaaler",
            "notes": "",
            "brief_id": "mike-charter-scheduling",
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
    instructions = seen["providers"][0].sent[0]["session"]["instructions"]
    assert instructions.startswith("You are an AI phone assistant making a personal call")
    assert MIKE_OPENER in instructions
    assert "SALES SCRIPT" not in instructions
    assert "Teams" not in instructions


def test_unknown_base_prompt_is_rejected(tmp_path):
    folder = tmp_path / "briefs"
    folder.mkdir()
    prompt = ROOT / "prompts" / "system.txt"
    (folder / "bad.md").write_text("---\nbase_prompt: system\n---\nHello\n", encoding="utf-8")
    with pytest.raises(BriefError, match="base_prompt"):
        resolve_brief(str(folder), None, "bad", prompt_path=str(prompt))
    (folder / "escape.md").write_text("---\nbase_prompt: ../system\n---\nHello\n", encoding="utf-8")
    with pytest.raises(BriefError, match="base_prompt"):
        resolve_brief(str(folder), None, "escape", prompt_path=str(prompt))
