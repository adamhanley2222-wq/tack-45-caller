"""Controller logic for the Gemini cascade. No network and no Twilio."""

import json

from app.gemini_turn import (
    BACKCHANNELS,
    OPT_OUT_LINE,
    POLITE_END,
    CallerTurns,
    GeminiController,
    conversation_contents,
    deepgram_headers,
    deepgram_listen_url,
    events_from_gemini_payload,
    is_rate_limit,
    spoken_only,
    take_sentences,
    text_request,
)
from app.tts_probe import DEFAULT_MODEL, DEFAULT_STYLE, interaction_body


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _controller() -> GeminiController:
    return GeminiController(
        call_id="call-1",
        call_sid="CA1",
        phone="+61400000000",
        opener="Hi, is this Sam?",
        instructions="be brief",
        hangup_grace_seconds=8.0,
        clock=Clock(),
    )


def _start(controller: GeminiController | None = None) -> GeminiController:
    controller = controller or _controller()
    controller.on_twilio(
        {
            "event": "start",
            "streamSid": "MZ1",
            "start": {"streamSid": "MZ1", "callSid": "CA1"},
        }
    )
    return controller


def _finish_opener(controller: GeminiController) -> None:
    mark = controller.queue_mark("opener")
    name = mark.twilio_messages[0]["mark"]["name"]
    controller.on_twilio({"event": "mark", "streamSid": "MZ1", "mark": {"name": name}})


def test_barge_in_is_ignored_while_the_opener_is_protected():
    controller = _start()
    assert controller.note_playback(8000, controller.generation) is True
    step = controller.on_caller_speech()
    assert step.twilio_messages == []
    assert step.cancel_speech is False
    assert step.drop_queued_audio is False
    assert controller.generation == 0


def test_barge_in_clears_twilio_and_drops_the_rest_of_the_reply():
    controller = _start()
    _finish_opener(controller)
    controller.on_caller_turn("Yes, this is Sam.")
    generation = controller.generation
    assert controller.note_playback(8000, generation) is True
    step = controller.on_caller_speech()
    assert step.cancel_speech is True
    assert step.drop_queued_audio is True
    assert step.twilio_messages == [{"event": "clear", "streamSid": "MZ1"}]
    assert controller.generation == generation + 1
    dropped = controller.on_assistant_sentence("I can explain the chat.", generation)
    assert dropped.plays == []
    assert controller.note_playback(160, generation) is False


def test_sentence_chunking_holds_a_fragment_and_keeps_abbreviations():
    sentences, rest = take_sentences("Thanks Sam. Is now", flush=False, early_clause=False)
    assert sentences == ["Thanks Sam."]
    assert rest == "Is now"
    done, rest = take_sentences(rest + " a good time?", flush=True, early_clause=False)
    assert done == ["Is now a good time?"]
    assert rest == ""

    doctor, rest = take_sentences(
        "Please ask Dr. Hanley about it.",
        flush=True,
        early_clause=False,
    )
    assert doctor == ["Please ask Dr. Hanley about it."]
    assert rest == ""

    decimal, rest = take_sentences(
        "It takes 2.5 minutes. Then we stop.",
        flush=False,
        early_clause=False,
    )
    assert decimal == ["It takes 2.5 minutes."]
    assert rest == "Then we stop."

    morning, rest = take_sentences("Call at 10 a.m. tomorrow.", flush=True, early_clause=False)
    assert morning == ["Call at 10 a.m. tomorrow."]
    assert rest == ""


def test_first_sentence_stays_short():
    early, rest = take_sentences(
        "Thanks so much, I'm an AI assistant calling for Tack45.",
        flush=False,
        early_clause=True,
    )
    assert early == ["Thanks so much,"]
    assert rest.startswith("I'm an AI")

    later, rest = take_sentences(
        "Thanks so much, I'm an AI assistant calling for Tack45.",
        flush=False,
        early_clause=False,
    )
    assert later == []
    assert "Thanks so much" in rest

    words = " ".join(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen"]
    )
    capped, rest = take_sentences(words, flush=False, early_clause=True)
    assert capped == ["one two three four five six seven eight nine ten eleven twelve"]
    assert rest == "thirteen"

    sooner, rest = take_sentences(
        "one two three four five six seven",
        flush=False,
        early_clause=True,
        early_words=6,
    )
    assert sooner == ["one two three four five six"]
    assert rest == "seven"

    held, rest = take_sentences("Yes, now is fine.", flush=False, early_clause=True)
    assert held == []
    assert rest == "Yes, now is fine."
    flushed, rest = take_sentences(
        "Yes, now is fine.",
        flush=False,
        early_clause=True,
        flush_trailing=True,
    )
    assert flushed == ["Yes, now is fine."]
    assert rest == ""


def test_spoken_text_never_includes_accent_instructions_or_tool_names():
    assert spoken_only("Thanks Sam.") == "Thanks Sam."
    assert spoken_only("(quietly) Thanks Sam.") == "Thanks Sam."
    assert spoken_only("(Australian English accent) Thanks Sam.") == "Thanks Sam."
    assert spoken_only("Australian English accent, relaxed and friendly") == ""
    assert spoken_only("Say this in an Australian accent: Hello there.") == ""
    assert spoken_only("I will call record_opt_out now.") == ""
    assert spoken_only("Please end_call.") == ""
    for line in (*BACKCHANNELS, POLITE_END, OPT_OUT_LINE, "Hi, is this Sam?"):
        assert spoken_only(line) == line
        assert "accent" not in line.lower()


def test_opt_out_is_honoured_before_end_call():
    controller = _start()
    _finish_opener(controller)
    heard = controller.on_caller_turn("Please stop calling me.")
    assert heard.begin_reply is True
    assert heard.transcripts[0]["role"] == "caller"
    step = controller.on_tool("record_opt_out", {"reason": "stop calling"}, controller.generation)
    assert step.opt_out_reason == "stop calling"
    assert step.hangup is False
    assert controller.opted_out is True
    assert controller.end_reason == "opt_out"
    assert controller.hangup_requested is True
    end = controller.on_tool("end_call", {"reason": "opt out"}, controller.generation)
    assert end.opt_out_reason is None
    assert controller.end_reason == "opt_out"
    again = controller.on_tool("record_opt_out", {"reason": "stop calling"}, controller.generation)
    assert again.opt_out_reason is None


def test_opt_out_without_speech_uses_the_cached_confirmation():
    controller = _start()
    _finish_opener(controller)
    controller.on_caller_turn("Remove me.")
    controller.on_tool("record_opt_out", {"reason": "remove me"}, controller.generation)
    done = controller.on_reply_finished(controller.generation)
    assert [play.text for play in done.plays] == [OPT_OUT_LINE]
    assert done.follow_with_end is True
    assert done.hangup is False
    assert done.transcripts[0]["text"] == OPT_OUT_LINE


def test_rate_limit_plays_one_cached_goodbye_then_stops():
    controller = _start()
    _finish_opener(controller)
    step = controller.on_rate_limited("tts")
    assert [play.text for play in step.plays] == [POLITE_END]
    assert step.cancel_speech is True
    assert step.drop_queued_audio is True
    assert step.twilio_messages[0]["event"] == "clear"
    assert controller.end_reason == "rate_limited"
    assert controller.rate_limited is True
    again = controller.on_rate_limited("tts")
    assert again.plays == []
    assert again.hangup is True
    assert again.end_reason == "rate_limited"


def test_caller_turn_after_the_opener_queues_a_backchannel():
    controller = _start()
    _finish_opener(controller)
    step = controller.on_caller_turn("Yes, this is Sam.")
    assert step.begin_reply is True
    assert step.caller_text == "Yes, this is Sam."
    assert step.plays[0].purpose == "backchannel"
    assert step.plays[0].cached_only is True
    assert step.plays[0].text in BACKCHANNELS
    assert controller.transcript[0]["text"] == "Hi, is this Sam?"
    assert controller.transcript[1]["text"] == "Yes, this is Sam."


def test_caller_speech_during_the_opener_is_kept_until_the_mark():
    controller = _start()
    held = controller.on_caller_turn("Yes this is Sam.")
    assert held.begin_reply is False
    assert controller.protecting_opener is True
    mark = controller.queue_mark("opener")
    released = controller.on_twilio(
        {"event": "mark", "streamSid": "MZ1", "mark": {"name": mark.twilio_messages[0]["mark"]["name"]}}
    )
    assert controller.protecting_opener is False
    assert released.begin_reply is True
    assert released.caller_text == "Yes this is Sam."


def test_deepgram_results_endpoint_once():
    turns = CallerTurns()
    assert turns.push({"type": "SpeechStarted"}).speech_started is True
    interim = turns.push(
        {
            "type": "Results",
            "is_final": False,
            "speech_final": False,
            "channel": {"alternatives": [{"transcript": "yes"}]},
        }
    )
    assert interim.text is None
    done = turns.push(
        {
            "type": "Results",
            "is_final": True,
            "speech_final": True,
            "channel": {"alternatives": [{"transcript": "yes this is Sam"}]},
        }
    )
    assert done.text == "yes this is Sam"
    assert turns.push({"type": "UtteranceEnd"}).text is None


def test_gemini_stream_events_skip_thoughts_and_keep_tools():
    events = events_from_gemini_payload(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "hidden", "thought": True},
                            {"text": "Thanks Sam. "},
                            {"functionCall": {"name": "record_opt_out", "args": {"reason": "stop"}}},
                        ]
                    }
                }
            ]
        }
    )
    assert [(event.text, event.tool_name) for event in events] == [
        ("Thanks Sam. ", None),
        ("", "record_opt_out"),
    ]
    assert events[1].tool_args == {"reason": "stop"}
    encoded = events_from_gemini_payload(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "end_call",
                                    "args": "{\"reason\": \"goodbye\"}",
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert encoded[0].tool_name == "end_call"
    assert encoded[0].tool_args == {"reason": "goodbye"}
    limited = events_from_gemini_payload({"error": {"code": 429, "message": "Resource exhausted"}})
    assert limited[0].rate_limited is True
    assert is_rate_limit(429, "HTTP 429: no") is True
    assert is_rate_limit(None, "HTTP 500: no") is False


def test_text_request_uses_the_tools_and_not_the_tts_style():
    class Settings:
        gemini_thinking_budget = 0

    transcript = [
        {"role": "assistant", "text": "Hi, is this Sam?"},
        {"role": "caller", "text": "Yes."},
    ]
    body = text_request("be brief", transcript, Settings())
    encoded = json.dumps(body)
    assert "Australian English accent" not in encoded
    names = {tool["name"] for tool in body["tools"][0]["functionDeclarations"]}
    assert names == {"end_call", "record_opt_out"}
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert body["contents"][0]["role"] == "user"
    assert body["contents"][1]["role"] == "model"
    assert body["contents"][2]["parts"][0]["text"] == "Yes."
    settings = Settings()
    settings.gemini_thinking_budget = -1
    omitted = text_request("be brief", transcript, settings)
    assert "thinkingConfig" not in omitted["generationConfig"]
    assert conversation_contents([])[0]["role"] == "user"


def test_tts_language_is_metadata_and_the_probe_default_is_unchanged():
    styled = interaction_body(
        "Thanks Sam.",
        model=DEFAULT_MODEL,
        style=DEFAULT_STYLE,
        voice="Charon",
        language="en-AU",
    )
    spoken = styled["input"][0]["content"][0]
    assert spoken["text"] == "Thanks Sam."
    assert "accent" not in spoken["text"].lower()
    assert spoken["annotations"] == [{"type": "speech_metadata", "style": DEFAULT_STYLE}]
    assert styled["generation_config"]["speech_config"] == [
        {"voice": "Charon", "language": "en-AU"}
    ]
    assert styled["response_format"]["mime_type"] == "audio/mulaw"
    assert styled["response_format"]["sample_rate"] == 8000
    probe = interaction_body("Thanks Ian.", model=DEFAULT_MODEL, style=DEFAULT_STYLE, voice="Kore")
    assert probe["generation_config"]["speech_config"] == [{"voice": "Kore"}]


def test_deepgram_url_is_mulaw_8k_with_endpointing():
    url = deepgram_listen_url(
        base="wss://api.deepgram.com/v1/listen",
        model="nova-3",
        language="en-AU",
        endpointing_ms=400,
    )
    assert url.startswith("wss://api.deepgram.com/v1/listen?")
    assert "encoding=mulaw" in url
    assert "sample_rate=8000" in url
    assert "language=en-AU" in url
    assert "model=nova-3" in url
    assert "vad_events=true" in url
    assert "endpointing=400" in url
    assert "interim_results=true" in url
    assert deepgram_headers("dg-test") == {"Authorization": "Token dg-test"}
