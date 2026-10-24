# tack45-caller

Outbound AI voice calls for Tack45, a Sydney AI consultancy. The service dials an Australian business on a Twilio number, waits for async answering-machine detection, and on a human answer bridges the call to a realtime speech-to-speech model. The same relay answers inbound calls to that number. Twilio audio stays 8 kHz G.711 μ-law. Grok, OpenAI, GPT-Live, and the Gemini cascade pass that payload through. Gemini Live (`VOICE_PROVIDER=geminilive`) resamples it: 16 kHz PCM into the Live API, and 24 kHz PCM back to 8 kHz μ-law.

The sales script is a placeholder. Compliance behaviour is already in place: a short identity check, then an AI disclosure and Tack45 identification before any pitch, an Australian accent instruction, calling-hours and do-not-call guards, and an opt-out tool.

No live Twilio or model calls are made by the tests. `scripts/place_call.py` is the only way to dial, and it does nothing until you point it at a running service that has credentials.

## Call flow

1. `POST /calls` checks the bearer token, the do-not-call list, and the callee's local calling window, then creates a Twilio call. The realtime provider is not contacted while the phone is ringing.
2. Twilio dials `TWILIO_FROM_NUMBER` (intended to be `+61290904464`) with `MachineDetection=DetectMessageEnd` and async AMD.
3. On answer, the voice webhook returns `<Connect><Stream>` immediately. It does not pause for AMD. The realtime session is opened at that moment, the session config is sent straight away, and the short opener is spoken without waiting for `AnsweredBy`.
4. Human, or AMD still pending: the media WebSocket is already up. A later human result is stored and the stream is left as it is.
5. Machine: hang up, or play `MACHINE_MESSAGE` and then hang up. No voicemail is left unless `MACHINE_MESSAGE` is set. Fax hangs up. `unknown` and any other unrecognised result stay on the line. Each result is stored.
6. Grok, OpenAI, and GPT-Live forward μ-law payloads unchanged. Gemini Live resamples each frame. Barge-in sends Twilio `clear` and drops audio still queued here. `end_call` hangs up via the Twilio REST API after the current audio finishes.

```
phone  <->  Twilio Media Streams (audio/pcmu, 8 kHz)
                <->  tack45-caller
                        <->  wss://api.x.ai/v1/realtime   (VOICE_PROVIDER=grok)
                        <->  wss://api.openai.com/v1/realtime  (VOICE_PROVIDER=openai)
                        <->  wss://api.openai.com/v1/live/sessions  (VOICE_PROVIDER=gptlive)
                        <->  Deepgram listen + Gemini text + Gemini TTS
                             (VOICE_PROVIDER=gemini, still μ-law 8 kHz)
                        <->  Gemini Live  (VOICE_PROVIDER=geminilive)
                             8 kHz μ-law <-> 16 kHz PCM in / 24 kHz PCM out
```

## Run locally

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in tokens when you are ready to dial; health works without them
uvicorn app.main:app --host 0.0.0.0 --port 8787
```

`GET /health` returns `{"status":"ok","service":"tack45-caller"}`.

Twilio must be able to reach `PUBLIC_BASE_URL`. For a laptop, point that at an https tunnel and set `CALLER_BASE_URL` to the local API when you use the test script.

```bash
pytest
```

## Place one call

```bash
CALLER_BASE_URL=http://127.0.0.1:8787 API_BEARER_TOKEN=your-token \
  python scripts/place_call.py \
  --to +61412345678 \
  --name "Sam Lee" \
  --business "Harbour Roasters" \
  --notes "Asked about invoices last month"
```

Optional `--timezone Australia/Perth` when the number's area code is ambiguous. The script exits non-zero on a refusal (hours, do-not-call, auth) and prints the JSON error.

`POST /calls` body:

```json
{"to": "+61412345678", "name": "Sam Lee", "business": "Harbour Roasters", "notes": ""}
```

`timezone` is optional. An optional `brief` string, or a `brief_id` that loads `prompts/briefs/<id>.md`, is appended to the outbound system prompt under `TARGET-SPECIFIC CALL BRIEF`. Where that brief has an opener line 2, it replaces the generic post-confirmation sentence. The spoken opener stays `Hi, is this {first name}?`. Disclosure, opt-out, honesty, and no-claims rules still apply. An unknown `brief_id` returns 400 and does not dial. Send one of `brief` or `brief_id`, not both.

```json
{"to": "+61412345678", "name": "Marc Hanley", "business": "Barrenjoey Inspections", "notes": "", "brief_id": "barrenjoey-marc"}
```

`GET /calls/{id}` returns the AMD result, timestamps, duration, transcript, opt-out flag, and end reason.

## Calling hours and do-not-call

Evaluated in the callee's timezone. The default is `Australia/Sydney`.

| When | Allowed |
| --- | --- |
| Monday–Friday | 09:00 inclusive to 20:00 exclusive |
| Saturday | 09:00 inclusive to 17:00 exclusive |
| Sunday | never |
| National public holiday | never, including the substitute weekday when New Year's Day, Australia Day, Christmas Day, or Boxing Day falls on a weekend |

Area codes `02`, `03`, and `07` select Sydney, Melbourne, and Brisbane. Mobiles and `08` numbers (Adelaide, Darwin, and Perth) keep the default unless you pass `timezone`. State holidays such as the King's Birthday are not blocked.

`data/do_not_call.txt` is one number per line (`#` comments allowed). It is re-read before every dial. Opt-outs from a live call are written to the `do_not_call` table and block the next attempt even if the file is unchanged. This list is yours to maintain; it is not the ACMA Do Not Call Register. Wash campaign numbers against that register before you dial.

`config/test_numbers.txt` is one number per line (`#` comments allowed). Those numbers skip the calling-hours check only, including Sundays and national holidays. The do-not-call list and the refusal to dial the caller ID still apply. `TEST_NUMBERS` is an optional comma-separated addition to that file. Both are re-read before every dial.

## Voice providers

`VOICE_PROVIDER=grok` (default), `openai`, `gptlive`, `gemini`, or `geminilive`.

Grok and OpenAI Realtime both request `audio/pcmu` in and out, `server_vad`, and the tools `end_call` and `record_opt_out`. Grok also enables `grok-transcribe` on input and, unless `USE_FORCE_MESSAGE=false`, speaks the opener with xAI `force_message` (verbatim TTS, not a model-written line). OpenAI has no `force_message`; the opener is requested with `response.create` instructions. See the [xAI voice agent guide](https://docs.x.ai/docs/guides/voice/agent) and the [phone cookbook](https://docs.x.ai/cookbook/examples/voice-agent-phone).

`gptlive` is OpenAI GPT-Live-1 (`gpt-live-1`, voice `ripple`) on `wss://api.openai.com/v1/live/sessions`, using the same `OPENAI_API_KEY`. The session asks for `{"type":"audio/pcmu","rate":8000}` so Twilio μ-law is forwarded unchanged. The opener is a developer message inside `session.start`. `end_call` and `record_opt_out` are registered on Responses delegation (`GPTLIVE_RESPONSES_MODEL`, default `gpt-6-luna`); this server runs those tools and returns `response.item.create`. Client delegation is not used: its delegation event does not include the tool name or arguments.

The opener in `prompts/compliance_opener.txt` is only `Hi, is this {first name}?`. The first name is the first word of the call's `name`. With no name it is `Hi, is that the business owner?`. The model waits. On a confirmation, the next turn must be the AI disclosure and the Tack45 identification before any pitch. A wrong person gets that identification and a request to reach the right person or a callback time, and no pitch. A no to the chat asks about removing the number. Any opt-out, at any time, calls `record_opt_out` and then `end_call`.

The system prompt is `prompts/system.txt`. Replace the placeholder sales section when the real script is ready, and keep the accent, disclosure, opt-out, and `end_call` rules. The accent line asks for a male Australian voice with a calm, clear, ABC-newsreader-style General Australian accent.

Barge-in is suppressed while the opener is still playing, so the disclosure is not cut off. After that, caller speech sends Twilio `{"event":"clear"}` and drops queued μ-law. When the model calls `end_call`, the relay waits until Twilio echoes a mark (playback finished), then `POST`s `Status=completed` on the call. A watchdog hangs up if the mark never returns.

### Gemini cascade

`VOICE_PROVIDER=gemini` does not open a realtime speech-to-speech socket. Grok, OpenAI, GPT-Live, and Gemini Live are unchanged when this provider is selected. The same prompt assembly is used (`prompts/system.txt`, an optional `prompts/briefs/<id>.md`, and the call notes), and the same tools run: `end_call` and `record_opt_out`. Opt-out is written as soon as the tool call arrives, before the goodbye finishes playing. Calling hours, the do-not-call list, transcripts, and the finished-call webhook are the same code paths as the other providers.

Audio stays 8 kHz G.711 μ-law. Nothing is resampled.

* Listener: Deepgram streaming STT for the whole call (`DEEPGRAM_API_KEY`), μ-law 8 kHz, `language=en-AU`, `vad_events` and `endpointing` (default 400 ms). `SpeechStarted` is barge-in. `speech_final` ends the turn.
* Speaker: Gemini text (`GEMINI_TEXT_MODEL`, default `gemini-2.5-flash`) streamed into `gemini-3.8-flash-lite-tts`. Voice `Charon`. The accent is only the speech-style metadata `Australian English accent, relaxed and friendly`. Language is `en-AU` on the speech config. Neither is written into the words that are spoken. The request is the same Interactions body as `POST /probe/tts`.
* Controller: the opener and a few fixed lines (backchannels, the opt-out confirmation, a polite goodbye) are pre-rendered and cached. A short backchannel plays when the caller finishes a turn, covering the wait for the first real sentence. The first sentence is kept short. The next sentence is sent to TTS while the current one is still playing. Barge-in sends Twilio `clear` and stops TTS, except during the protected opener.

Measured TTS latency on this path is about 480 ms to the first audio byte and about 1.0 s for a whole sentence. After the caller stops, expect about 400 ms of endpointing, then a cached backchannel immediately, then the first real sentence once the text model has a short clause (a few hundred milliseconds) plus that 480 ms. Later sentences overlap synthesis with playback, so the gap between them is mostly hidden when a sentence plays for about a second.

Gemini API Tier 1 is 10 requests per minute and 100 per day. That is too low for production. Each uncached sentence is one TTS request, and each caller turn is one text request. A fresh process also spends a handful of requests pre-rendering the opener and the fixed lines; those clips are reused until the process restarts. A short call is on the order of 10–20 Gemini requests. Two calls close together can trip the per-minute cap. A 429 is logged and the caller hears one cached line — "Sorry, I'm having trouble on my side. I'll leave it there. Goodbye." — then the call ends. There is no retry loop. Raise the quota before pointing a live number at this provider. The default provider is still `grok`, so deploying this code does not switch traffic until `VOICE_PROVIDER=gemini`.

Cost, using published rates (check the vendor pages before a campaign; promotions move):

| Piece | Rate | A short call (about 3 minutes, under a minute of assistant speech) |
| --- | --- | --- |
| Deepgram Nova-3 monolingual streaming | $0.0048/min promotional, $0.0077/min regular | about 1–2 cents |
| Gemini 2.5 Flash text | $0.30 / 1M input tokens, $2.50 / 1M output tokens | well under 1 cent |
| Gemini 3.8 Flash-Lite TTS through 31 Dec 2026 | $0.50 / 1M text tokens in, $6 / 1M audio tokens out (25 audio tokens/second, about $0.0015 per 10 seconds) | about 1 cent |
| Same TTS from 1 Jan 2027 | input and output prices double | about 2 cents |

The dollar cost of one test call is a few cents. The daily request cap is what runs out first.

Try the logic with no credentials and no dial:

```bash
python scripts/simulate_gemini.py
```

That runs barge-in, overlapping sentence synthesis, immediate opt-out, and the 429 fallback against fakes.

One real call, only after the quota is raised, to a number you control, during calling hours. Do not change the Twilio number. From a shell with the service already running:

```bash
VOICE_PROVIDER=gemini GEMINI_API_KEY=... DEEPGRAM_API_KEY=... \
CALLER_BASE_URL=http://127.0.0.1:8787 API_BEARER_TOKEN=your-token \
  python scripts/place_call.py \
  --to +61412345678 \
  --name "Sam Lee" \
  --business "Harbour Roasters"
```

Use your own mobile, not a guessed number. `POST /probe/tts` still measures TTS latency without dialling. Set `GEMINI_THINKING_BUDGET=-1` if the text model rejects `thinkingConfig`. `GEMINI_THINKING_BUDGET=0` (the default) skips thinking on models that accept it, which matters on a phone call.

### Gemini Live

`VOICE_PROVIDER=geminilive` is the native speech-to-speech socket, the same shape of call as Grok and GPT-Live. It exists because Flash-Lite TTS on Tier 1 is capped at 100 requests per day, while Gemini Live on that plan is not: unlimited requests per minute and per day, with a 150,000 token-per-minute ceiling. Confirm the current numbers in AI Studio before a campaign. The cascade provider and the default (`grok`) are unchanged. Selecting this does not switch a deployment until `VOICE_PROVIDER=geminilive`.

The model id is `gemini-3.8-live` (AI Studio shows "Gemini 3.8 Live"). That is the stable id in [Google's model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live). The socket is the v1beta Live endpoint in the [Live API WebSocket reference](https://ai.google.dev/api/live):

`wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=GEMINI_API_KEY`

Setup is one JSON message: `models/gemini-3.8-live`, `responseModalities: ["AUDIO"]`, voice `Charon`, `speechConfig.languageCode` `en-AU`. The Australian accent is the system-instruction line `Australian English accent, relaxed and friendly` plus that language code. It is not written into the greeting. `end_call` and `record_opt_out` are function declarations with `behavior: BLOCKING`, so the model waits for the tool result. Prompt assembly, calling hours, the do-not-call list, and the finished-call webhook are the same code as the other providers. Input and output audio transcription are turned on (`languageCodes: ["en-AU"]`) and stored as caller and assistant lines.

Audio bridging, per 20 ms Twilio frame:

* In: 8 kHz G.711 μ-law → 16 kHz little-endian 16-bit PCM (`audio/pcm;rate=16000`).
* Out: 24 kHz little-endian 16-bit PCM → 8 kHz G.711 μ-law, then Twilio media.
* Barge-in: Live sends `serverContent.interrupted`. The relay sends Twilio `clear` and drops queued μ-law, except while the opener is still playing. Caller audio is not forwarded during that opener, so the model does not cut its own greeting.
* The opener is a `clientContent` turn sent after `setupComplete`, asking the model to say the compliance line and nothing else.

No separate STT or TTS request is made, so the Flash-Lite daily cap does not apply. Resampling is local and adds well under a millisecond to a 20 ms frame. The wait after the caller stops is the Live API's own voice-activity silence window plus the model's first audio. Google describes `gemini-3.8-live` as the low-latency default for voice agents. This repo has not measured a live call: no call was placed to check it. Expect that wait to be shorter than the cascade, which adds Deepgram endpointing, a text-model hop, and about 480 ms of TTS before the first real sentence.

Published rates for `gemini-3.8-live` (Gemini Developer API pricing page, 22 October 2026; check it again before a campaign):

| Piece | Rate |
| --- | --- |
| Audio in | $3.00 / 1M tokens, which is $0.005 per minute (25 tokens per second) |
| Audio out | $12.00 / 1M tokens, which is $0.018 per minute |
| Text in | $0.75 / 1M tokens |
| Text out, including transcription | $4.50 / 1M tokens |

A single pass over a 3 minute call is about 1.5 cents of caller audio plus about 1.8 cents for a minute of the assistant's voice. Live re-bills audio that is still in the context on later turns, so a long back-and-forth costs more than that single pass. Setup enables sliding-window context compression so the session does not keep the whole recording. Transcription is a small extra at the text output rate. A short test call is a few cents. Audio-only Live sessions last 15 minutes; this build does not resume them. `goAway` is logged and the socket close ends the call.

`GEMINI_API_KEY` is required. `DEEPGRAM_API_KEY` is not. Grok still does not read the Gemini key.

Gemini sends `setupComplete` and audio as binary WebSocket frames whose payload is UTF-8 JSON. The Live socket decodes those frames. Grok, OpenAI, and GPT-Live still ignore unexpected binary frames.

One real call, only when you mean to dial, to a number you control, during calling hours. Do not change the Twilio number.

```bash
VOICE_PROVIDER=geminilive GEMINI_API_KEY=... \
CALLER_BASE_URL=http://127.0.0.1:8787 API_BEARER_TOKEN=your-token \
  python scripts/place_call.py \
  --to +61412345678 \
  --name "Sam Lee" \
  --business "Harbour Roasters"
```

Use your own mobile, not a guessed number. Codec and tool behaviour without a socket:

```bash
python -m pytest tests/test_pcm_bridge.py tests/test_gemini_live.py
```

## Storage

SQLite at `SQLITE_PATH` (default `data/calls.db`) unless `DATABASE_URL` is set. A Railway `postgres://` URL is rewritten for asyncpg, and `sslmode=require` becomes an SSL connection.

Each call row stores `direction` (`outbound` or `inbound`), `call_sid`, both numbers, AMD result, provider, timestamps, duration, end reason, opt-out, and a transcript with `assistant` and `caller` lines.

When `TRANSCRIPT_WEBHOOK_URL` is set, a finished call is POSTed there once. The body is the call id, sid, direction, both numbers, name, business, notes, provider, status, AMD result, end reason, opt-out, duration, timestamps, and the transcript. Delivery waits until the media stream has closed, runs in the background with a 10 second timeout, and retries up to three times. A delivery failure is logged and does not affect the call. `TRANSCRIPT_WEBHOOK_SECRET` is optional. With the default `TRANSCRIPT_WEBHOOK_AUTH_HEADER=Authorization` it is sent as `Bearer <secret>`; any other header name sends the raw secret. A full header line is also accepted (`Authorization: Bearer <token>`, or `X-Webhook-Token: <token>`). The name is the text before the first colon. The whole line is not used as the header name.

## Environment variables

| Variable | Required to dial | Purpose |
| --- | --- | --- |
| `API_BEARER_TOKEN` | yes | Bearer token for `/calls` |
| `PUBLIC_BASE_URL` | yes | Public `https://` origin Twilio uses for webhooks and `wss://` media |
| `TWILIO_ACCOUNT_SID` | yes | Twilio account |
| `TWILIO_API_KEY_SID` | yes | API key sid (`SK...`), used as HTTP basic auth username |
| `TWILIO_API_KEY_SECRET` | yes | API key secret |
| `TWILIO_FROM_NUMBER` | yes | Caller ID. Use `+61290904464` |
| `VOICE_PROVIDER` | yes | `grok` (default), `openai`, `gptlive`, `gemini`, or `geminilive` |
| `XAI_API_KEY` | when provider is grok | xAI key for `wss://api.x.ai/v1/realtime` |
| `OPENAI_API_KEY` | when provider is openai or gptlive | OpenAI key for Realtime or GPT-Live |
| `TWILIO_AUTH_TOKEN` | no | When set, webhooks must carry a valid `X-Twilio-Signature` |
| `XAI_VOICE` | no | Default `eve` |
| `OPENAI_VOICE` | no | Default `marin` |
| `XAI_REALTIME_MODEL` | no | Default `grok-voice-latest` |
| `OPENAI_REALTIME_MODEL` | no | Default `gpt-realtime` |
| `XAI_REALTIME_URL` | no | Default `wss://api.x.ai/v1/realtime` |
| `OPENAI_REALTIME_URL` | no | Default `wss://api.openai.com/v1/realtime` |
| `OPENAI_TRANSCRIBE_MODEL` | no | Default `gpt-4o-mini-transcribe` |
| `GPTLIVE_URL` | no | Default `wss://api.openai.com/v1/live/sessions`. No `model` query string |
| `GPTLIVE_MODEL` | no | Default `gpt-live-1` |
| `GPTLIVE_VOICE` | no | Default `ripple` |
| `GPTLIVE_RESPONSES_MODEL` | no | Default `gpt-6-luna`. Backend that selects the call tools |
| `XAI_REASONING_EFFORT` | no | Default `none`, to keep phone turns short |
| `USE_FORCE_MESSAGE` | no | Default `true`. Grok opener via `force_message` |
| `DATABASE_URL` | no | Postgres URL. Empty uses SQLite |
| `SQLITE_PATH` | no | Default `data/calls.db` |
| `DO_NOT_CALL_PATH` | no | Default `data/do_not_call.txt` |
| `TEST_NUMBERS_PATH` | no | Default `config/test_numbers.txt`. These numbers skip calling hours only |
| `TEST_NUMBERS` | no | Comma-separated numbers added to that file |
| `DEFAULT_TIMEZONE` | no | Default `Australia/Sydney` |
| `PROMPT_PATH` | no | Default `prompts/system.txt` |
| `OPENER_PATH` | no | Default `prompts/compliance_opener.txt` |
| `INBOUND_PROMPT_PATH` | no | Default `prompts/inbound_system.txt` |
| `INBOUND_LOOKUP_DAYS` | no | Match a caller to an outbound call from the last N days. Default 90. `0` keeps all |
| `MACHINE_MESSAGE` | no | Empty hangs up on machines. Otherwise spoken, then hangup |
| `AMD_HOLD_SECONDS` | no | Unused. The greeting is not held for AMD. Default 60 |
| `MACHINE_DETECTION_TIMEOUT` | no | Twilio `MachineDetectionTimeout`. Default 30 |
| `HANGUP_GRACE_SECONDS` | no | Extra wait after playback before the hangup watchdog. Default 8 |
| `LOG_LEVEL` | no | Default `INFO` |
| `GEMINI_API_KEY` | when provider is gemini or geminilive | Cascade text and TTS, Gemini Live, and `POST /probe/tts`. Grok, OpenAI, and GPT-Live do not read it |
| `GOOGLE_TTS_API_KEY` | no | Admin `POST /probe/tts` with `provider: "cloudtts"` only. Calls do not use it |
| `DEEPGRAM_API_KEY` | when provider is gemini | Streaming speech-to-text for the cascade. Gemini Live does not use it |
| `DEEPGRAM_URL` | no | Default `wss://api.deepgram.com/v1/listen` |
| `DEEPGRAM_MODEL` | no | Default `nova-3` |
| `DEEPGRAM_LANGUAGE` | no | Default `en-AU` |
| `DEEPGRAM_ENDPOINTING_MS` | no | Silence before a caller turn ends. Default 400 |
| `GEMINI_TEXT_MODEL` | no | Default `gemini-2.5-flash` |
| `GEMINI_TTS_MODEL` | no | Default `gemini-3.8-flash-lite-tts` |
| `GEMINI_TTS_VOICE` | no | Default `Charon` |
| `GEMINI_TTS_STYLE` | no | Speech metadata only. Default `Australian English accent, relaxed and friendly` |
| `GEMINI_TTS_LANGUAGE` | no | Default `en-AU` |
| `GEMINI_THINKING_BUDGET` | no | Default `0`. `-1` omits `thinkingConfig`. Cascade only |
| `GEMINILIVE_MODEL` | no | Default `gemini-3.8-live` |
| `GEMINILIVE_URL` | no | Default v1beta `BidiGenerateContent` WebSocket. The API key is added as `?key=` |
| `GEMINILIVE_VOICE` | no | Default `Charon` |
| `GEMINILIVE_LANGUAGE` | no | Default `en-AU`, sent as `speechConfig.languageCode` and as a transcription hint |
| `GEMINILIVE_STYLE` | no | System instruction only. Default `Australian English accent, relaxed and friendly` |
| `TRANSCRIPT_WEBHOOK_URL` | no | Where to POST each finished call. Empty disables the feature |
| `TRANSCRIPT_WEBHOOK_SECRET` | no | Secret sent with that POST. Empty sends no auth header |
| `TRANSCRIPT_WEBHOOK_AUTH_HEADER` | no | Header name, or a full `Name: value` line. Default `Authorization`, which sends `Bearer <secret>`. Any other name sends the raw secret |

The service boots and serves `/health` without these. `POST /calls` returns 503 until the dial settings for the selected provider are present.

## Inbound calls

Set the number's "A call comes in" webhook to `POST {PUBLIC_BASE_URL}/twilio/inbound`. That URL has to be exactly `PUBLIC_BASE_URL` plus `/twilio/inbound`, because signature checks use `PUBLIC_BASE_URL` rather than the request host.

There is no answering-machine detection and no calling-hours check. A do-not-call entry does not block the person from calling us. If they opt out during the call, `record_opt_out` still stores their number and later outbound dials are refused.

The opener is in the inbound prompt. When the caller's number matches an outbound call from the last `INBOUND_LOOKUP_DAYS` (default 90) that was actually placed, the relay is given that call's name, business, and notes. The opener is only `Hello, Tack45, is this {first name}?`. After they confirm, the next turn is exactly `Thanks for calling back. Just so you know, I'm an AI assistant for Tack45.` Refused outbound attempts are not a match. An unknown caller hears `Hello, Tack45. Just so you know, I'm an AI assistant. How can I help?` The AI disclosure is always in the first turn or the second. The same rules apply as outbound: no prices, no client results or ROI, and the call to action is a free 60-minute chat with Adam.

A status callback on the number is optional: `POST {PUBLIC_BASE_URL}/twilio/status`. The relay already stores duration from `answered_at` when the stream ends.

## Twilio configuration

Outbound calls set their own answer, AMD, and status webhooks. Inbound calls use the number's "A call comes in" webhook above.

1. In the Twilio console, confirm `+61290904464` is a voice-capable number on the account that owns `TWILIO_ACCOUNT_SID`. Set "A call comes in" to HTTP POST `{PUBLIC_BASE_URL}/twilio/inbound`.
2. Create a standard API key. Put the sid and secret in `TWILIO_API_KEY_SID` and `TWILIO_API_KEY_SECRET`.
3. Voice geographic permissions must allow calls to Australia.
4. Deploy so `PUBLIC_BASE_URL` is the public `https` origin (no trailing path).
5. The service then calls:
   - `POST {PUBLIC_BASE_URL}/twilio/voice` when the call is answered. The response connects the media stream at once
   - `POST {PUBLIC_BASE_URL}/twilio/amd` with `AnsweredBy`. A later machine or fax result hangs up. `unknown` leaves the live conversation running
   - `POST {PUBLIC_BASE_URL}/twilio/status` for `initiated`, `ringing`, `answered`, `completed`
   - `wss://{host}/streams/media/{token}` for media, starting on answer rather than after AMD

Set `TWILIO_AUTH_TOKEN` (the account auth token, not the API key) if you want webhook signatures checked. Leave it unset in local development.

## TTS latency probe

`POST /probe/tts` is an admin check for speech latency. It uses the same bearer token as `/calls` and does not dial or open a voice session. `provider` is `gemini` (the default) or `cloudtts`. `GEMINI_API_KEY` is optional; without it a Gemini probe returns 503 and startup is unchanged. `GOOGLE_TTS_API_KEY` is optional; without it a Cloud TTS probe returns 400 and startup is unchanged. Calls never read that key.

The body may set `sentences`, `runs_per_sentence` (default 1), `style`, `voice`, `model`, and `save_samples` (default 3). Omitted sentences are about twenty short Tack45 lines. A Gemini probe calls `gemini-3.8-flash-lite-tts` on the Interactions API with `stream: true`, μ-law at 8 kHz, and the style `Australian English accent, relaxed and friendly` unless overridden. The voice goes in `generation_config.speech_config` and defaults to `Kore`. The result records `method: "interactions"`.

A Cloud TTS probe uses Chirp 3 HD. The default voice is `en-AU-Chirp3-HD-Charon` and `languageCode` is `en-AU`. StreamingSynthesize is a bidirectional gRPC method. Its documented credential is an OAuth token for `https://www.googleapis.com/auth/cloud-platform`, and the method is not in the REST discovery document, so an API key cannot reach it. The probe therefore calls REST `text:synthesize` (`method: "text:synthesize"`) with `audioEncoding: MULAW` and `sampleRateHertz: 8000`. The key is sent in the `X-Goog-Api-Key` header and is not logged. That response is one JSON body, so time to the first audio byte and total time are the time until that audio is decoded. MULAW audio includes a WAV header, which is stripped before the clip is stored. If the body is 16-bit PCM instead, it is converted to 8 kHz μ-law and `conversion_ms` on the run is only that conversion.

The JSON reports time to first audio byte and total time for each run, p50/p90/min/max, audio bytes and seconds, and `RAILWAY_REPLICA_REGION` when that environment variable is set. Up to `save_samples` clips are returned as base64 μ-law and kept in memory for `GET /probe/tts/sample/{n}.wav`.

`VOICE_PROVIDER=gemini` reuses this request builder. The call path passes voice `Charon` and language `en-AU` instead of the probe default `Kore`.

## Deploy on Railway

1. Create a Railway project and deploy this repository. Nixpacks uses `.python-version` (3.12) and `railway.toml`.
2. The start command is `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Health check path is `/health`.
3. Optional: add the Postgres plugin. Railway sets `DATABASE_URL` automatically. Without it, the service uses SQLite on the container disk, which does not survive a new deploy.
4. Generate a public domain. Set `PUBLIC_BASE_URL` to `https://<that-domain>` and redeploy so Twilio webhook URLs match.
5. Set the variables in the table above. Minimum for a Grok deployment: `API_BEARER_TOKEN`, `PUBLIC_BASE_URL`, `TWILIO_ACCOUNT_SID`, `TWILIO_API_KEY_SID`, `TWILIO_API_KEY_SECRET`, `TWILIO_FROM_NUMBER=+61290904464`, `VOICE_PROVIDER=grok`, `XAI_API_KEY`.
6. Open `/health`, then place one test call with `scripts/place_call.py` during calling hours to a number you control.

SQLite files on Railway's ephemeral disk are wiped on redeploy. Use Postgres if the do-not-call list and transcripts need to persist. The text file `data/do_not_call.txt` is part of the image; edit it in git or set `DO_NOT_CALL_PATH`. Opt-outs recorded at runtime live in the database.
