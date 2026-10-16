# tack45-caller

Outbound AI voice calls for Tack45, a Sydney AI consultancy. The service dials an Australian business on a Twilio number, waits for async answering-machine detection, and on a human answer bridges the call to a realtime speech-to-speech model. Audio stays 8 kHz G.711 μ-law from Twilio to the model and back. Nothing is resampled or transcoded.

The sales script is a placeholder. Compliance behaviour is already in place: a fixed opener, an Australian accent instruction, calling-hours and do-not-call guards, and an opt-out tool.

No live Twilio or model calls are made by the tests. `scripts/place_call.py` is the only way to dial, and it does nothing until you point it at a running service that has credentials.

## Call flow

1. `POST /calls` checks the bearer token, the do-not-call list, and the callee's local calling window, then creates a Twilio call.
2. Twilio dials `TWILIO_FROM_NUMBER` (intended to be `+61290904464`) with `MachineDetection=DetectMessageEnd` and async AMD.
3. While detection runs, the answer webhook returns `<Pause>`.
4. Human: the call is updated to `<Connect><Stream>` and the media WebSocket opens.
5. Machine: hang up, or play `MACHINE_MESSAGE` and then hang up. Fax and unknown: hang up. Each result is stored.
6. The relay forwards μ-law payloads unchanged. Barge-in sends Twilio `clear` and drops audio still queued here. `end_call` hangs up via the Twilio REST API after the current audio finishes.

```
phone  <->  Twilio Media Streams (audio/pcmu, 8 kHz)
                <->  tack45-caller
                        <->  wss://api.x.ai/v1/realtime   (VOICE_PROVIDER=grok)
                        <->  wss://api.openai.com/v1/realtime  (VOICE_PROVIDER=openai)
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

`timezone` is optional. `GET /calls/{id}` returns the AMD result, timestamps, duration, transcript, opt-out flag, and end reason.

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

## Voice providers

`VOICE_PROVIDER=grok` (default) or `openai`.

Both sessions request `audio/pcmu` in and out, `server_vad`, and the tools `end_call` and `record_opt_out`. Grok also enables `grok-transcribe` on input and, unless `USE_FORCE_MESSAGE=false`, speaks the compliance opener with xAI `force_message` (verbatim TTS, not a model-written line). OpenAI has no `force_message`; the opener is requested with `response.create` instructions. See the [xAI voice agent guide](https://docs.x.ai/docs/guides/voice/agent) and the [phone cookbook](https://docs.x.ai/cookbook/examples/voice-agent-phone).

The system prompt is `prompts/system.txt`. The opener is `prompts/compliance_opener.txt`. Replace the placeholder sales section when the real script is ready, and keep the accent, opt-out, and `end_call` rules.

Barge-in is suppressed while the opener is still playing, so the disclosure is not cut off. After that, caller speech sends Twilio `{"event":"clear"}` and drops queued μ-law. When the model calls `end_call`, the relay waits until Twilio echoes a mark (playback finished), then `POST`s `Status=completed` on the call. A watchdog hangs up if the mark never returns.

## Storage

SQLite at `SQLITE_PATH` (default `data/calls.db`) unless `DATABASE_URL` is set. A Railway `postgres://` URL is rewritten for asyncpg, and `sslmode=require` becomes an SSL connection.

Each call row stores `call_sid`, destination, AMD result, provider, timestamps, duration, end reason, opt-out, and a transcript with `assistant` and `caller` lines.

## Environment variables

| Variable | Required to dial | Purpose |
| --- | --- | --- |
| `API_BEARER_TOKEN` | yes | Bearer token for `/calls` |
| `PUBLIC_BASE_URL` | yes | Public `https://` origin Twilio uses for webhooks and `wss://` media |
| `TWILIO_ACCOUNT_SID` | yes | Twilio account |
| `TWILIO_API_KEY_SID` | yes | API key sid (`SK...`), used as HTTP basic auth username |
| `TWILIO_API_KEY_SECRET` | yes | API key secret |
| `TWILIO_FROM_NUMBER` | yes | Caller ID. Use `+61290904464` |
| `VOICE_PROVIDER` | yes | `grok` or `openai` |
| `XAI_API_KEY` | when provider is grok | xAI key for `wss://api.x.ai/v1/realtime` |
| `OPENAI_API_KEY` | when provider is openai | OpenAI key for the Realtime WebSocket |
| `TWILIO_AUTH_TOKEN` | no | When set, webhooks must carry a valid `X-Twilio-Signature` |
| `XAI_VOICE` | no | Default `eve` |
| `OPENAI_VOICE` | no | Default `marin` |
| `XAI_REALTIME_MODEL` | no | Default `grok-voice-latest` |
| `OPENAI_REALTIME_MODEL` | no | Default `gpt-realtime` |
| `XAI_REALTIME_URL` | no | Default `wss://api.x.ai/v1/realtime` |
| `OPENAI_REALTIME_URL` | no | Default `wss://api.openai.com/v1/realtime` |
| `OPENAI_TRANSCRIBE_MODEL` | no | Default `gpt-4o-mini-transcribe` |
| `XAI_REASONING_EFFORT` | no | Default `none`, to keep phone turns short |
| `USE_FORCE_MESSAGE` | no | Default `true`. Grok opener via `force_message` |
| `DATABASE_URL` | no | Postgres URL. Empty uses SQLite |
| `SQLITE_PATH` | no | Default `data/calls.db` |
| `DO_NOT_CALL_PATH` | no | Default `data/do_not_call.txt` |
| `DEFAULT_TIMEZONE` | no | Default `Australia/Sydney` |
| `PROMPT_PATH` | no | Default `prompts/system.txt` |
| `OPENER_PATH` | no | Default `prompts/compliance_opener.txt` |
| `MACHINE_MESSAGE` | no | Empty hangs up on machines. Otherwise spoken, then hangup |
| `AMD_HOLD_SECONDS` | no | Pause while AMD runs. Default 60, and at least detection timeout + 15 |
| `MACHINE_DETECTION_TIMEOUT` | no | Twilio `MachineDetectionTimeout`. Default 30 |
| `HANGUP_GRACE_SECONDS` | no | Extra wait after playback before the hangup watchdog. Default 8 |
| `LOG_LEVEL` | no | Default `INFO` |

The service boots and serves `/health` without these. `POST /calls` returns 503 until the dial settings for the selected provider are present.

## Twilio configuration

Outbound calls set their own webhooks. You do not need a "call comes in" URL on the number for this service.

1. In the Twilio console, confirm `+61290904464` is a voice-capable number on the account that owns `TWILIO_ACCOUNT_SID`.
2. Create a standard API key. Put the sid and secret in `TWILIO_API_KEY_SID` and `TWILIO_API_KEY_SECRET`.
3. Voice geographic permissions must allow calls to Australia.
4. Deploy so `PUBLIC_BASE_URL` is the public `https` origin (no trailing path).
5. The service then calls:
   - `POST {PUBLIC_BASE_URL}/twilio/voice` when the call is answered
   - `POST {PUBLIC_BASE_URL}/twilio/amd` with `AnsweredBy`
   - `POST {PUBLIC_BASE_URL}/twilio/status` for `initiated`, `ringing`, `answered`, `completed`
   - `wss://{host}/streams/media?token=...` for media, only after a human answer

Set `TWILIO_AUTH_TOKEN` (the account auth token, not the API key) if you want webhook signatures checked. Leave it unset in local development.

## Deploy on Railway

1. Create a Railway project and deploy this repository. Nixpacks uses `.python-version` (3.12) and `railway.toml`.
2. The start command is `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Health check path is `/health`.
3. Optional: add the Postgres plugin. Railway sets `DATABASE_URL` automatically. Without it, the service uses SQLite on the container disk, which does not survive a new deploy.
4. Generate a public domain. Set `PUBLIC_BASE_URL` to `https://<that-domain>` and redeploy so Twilio webhook URLs match.
5. Set the variables in the table above. Minimum for a Grok deployment: `API_BEARER_TOKEN`, `PUBLIC_BASE_URL`, `TWILIO_ACCOUNT_SID`, `TWILIO_API_KEY_SID`, `TWILIO_API_KEY_SECRET`, `TWILIO_FROM_NUMBER=+61290904464`, `VOICE_PROVIDER=grok`, `XAI_API_KEY`.
6. Open `/health`, then place one test call with `scripts/place_call.py` during calling hours to a number you control.

SQLite files on Railway's ephemeral disk are wiped on redeploy. Use Postgres if the do-not-call list and transcripts need to persist. The text file `data/do_not_call.txt` is part of the image; edit it in git or set `DO_NOT_CALL_PATH`. Opt-outs recorded at runtime live in the database.
