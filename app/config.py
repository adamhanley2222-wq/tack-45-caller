"""Environment configuration."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Railway injects these as environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_bearer_token: str = ""
    public_base_url: str = ""

    twilio_account_sid: str = ""
    twilio_api_key_sid: str = ""
    twilio_api_key_secret: str = ""
    # Optional. When set, Twilio webhooks must carry a valid X-Twilio-Signature.
    twilio_auth_token: str = ""
    twilio_from_number: str = "+61290904464"

    voice_provider: str = "grok"
    xai_api_key: str = ""
    openai_api_key: str = ""
    xai_voice: str = "eve"
    openai_voice: str = "marin"
    xai_realtime_model: str = "grok-voice-latest"
    openai_realtime_model: str = "gpt-realtime"
    xai_realtime_url: str = "wss://api.x.ai/v1/realtime"
    openai_realtime_url: str = "wss://api.openai.com/v1/realtime"
    openai_transcribe_model: str = "gpt-4o-mini-transcribe"
    # GPT-Live (VOICE_PROVIDER=gptlive). Off unless that provider is selected.
    gptlive_url: str = "wss://api.openai.com/v1/live/sessions"
    gptlive_model: str = "gpt-live-1"
    gptlive_voice: str = "ripple"
    # Responses backend that selects end_call / record_opt_out. The Live
    # delegation guide names gpt-6-luna; the WebSocket guide's example uses
    # gpt-5.6-luna. Override if session start returns responses_validation_failed.
    gptlive_responses_model: str = "gpt-6-luna"
    # Phone calls should not wait on a long reasoning pass.
    xai_reasoning_effort: str = "none"
    use_force_message: bool = True

    database_url: str = ""
    sqlite_path: str = "data/calls.db"
    do_not_call_path: str = "data/do_not_call.txt"
    test_numbers_path: str = "config/test_numbers.txt"
    # Comma-separated E.164 numbers, added to the test-number file.
    test_numbers: str = ""
    default_timezone: str = "Australia/Sydney"

    prompt_path: str = "prompts/system.txt"
    opener_path: str = "prompts/compliance_opener.txt"
    # Outbound only. The session still opens immediately. The opener waits
    # until the person called has spoken and then been quiet, or until the
    # max wait if they never speak. Inbound calls ignore this. Same names as
    # the opener-wait change so the two stay compatible.
    opener_wait_for_speech: bool = True
    opener_silence_ms: int = 500
    opener_max_wait_ms: int = 2500
    inbound_prompt_path: str = "prompts/inbound_system.txt"
    briefs_dir: str = "prompts/briefs"
    # How far back an inbound caller is matched to an outbound call. 0 keeps all.
    inbound_lookup_days: int = 90
    # Empty means hang up on an answering machine. Any text is spoken, then the call ends.
    machine_message: str = ""
    # No longer delays the greeting. The stream starts on answer; AMD hangs up machines later.
    amd_hold_seconds: int = 60
    machine_detection_timeout: int = 30
    hangup_grace_seconds: float = 8.0

    log_level: str = "INFO"

    # POST /probe/tts, the cascade (VOICE_PROVIDER=gemini), and Gemini Live
    # (VOICE_PROVIDER=geminilive). Grok, OpenAI, and GPT-Live do not read it.
    gemini_api_key: str = ""
    # POST /probe/tts with provider "cloudtts", and VOICE_PROVIDER=pipeline.
    # Preferred credential: the full service-account JSON key. Not logged.
    google_tts_credentials_json: str = Field(default="", repr=False)
    # Fallback only when GOOGLE_TTS_CREDENTIALS_JSON is empty.
    # Chirp 3 HD rejects API keys with 401.
    google_tts_api_key: str = ""
    # Same service-account JSON, used by VOICE_PROVIDER=pipeline when
    # GOOGLE_TTS_CREDENTIALS_JSON is empty. Not logged.
    google_cloud_credentials_json: str = Field(default="", repr=False)
    # VOICE_PROVIDER=pipeline. Gemini text model, one streaming request per turn.
    pipeline_llm_model: str = "gemini-2.5-flash"
    pipeline_tts_voice: str = "en-AU-Chirp3-HD-Charon"
    # Speech-to-Text v2. telephony + global accepts 8 kHz μ-law en-AU.
    pipeline_stt_model: str = "telephony"
    pipeline_stt_language: str = "en-AU"
    pipeline_stt_location: str = "global"
    # Silence after the caller stops before the final transcript closes the turn.
    pipeline_stt_endpoint_ms: int = 500
    # Cascaded Gemini provider. Ignored unless VOICE_PROVIDER=gemini.
    deepgram_api_key: str = ""
    deepgram_url: str = "wss://api.deepgram.com/v1/listen"
    deepgram_model: str = "nova-3"
    deepgram_language: str = "en-AU"
    # Silence, in milliseconds, before Deepgram ends a caller turn.
    deepgram_endpointing_ms: int = 400
    gemini_text_model: str = "gemini-2.5-flash"
    gemini_tts_model: str = "gemini-3.8-flash-lite-tts"
    gemini_tts_voice: str = "Charon"
    gemini_tts_style: str = "Australian English accent, relaxed and friendly"
    gemini_tts_language: str = "en-AU"
    # 0 disables thinking on models that accept thinkingConfig. -1 omits the field.
    gemini_thinking_budget: int = 0
    # Gemini Live (VOICE_PROVIDER=geminilive). Native audio over one WebSocket.
    # The accent string is system-instruction text, not words the model speaks.
    geminilive_model: str = "gemini-3.8-live"
    geminilive_url: str = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    )
    geminilive_voice: str = "Charon"
    geminilive_language: str = "en-AU"
    geminilive_style: str = "Australian English accent, relaxed and friendly"

    # Empty disables the finished-call webhook.
    transcript_webhook_url: str = ""
    transcript_webhook_secret: str = ""
    transcript_webhook_auth_header: str = "Authorization"


def missing_dial_configuration(settings: Settings) -> list[str]:
    """Names of settings that must be present before a call is placed."""
    missing: list[str] = []
    if not settings.api_bearer_token:
        missing.append("API_BEARER_TOKEN")
    if not settings.public_base_url:
        missing.append("PUBLIC_BASE_URL")
    if not settings.twilio_account_sid:
        missing.append("TWILIO_ACCOUNT_SID")
    if not settings.twilio_api_key_sid:
        missing.append("TWILIO_API_KEY_SID")
    if not settings.twilio_api_key_secret:
        missing.append("TWILIO_API_KEY_SECRET")
    if not settings.twilio_from_number:
        missing.append("TWILIO_FROM_NUMBER")
    provider = settings.voice_provider.strip().lower()
    if provider == "grok" and not settings.xai_api_key:
        missing.append("XAI_API_KEY")
    elif provider == "openai" and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    elif provider == "gptlive" and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    elif provider == "gemini":
        if not settings.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if not settings.deepgram_api_key:
            missing.append("DEEPGRAM_API_KEY")
    elif provider == "geminilive":
        if not settings.gemini_api_key:
            missing.append("GEMINI_API_KEY")
    elif provider == "pipeline":
        if not settings.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if not (settings.google_tts_credentials_json or "").strip() and not (
            settings.google_cloud_credentials_json or ""
        ).strip():
            missing.append("GOOGLE_TTS_CREDENTIALS_JSON")
    elif provider not in {"grok", "openai", "gptlive", "gemini", "geminilive", "pipeline"}:
        missing.append("VOICE_PROVIDER")
    return missing
