"""Environment configuration."""

from __future__ import annotations

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
    elif provider not in {"grok", "openai", "gptlive"}:
        missing.append("VOICE_PROVIDER")
    return missing
