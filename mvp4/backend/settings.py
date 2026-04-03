from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional runtime dependency
    def load_dotenv() -> None:
        return None


load_dotenv()


def _split_csv(raw_value: str | None, default: list[str]) -> list[str]:
    if not raw_value:
        return default
    values = [item.strip() for item in raw_value.split(",")]
    return [item for item in values if item]


@dataclass(slots=True)
class Settings:
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", "").strip())
    supabase_key: str = field(default_factory=lambda: os.getenv("SUPABASE_KEY", "").strip())
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", "").strip())
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS", os.getenv("ElevenLabs", "")).strip())
    app_domain: str = field(default_factory=lambda: os.getenv("APP_DOMAIN", "healthcare").strip() or "healthcare")
    classifier_model: str = field(default_factory=lambda: os.getenv("CLASSIFIER_MODEL", "gemini-2.5-flash"))
    action_model: str = field(default_factory=lambda: os.getenv("ACTION_MODEL", "gemini-2.5-flash"))
    triage_model: str = field(default_factory=lambda: os.getenv("TRIAGE_MODEL", "gemini-2.5-flash"))
    cors_origins: list[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv("CORS_ORIGINS"),
            ["http://localhost:3000", "http://127.0.0.1:3000"],
        )
    )
    cors_origin_regex: str = field(
        default_factory=lambda: os.getenv(
            "CORS_ORIGIN_REGEX",
            r"^https?://(?:localhost|127\.0\.0\.1|\[::1\]|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?::\d+)?$",
        ).strip()
    )

    @property
    def has_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    @property
    def has_google(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)


settings = Settings()
