from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 1. Get the absolute path to the directory where settings.py lives (the backend folder)
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

print("\n" + "="*50)
print("🔍 SETTINGS DEBUG START")
print(f"📍 Current Working Dir: {os.getcwd()}")
print(f"📄 Looking for .env at: {ENV_PATH}")
print(f"❓ Does .env exist?:    {ENV_PATH.exists()}")

try:
    from dotenv import load_dotenv
    print("📦 python-dotenv:       Installed successfully.")
    # 2. Force it to load from the exact path we found above
    load_dotenv(dotenv_path=ENV_PATH)
    print("✅ load_dotenv():       Executed.")
except ImportError:  # Changed to ImportError to be more specific
    print("❌ python-dotenv:       NOT INSTALLED! Run: pip install python-dotenv")
    def load_dotenv(**kwargs) -> None:
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

print("-" * 50)
print(f"🔑 Supabase URL loaded? {bool(settings.supabase_url)}")
print(f"🔑 Supabase Key loaded? {bool(settings.supabase_key)}")
print(f"🔑 Google Key loaded?   {bool(settings.google_api_key)}")
print(f"🛠️  has_supabase:        {settings.has_supabase}")
print("=" * 50 + "\n")