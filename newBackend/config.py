# config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from supabase import create_client, Client
except ImportError:
    Client = object
    create_client = None

# 1. Get the absolute path to the directory where config.py lives
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

print("\n" + "="*50)
print("🔍 SYSTEM BOOT START")
print(f"📍 Current Working Dir: {os.getcwd()}")
print(f"📄 Looking for .env at: {ENV_PATH}")

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ENV_PATH)
    print("✅ load_dotenv():       Executed.")
except ImportError:
    print("❌ python-dotenv:       NOT INSTALLED! Run: pip install python-dotenv")
    def load_dotenv(**kwargs) -> None:
        return None
    load_dotenv()

def _split_csv(raw_value: str | None, default: list[str]) -> list[str]:
    if not raw_value:
        return default
    return [item.strip() for item in raw_value.split(",") if item.strip()]

@dataclass(slots=True)
class Settings:
    # Database
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", "").strip())
    supabase_key: str = field(default_factory=lambda: os.getenv("SUPABASE_KEY", "").strip())
    
    # LLMs & AI APIs
    huggingface_api_key: str = field(default_factory=lambda: os.getenv("HUGGINGFACE_API_KEY", "").strip())
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", os.getenv("ElevenLabs", "")).strip())
    
    # Models
    action_model: str = field(default_factory=lambda: os.getenv("ACTION_MODEL", "Qwen/Qwen3-14B"))
    
    # App Settings
    app_domain: str = field(default_factory=lambda: os.getenv("APP_DOMAIN", "healthcare").strip() or "healthcare")
    cors_origins: list[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv("CORS_ORIGINS"),
            ["http://localhost:3000", "http://127.0.0.1:3000"],
        )
    )

    @property
    def has_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

settings = Settings()

# 2. Instantiate the single Supabase Client to be imported across the app
supabase: Client | None = None
if settings.has_supabase and create_client is not None:
    supabase = create_client(settings.supabase_url, settings.supabase_key)
    print("✅ Supabase Client:     Connected.")
else:
    print("⚠️ Supabase Client:     NOT CONNECTED. Check keys.")

print("=" * 50 + "\n")