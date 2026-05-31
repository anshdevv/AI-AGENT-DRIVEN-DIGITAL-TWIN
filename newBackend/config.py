# config.py
from __future__ import annotations

import os
import sys
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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct:fastest").strip())
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", os.getenv("ElevenLabs", "")).strip())
    elevenlabs_stt_model: str = field(default_factory=lambda: os.getenv("ELEVENLABS_STT_MODEL", "scribe_v2").strip())
    elevenlabs_stt_language_hint: str = field(default_factory=lambda: os.getenv("ELEVENLABS_STT_LANGUAGE_HINT", "ur").strip().lower())
    elevenlabs_voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "nPczCjzI2devNBz1zQrb").strip())
    elevenlabs_tts_model: str = field(default_factory=lambda: os.getenv("ELEVENLABS_TTS_MODEL", "eleven_turbo_v2_5").strip())
    elevenlabs_tts_fallback_model: str = field(default_factory=lambda: os.getenv("ELEVENLABS_TTS_FALLBACK_MODEL", "eleven_turbo_v2_5").strip())
    elevenlabs_output_format: str = field(default_factory=lambda: os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128").strip())
    
    # Models (UPDATED TO DEFAULT TO QWEN 14B)
    action_model: str = field(default_factory=lambda: os.getenv("ACTION_MODEL", "Qwen/Qwen2.5-14B-Instruct"))
    
    # App Settings
    app_domain: str = field(default_factory=lambda: os.getenv("APP_DOMAIN", "healthcare").strip() or "healthcare")
    admin_username: str = field(default_factory=lambda: os.getenv("ADMIN_USERNAME", "admin").strip() or "admin")
    admin_password: str = field(default_factory=lambda: os.getenv("ADMIN_PASSWORD", "admin123").strip() or "admin123")
    csr_username: str = field(default_factory=lambda: os.getenv("CSR_USERNAME", "csr").strip() or "csr")
    csr_password: str = field(default_factory=lambda: os.getenv("CSR_PASSWORD", "csr123").strip() or "csr123")
    doctor_portal_password: str = field(
        default_factory=lambda: os.getenv("DOCTOR_PORTAL_PASSWORD", os.getenv("DOCTOR_PASSWORD", "doctor123")).strip()
        or "doctor123"
    )
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
