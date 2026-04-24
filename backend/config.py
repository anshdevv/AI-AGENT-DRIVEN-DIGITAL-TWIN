from __future__ import annotations

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover - optional runtime dependency
    Client = object  # type: ignore[assignment]
    create_client = None

from .settings import settings


supabase: Client | None = None
GOOGLE_API_KEY = settings.google_api_key
OPENROUTER_API_KEY = ""

if settings.has_supabase and create_client is not None:
    supabase = create_client(settings.supabase_url, settings.supabase_key)
