from __future__ import annotations

import json
import re
from typing import Any

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional runtime dependency
    genai = None

from .settings import settings


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class ActionLLM:
    def __init__(self) -> None:
        self.enabled = settings.has_google and genai is not None
        if self.enabled:
            genai.configure(api_key=settings.google_api_key)

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 350,
    ) -> str | None:
        if not self.enabled:
            return None

        active_model = model or settings.action_model
        try:
            response = genai.GenerativeModel(active_model).generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
        except Exception:
            return None

        text = getattr(response, "text", "") or ""
        cleaned = text.strip()
        return cleaned or None

    def complete_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        fallback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fallback = fallback or {}
        text = self.complete(prompt, model=model, temperature=temperature, max_tokens=250)
        if not text:
            return fallback

        match = JSON_RE.search(text)
        candidate = match.group(0) if match else text
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return fallback


llm = ActionLLM()
