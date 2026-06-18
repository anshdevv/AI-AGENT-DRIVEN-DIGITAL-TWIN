# agents/llm_config.py
# ─────────────────────────────────────────────────────────────────
# Switch providers via .env:
#   LLM_PROVIDER=groq      → Groq API (qwen/qwen3-32b by default)
#   LLM_PROVIDER=deepseek  → DeepSeek API (deepseek-chat by default)
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations
import os

_PROVIDER    = os.getenv("LLM_PROVIDER",   "groq").strip().lower()
_GROQ_MODEL  = os.getenv("GROQ_MODEL",     "qwen/qwen3-32b").strip()
_DS_MODEL    = "deepseek-v4-flash".strip()

print(f"🤖 [LLMConfig] Provider='{_PROVIDER}'  "
      f"model='{_DS_MODEL if _PROVIDER == 'deepseek' else _GROQ_MODEL}'")


def get_llm(temperature: float = 0.1):
    if _PROVIDER == "deepseek":
        from langchain_openai import ChatOpenAI
        from config import settings
        key = getattr(settings, "deepseek_api_key", None) or os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            print("⚠️  [LLMConfig] DEEPSEEK_API_KEY is missing in .env!")
        return ChatOpenAI(
            model=_DS_MODEL,
            temperature=0,
            api_key=key,
            base_url="https://api.deepseek.com",
            # Lower max_tokens prevents the model rambling into tool-format leaks
            max_tokens=800,
        )
    else:
        from langchain_groq import ChatGroq
        from config import settings
        if not settings.groq_api_key:
            print("⚠️  [LLMConfig] GROQ_API_KEY is missing in .env!")
        return ChatGroq(
            model=_GROQ_MODEL,
            temperature=temperature,
            api_key=settings.groq_api_key,
        )


if __name__ == "__main__":
    try:
        llm = get_llm()
        r   = llm.invoke("Reply with: 'online'")
        print(f"✅ {_PROVIDER} → {r.content[:60]}")
    except Exception as e:
        print(f"❌ {e}")