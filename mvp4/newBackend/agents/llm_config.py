# agents/llm_config.py
from __future__ import annotations

import sys
from pathlib import Path

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import settings


def get_llm(temperature: float = 0.1):
    """Returns a configured ChatHuggingFace instance pointing to the HF Inference API."""
    if not settings.huggingface_api_key:
        print("⚠️ HUGGINGFACE_API_KEY is missing! Check your .env file.")

    safe_temp = max(temperature, 0.01)

    # Split "Qwen/Qwen3-14B:nscale" → repo_id="Qwen/Qwen3-14B", provider="nscale"
    # If there's no colon, provider defaults to "auto" (HF picks the best one)
    if ":" in settings.action_model:
        repo_id, provider = settings.action_model.split(":", 1)
    else:
        repo_id = settings.action_model
        provider = "auto"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        provider=provider,                          # ✅ correct way to set provider
        huggingfacehub_api_token=settings.huggingface_api_key,
        temperature=safe_temp,
        task="text-generation",
        max_new_tokens=512,
        do_sample=True,
    )

    chat_model = ChatHuggingFace(llm=llm)
    return chat_model


# ==========================================
# 🧪 QUICK TEST SCRIPT
# ==========================================
if __name__ == "__main__":
    # --- Show parsed config before connecting ---
    if ":" in settings.action_model:
        repo_id, provider = settings.action_model.split(":", 1)
    else:
        repo_id, provider = settings.action_model, "auto"

    print("\n🔧 Config Check:")
    print(f"   Full setting  : {settings.action_model}")
    print(f"   Repo ID       : {repo_id}")
    print(f"   Provider      : {provider}")
    print(f"   API Key set   : {'✅ Yes' if settings.huggingface_api_key else '❌ No'}")
    print(f"\n⏳ Connecting to Hugging Face API...\n")

    try:
        test_llm = get_llm(temperature=0.1)

        print("📡 Sending test message...")
        response = test_llm.invoke(
            "Hello! Are you online? Please reply with exactly: 'Yes, I am online and ready.'"
        )

        print("✅ SUCCESS! The LLM is connected.")
        print(f"🤖 AI Says: {response.content}")
        print("\n🚀 You are ready to run main.py!")

    except ValueError as e:
        print(f"❌ CONFIG ERROR: {e}")
        print("👉 Check your ACTION_MODEL value in .env")

    except Exception as e:
        error_msg = str(e).lower()

        if "401" in error_msg or "unauthorized" in error_msg:
            print("❌ AUTH ERROR: Your HUGGINGFACE_API_KEY is invalid or expired.")
            print("👉 Get a new key at: https://huggingface.co/settings/tokens")

        elif "403" in error_msg or "forbidden" in error_msg:
            print("❌ ACCESS DENIED: You may not have access to this model.")
            print(f"👉 Request access at: https://huggingface.co/{repo_id}")

        elif "404" in error_msg or "not found" in error_msg:
            print(f"❌ MODEL NOT FOUND: '{repo_id}' doesn't exist or isn't available.")
            print("👉 Double-check the model name on https://huggingface.co/models")

        elif "rate" in error_msg or "429" in error_msg:
            print("❌ RATE LIMITED: Too many requests. Wait a moment and try again.")

        else:
            print(f"❌ ERROR: Could not connect to the LLM.")
            print(f"   Details: {e}")