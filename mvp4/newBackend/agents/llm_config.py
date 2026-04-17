# agents/llm_config.py
from __future__ import annotations

import sys
from pathlib import Path

from langchain_groq import ChatGroq

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import settings

def get_llm(temperature: float = 0.1):
    """Returns a configured ChatGroq instance pointing to Groq's API."""
    if not settings.groq_api_key:
        print("⚠️ GROQ_API_KEY is missing! Check your .env file.")
        
    # The official Groq model ID for Qwen 3 32B is "qwen/qwen3-32b"
    return ChatGroq(
        model="qwen/qwen3-32b", 
        temperature=temperature,
        api_key=settings.groq_api_key,
    )

# ==========================================
# 🧪 QUICK TEST SCRIPT
# ==========================================
if __name__ == "__main__":
    print("\n🔧 Config Check (Groq):")
    print(f"   Model Target  : qwen/qwen3-32b")
    print(f"   API Key set   : {'✅ Yes' if settings.groq_api_key else '❌ No'}")
    print(f"\n⏳ Connecting to Groq API...\n")

    try:
        test_llm = get_llm(temperature=0.1)

        print("📡 Sending test message...")
        response = test_llm.invoke(
            "Hello! Are you online? Please reply with exactly: 'Yes, I am online and ready.'"
        )

        print("✅ SUCCESS! The LLM is connected.")
        print(f"🤖 AI Says: {response.content}")
        print("\n🚀 You are ready to run main.py!")

    except Exception as e:
        error_msg = str(e).lower()
        
        if "401" in error_msg or "unauthorized" in error_msg:
            print("❌ AUTH ERROR: Your GROQ_API_KEY is invalid or expired.")
            print("👉 Get a new key at: https://console.groq.com/keys")

        elif "404" in error_msg or "not found" in error_msg:
            print("❌ MODEL NOT FOUND: The requested model doesn't exist on Groq.")
            print("👉 Check the Groq console for supported model IDs.")

        elif "rate" in error_msg or "429" in error_msg:
            print("❌ RATE LIMITED: Too many requests. Wait a moment and try again.")

        else:
            print(f"❌ ERROR: Could not connect to the LLM.")
            print(f"   Details: {e}")