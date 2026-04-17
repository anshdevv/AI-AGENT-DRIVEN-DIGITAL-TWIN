# agents/llm_config.py
from __future__ import annotations

import sys
from pathlib import Path

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Adjust path to import config from the root directory
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import settings


def get_llm(temperature: float = 0.1):
    """Returns a configured ChatHuggingFace instance pointing to the HF Inference API."""
    if not settings.huggingface_api_key:
        print("⚠️ HUGGINGFACE_API_KEY is missing! Check your .env file.")

    # HF Inference API can sometimes throw errors with a temperature of exactly 0.0
    safe_temp = max(temperature, 0.01)

    # 1. Connect to the Hugging Face Endpoint
    llm = HuggingFaceEndpoint(
        repo_id=settings.action_model,
        huggingfacehub_api_token=settings.huggingface_api_key,
        temperature=safe_temp,
        task="text-generation",
        max_new_tokens=512,
        do_sample=True,
    )
    
    # 2. Wrap it for Chat interactions (System/Human/AI messages)
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model