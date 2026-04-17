# agents/llm_config.py
from __future__ import annotations

import sys
from pathlib import Path

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Adjust path to import config from the root directory
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import settings

# In agents/llm_config.py
def get_llm(temperature: float = 0.1, custom_model: str = None):
    # Use the custom model if provided, otherwise default to settings
    repo = custom_model or settings.action_model
    
    safe_temp = max(temperature, 0.01)

    llm = HuggingFaceEndpoint(
        repo_id=repo,
        huggingfacehub_api_token=settings.huggingface_api_key,
        temperature=safe_temp,
        task="text-generation",
        max_new_tokens=512,
        do_sample=True,
    )
    
    return ChatHuggingFace(llm=llm)