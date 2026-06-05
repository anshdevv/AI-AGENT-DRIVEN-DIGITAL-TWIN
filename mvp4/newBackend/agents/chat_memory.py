# agents/chat_memory.py
# ─────────────────────────────────────────────────────────────────
# Persistent chat history with vector search.
#
# Every user + assistant message is embedded and stored in Supabase.
# On each supervisor turn, the current user message is embedded and
# the most semantically relevant past messages are retrieved and
# injected into the system prompt as context.
#
# Embedding model: all-MiniLM-L6-v2 via sentence-transformers
#   - 22MB download on first use
#   - 384 dimensions
#   - ~5ms per embedding on CPU
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

_embed_model = None
_model_load_attempted = False


def _get_embed_model():
    """Lazy-load the embedding model. Prints a warning if unavailable."""
    global _embed_model, _model_load_attempted
    if _model_load_attempted:
        return _embed_model
    _model_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ [ChatMemory] Embedding model loaded (all-MiniLM-L6-v2)")
    except ImportError:
        print(
            "⚠️  [ChatMemory] sentence-transformers not installed.\n"
            "   Run: pip install sentence-transformers --break-system-packages\n"
            "   Chat history will be stored without embeddings (no semantic search)."
        )
    except Exception as e:
        print(f"⚠️  [ChatMemory] Could not load embedding model: {e}")
    return _embed_model


def _embed(text: str) -> list[float] | None:
    """Return a 384-dim embedding or None if model is unavailable."""
    model = _get_embed_model()
    if model is None:
        return None
    try:
        return model.encode(text, show_progress_bar=False).tolist()
    except Exception as e:
        print(f"⚠️  [ChatMemory] Embedding failed: {e}")
        return None


def store_message(
    session_id: str,
    patient_id: int | None,
    role: str,
    content: str,
) -> None:
    """
    Persist a single message to chat_history.
    Safe to call even if Supabase or the embedding model is unavailable.
    role: 'user' or 'assistant'
    """
    if not content or not content.strip():
        return
    try:
        from config import supabase
        if not supabase:
            return
        row: dict = {
            "session_id": session_id,
            "patient_id": patient_id,
            "role":        role,
            "content":     content.strip(),
        }
        vec = _embed(content)
        if vec is not None:
            row["embedding"] = vec
        supabase.table("chat_history").insert(row).execute()
    except Exception as e:
        print(f"⚠️  [ChatMemory] store_message failed: {e}")


def backfill_session_patient_id(session_id: str, patient_id: int) -> None:
    """
    When patient_id is first resolved (after lookup_customer_profile),
    retroactively set patient_id on all earlier rows from this session
    that were stored with patient_id=NULL.
    This ensures the initial complaint and early messages are searchable.
    """
    if not session_id or not patient_id:
        return
    try:
        from config import supabase
        if not supabase:
            return
        result = (
            supabase.table("chat_history")
            .update({"patient_id": patient_id})
            .eq("session_id", session_id)
            .is_("patient_id", "null")
            .execute()
        )
        count = len(result.data) if result.data else 0
        if count:
            print(f"🧠 [ChatMemory] Backfilled patient_id={patient_id} on {count} early message(s)")
    except Exception as e:
        print(f"⚠️  [ChatMemory] backfill_session_patient_id failed: {e}")


def search_relevant(
    patient_id: int,
    query: str,
    top_k: int = 3,
) -> str:
    """
    Find the most semantically relevant past messages for this patient.
    Returns a formatted string ready to inject into the system prompt,
    or empty string if nothing relevant is found.
    """
    if not patient_id or not query:
        return ""
    vec = _embed(query)
    if vec is None:
        return ""
    try:
        from config import supabase
        if not supabase:
            return ""
        result = supabase.rpc("match_chat_history", {
            "p_patient_id":    patient_id,
            "query_embedding": vec,
            "match_count":     top_k,
            "min_similarity":  0.35,
        }).execute()
        if not result.data:
            return ""
        lines = ["── RELEVANT PAST CONTEXT (from this patient's history) ──"]
        for row in result.data:
            role_label = "Patient" if row["role"] == "user" else "Assistant"
            snippet    = row["content"][:250].replace("\n", " ")
            sim        = row.get("similarity", 0)
            lines.append(f"  [{role_label} | sim={sim:.2f}]: {snippet}")
        lines.append("─────────────────────────────────────────────────────")
        return "\n".join(lines)
    except Exception as e:
        print(f"⚠️  [ChatMemory] search_relevant failed: {e}")
        return ""