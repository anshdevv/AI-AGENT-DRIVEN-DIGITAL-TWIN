# agents/policy_rag.py
# ─────────────────────────────────────────────────────────────────
# RAG lookup for hospital policy.
#
# Loads policy_store.json from the project root, embeds all chunks
# once on first use, then answers queries via cosine similarity.
#
# The chunks are embedded with the same all-MiniLM-L6-v2 model used
# in chat_memory.py — no extra dependencies needed.
#
# Place policy_store.json in your project root (same folder as main.py).
# Edit the "chunks" array to match your actual hospital policy.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import math
import os
from pathlib import Path

_embeddings: list[list[float]] | None = None
_chunks:     list[dict]                = []
_loaded:     bool                      = False

_POLICY_FILE = Path(__file__).resolve().parents[1] / "policy_store.json"


def _cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    mag  = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / mag if mag else 0.0


def _load() -> None:
    global _embeddings, _chunks, _loaded
    if _loaded:
        return
    _loaded = True
    if not _POLICY_FILE.exists():
        print(f"⚠️  [PolicyRAG] {_POLICY_FILE} not found — policy lookup disabled")
        return
    try:
        data   = json.loads(_POLICY_FILE.read_text(encoding="utf-8"))
        chunks = data.get("chunks", [])
        if not chunks:
            return
        from agents.chat_memory import _embed
        _chunks     = chunks
        _embeddings = [_embed(c["text"]) or [] for c in chunks]
        print(f"✅ [PolicyRAG] Loaded {len(chunks)} policy chunks from {_POLICY_FILE.name}")
    except Exception as e:
        print(f"⚠️  [PolicyRAG] Load failed: {e}")


def search_policy(query: str, top_k: int = 2, threshold: float = 0.35) -> str:
    """
    Find the most relevant policy chunks for a query.
    Returns a formatted string for injection into the prompt, or "" if nothing relevant.
    """
    _load()
    if not _embeddings or not _chunks:
        return ""
    try:
        from agents.chat_memory import _embed
        qvec = _embed(query)
        if not qvec:
            return ""
        scored = [
            (i, _cosine(qvec, ev))
            for i, ev in enumerate(_embeddings)
            if ev
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [(i, s) for i, s in scored[:top_k] if s >= threshold]
        if not top:
            return ""
        lines = ["── HOSPITAL POLICY (relevant to this query) ──"]
        for i, sim in top:
            c = _chunks[i]
            lines.append(f"  [{c.get('topic', 'Policy')} | sim={sim:.2f}]")
            lines.append(f"  {c['text']}")
        lines.append("──────────────────────────────────────────────")
        return "\n".join(lines)
    except Exception as e:
        print(f"⚠️  [PolicyRAG] Search failed: {e}")
        return ""