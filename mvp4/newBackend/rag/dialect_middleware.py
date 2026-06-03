"""
rag/dialect_middleware.py
─────────────────────────
Hybrid FAISS + BM25 RAG normalization layer — used exclusively by triage_agent.

Purpose:
    When a patient's message contains dialectal Urdu, Roman Urdu, or code-mixed
    text, standard clinical NLP models misclassify or miss symptom terms entirely.
    This middleware:
        1. Detects whether the incoming text is English / Roman Urdu / Urdu script.
        2. Strips grammatical filler so only content words reach the index.
        3. Runs a hybrid FAISS (semantic) + BM25 (lexical) search with RRF fusion
           against the dialect_dictionary.csv.
        4. Returns a formatted context string that is injected directly into
           MedGemma's system prompt inside triage_node.
        5. Logs any content-word tokens that could NOT be matched above the
           semantic threshold to logs/unknown_terms.jsonl for clinician review.

NOT used by orchestrator, booking, or voice agents.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Optional heavy deps (gracefully disabled if unavailable) ──────────────────
try:
    import faiss
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE            = Path(__file__).resolve().parent
DICT_PATH        = _HERE.parent / "data" / "dialect_dictionary.csv"
UNKNOWN_LOG_PATH = _HERE.parent / "logs" / "unknown_terms.jsonl"

# ── Model + search config ─────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_THRESH = 0.55   # minimum cosine similarity to count as a match
BM25_MIN_SCORE  = 0.01   # any positive BM25 score counts
TOP_K           = 5      # max hints returned to triage prompt
MIN_TOKEN_LEN   = 3      # ignore tokens shorter than this (noise)
RRF_K           = 60     # reciprocal rank fusion constant

# ── Roman Urdu grammar markers (stripped as stopwords) ────────────────────────
ROMAN_URDU_MARKERS: set[str] = {
    "mein", "hai", "hoon", "ka", "ki", "ke", "aur", "nahi", "kya",
    "bhi", "se", "ko", "pe", "par", "ne", "ho", "raha", "rahi",
    "tha", "thi", "hun", "wala", "wali", "kuch", "sab", "yeh",
    "woh", "ap", "tum", "main", "rha", "rhi",
}

# ── Combined stopword set ──────────────────────────────────────────────────────
STOPWORDS: set[str] = ROMAN_URDU_MARKERS | {
    # English function words
    "i", "me", "my", "the", "a", "an", "is", "are", "was", "were",
    "have", "has", "had", "be", "been", "do", "does", "did", "and",
    "or", "but", "so", "if", "then", "of", "to", "in", "on", "at",
    "for", "from", "with", "by", "this", "that", "these", "those",
    "very", "really", "quite", "also", "too", "just", "now",
    # Temporal (not clinically useful alone)
    "din", "roz", "ab", "aaj", "kal", "subh", "shaam", "raat",
    "since", "ago", "days", "day", "hours", "hour", "weeks", "week",
    "morning", "night", "evening",
    # Intensifiers
    "bohat", "bahut", "zyada", "kam", "thora", "thori",
    "kar", "karne", "rahe", "thay",
    # Greetings / filler
    "hi", "hello", "hey", "salam", "assalam", "assalamualaikum",
    "walaikum", "namaste", "bhai", "sir", "madam", "doctor",
    "please", "thanks", "thankyou", "ok", "okay", "yes", "no",
    # Generic body / feeling words (too broad to map alone)
    "body", "part", "feel", "feeling", "felt",
}

_NUM_RE = re.compile(r"^[\d]+$")


# ─────────────────────────────────────────────────────────────────────────────
class DialectMiddleware:
    """
    Singleton — one instance loaded at import time.
    Call .get_context(text, session_id) from triage_node.
    """

    def __init__(self) -> None:
        self._terms: list[tuple[str, str]] = []          # (dialectal, clinical)
        self._model: Any = None                          # SentenceTransformer
        self._faiss_index: Any = None                   # faiss.IndexFlatIP
        self._bm25: Any = None                          # BM25Okapi
        self._known_tokens: set[str] = set()            # all tokenized dialectal terms
        self._ready = False

        UNKNOWN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not _DEPS_AVAILABLE:
            print(
                "⚠️  [DialectMW] faiss / sentence-transformers / rank-bm25 not installed.\n"
                "   Install with: pip install faiss-cpu sentence-transformers rank-bm25\n"
                "   Middleware will run in PASSTHROUGH mode (no RAG hints)."
            )
            return

        self._load()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not DICT_PATH.exists():
            print(f"⚠️  [DialectMW] dialect_dictionary.csv not found at {DICT_PATH}.")
            print("   Place the CSV at data/dialect_dictionary.csv and restart.")
            return

        # 1. Load CSV
        with open(DICT_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = (row.get("dialectal_term") or "").strip()
                c = (row.get("clinical_term")  or "").strip()
                if d and c:
                    self._terms.append((d, c))
                    for tok in _tokenize(d):
                        if len(tok) >= MIN_TOKEN_LEN:
                            self._known_tokens.add(tok.lower())

        if not self._terms:
            print("⚠️  [DialectMW] CSV loaded but no valid rows found.")
            return

        print(f"🔧 [DialectMW] Loading multilingual embedding model ({EMBEDDING_MODEL})…")
        self._model = SentenceTransformer(EMBEDDING_MODEL)

        # 2. Build FAISS index
        dialectal_texts = [t[0] for t in self._terms]
        embeddings = self._model.encode(
            dialectal_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        self._faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self._faiss_index.add(embeddings)

        # 3. Build BM25 index
        tokenized = [_tokenize(t) for t in dialectal_texts]
        self._bm25 = BM25Okapi(tokenized)

        self._ready = True
        print(
            f"✅ [DialectMW] Ready — {len(self._terms)} terms indexed "
            f"({EMBEDDING_MODEL}, FAISS + BM25 hybrid)."
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_context(
        self,
        text: str,
        session_id: str = "unknown",
        top_k: int = TOP_K,
    ) -> dict[str, Any]:
        """
        Main entry point called from triage_node.

        Returns a dict with:
            language       : "english" | "roman_urdu" | "urdu_script"
            cleaned_text   : content-words extracted from text
            hints          : list of {dialectal, clinical, score} dicts
            context_text   : formatted string ready to inject into system prompt
            logged_tokens  : unknown tokens written to unknown_terms.jsonl
        """
        _empty = {
            "language": "unknown", "cleaned_text": "", "hints": [],
            "context_text": "", "logged_tokens": [],
        }

        if not text or not text.strip():
            return _empty

        language = _detect_language(text)

        # English → no RAG needed; pass straight through
        if language == "english":
            return {**_empty, "language": "english", "cleaned_text": text}

        # Non-English but middleware not ready → passthrough with warning
        if not self._ready:
            print("⚠️  [DialectMW] Not ready — returning empty context.")
            return {**_empty, "language": language}

        # Extract only content words for indexing
        cleaned = _extract_content_words(text)
        if not cleaned.strip():
            return {**_empty, "language": language}

        # Hybrid RAG search
        hints = self._hybrid_search(cleaned, top_k=top_k)

        # Log unknown tokens (tokens not seen in any dialectal term)
        cleaned_tokens  = _tokenize(cleaned)
        unknown_tokens  = [
            t for t in cleaned_tokens
            if t.lower() not in self._known_tokens
        ]
        if unknown_tokens:
            self._log_unknown(unknown_tokens, text, session_id, language)

        context_text = _build_context_string(hints) if hints else ""

        return {
            "language":      language,
            "cleaned_text":  cleaned,
            "hints":         hints,
            "context_text":  context_text,
            "logged_tokens": unknown_tokens,
        }

    # ── Hybrid FAISS + BM25 search ───────────────────────────────────────────

    def _hybrid_search(
        self, text: str, top_k: int
    ) -> list[dict[str, Any]]:
        """
        Reciprocal Rank Fusion of:
          - FAISS cosine similarity (semantic, multilingual)
          - BM25 Okapi (lexical, token overlap)

        Only results above SEMANTIC_THRESH (for FAISS) or with positive
        BM25 score are included, so irrelevant terms are filtered out.
        """
        seen: dict[int, dict[str, Any]] = {}
        rrf: dict[int, float] = {}

        # ── FAISS ──
        q_emb = self._model.encode(
            [text], normalize_embeddings=True
        ).astype(np.float32)
        faiss_scores, faiss_idx = self._faiss_index.search(q_emb, top_k * 4)
        for rank, (score, idx) in enumerate(zip(faiss_scores[0], faiss_idx[0]), 1):
            if float(score) >= SEMANTIC_THRESH:
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)
                seen[idx] = {
                    "dialectal": self._terms[idx][0],
                    "clinical":  self._terms[idx][1],
                    "score":     round(float(score), 3),
                    "method":    "semantic",
                }

        # ── BM25 ──
        bm25_scores = self._bm25.get_scores(_tokenize(text))
        bm25_top    = np.argsort(bm25_scores)[::-1][: top_k * 4]
        for rank, idx in enumerate(bm25_top, 1):
            if bm25_scores[idx] > BM25_MIN_SCORE:
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)
                seen.setdefault(idx, {
                    "dialectal": self._terms[idx][0],
                    "clinical":  self._terms[idx][1],
                    "score":     round(float(bm25_scores[idx]), 3),
                    "method":    "lexical",
                })

        # ── RRF fusion & rank ──
        ranked = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)
        return [seen[i] for i in ranked[:top_k]]

    # ── Unknown-token logger ──────────────────────────────────────────────────

    def _log_unknown(
        self,
        unknown_tokens: list[str],
        raw_text:       str,
        session_id:     str,
        language:       str,
    ) -> None:
        """
        Writes each unrecognised token as a separate JSONL line so the
        admin dashboard can surface them for human-in-the-loop review.
        New mappings validated by a clinician are appended back to the CSV.
        """
        ts = datetime.now(timezone.utc).isoformat()
        with open(UNKNOWN_LOG_PATH, "a", encoding="utf-8") as f:
            for tok in unknown_tokens:
                entry = {
                    "timestamp":  ts,
                    "session_id": session_id,
                    "token":      tok,
                    "raw_text":   raw_text[:200],
                    "language":   language,
                    "status":     "pending_review",
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(
            f"📝 [DialectMW] Logged {len(unknown_tokens)} unknown token(s) "
            f"→ {UNKNOWN_LOG_PATH.name}: {unknown_tokens}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions (no class state)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """
    Heuristic language detection based on Unicode ranges and Roman Urdu markers.
    Returns: "urdu_script" | "roman_urdu" | "english"
    """
    # Any Arabic-block character → Urdu script
    if re.search(r"[\u0600-\u06FF]", text):
        return "urdu_script"
    # Roman Urdu marker words in lowercased token set
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & ROMAN_URDU_MARKERS:
        return "roman_urdu"
    return "english"


def _extract_content_words(text: str) -> str:
    """
    Pull out medically relevant tokens from a patient utterance, discarding:
      - Urdu/Roman Urdu grammatical particles
      - English function words
      - Numbers (standalone digits)
      - Tokens shorter than MIN_TOKEN_LEN
    """
    out: list[str] = []
    tokens = re.findall(r"[a-zA-Z]+|[\u0600-\u06FF]+|\d+", text)
    for tok in tokens:
        # Always keep Urdu-script tokens — they map to specific clinical terms
        if re.match(r"[\u0600-\u06FF]+", tok):
            out.append(tok)
            continue
        if _NUM_RE.match(tok):
            continue
        tok_low = tok.lower()
        if tok_low in STOPWORDS or len(tok_low) < MIN_TOKEN_LEN:
            continue
        out.append(tok_low)
    return " ".join(out)


def _build_context_string(hints: list[dict[str, Any]]) -> str:
    """
    Formats the RAG hints into a block that MedGemma's system prompt receives.
    Kept deliberately brief — just enough for the clinical model to normalise.
    """
    if not hints:
        return ""
    lines = [
        f"  - '{h['dialectal']}' → '{h['clinical']}'"
        for h in hints
    ]
    return (
        "DIALECT CONTEXT — patient may be using Pakistani/Urdu colloquialisms.\n"
        "Use these mappings when interpreting their symptom descriptions:\n"
        + "\n".join(lines)
    )


def _tokenize(text: str) -> list[str]:
    """Shared tokenizer for both BM25 indexing and query tokenization."""
    return [
        t for t in
        re.sub(r"[^\w\s\u0600-\u06FF]", " ", text.lower()).split()
        if t
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — imported by triage_agent
# ─────────────────────────────────────────────────────────────────────────────
dialect_middleware = DialectMiddleware()