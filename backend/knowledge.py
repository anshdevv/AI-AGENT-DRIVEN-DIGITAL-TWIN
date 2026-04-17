from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .domain import DomainConfig


WORD_RE = re.compile(r"[a-zA-Z0-9]+")


@dataclass(slots=True)
class KnowledgeHit:
    source: str
    score: int
    content: str


class KnowledgeBase:
    def __init__(self, domain: DomainConfig) -> None:
        self._documents = self._load_documents(domain.faq_dir)

    @staticmethod
    def _load_documents(directory: Path) -> list[tuple[str, str]]:
        documents: list[tuple[str, str]] = []
        if not directory.exists():
            return documents

        for path in sorted(directory.glob("*.md")):
            documents.append((path.stem.replace("_", " "), path.read_text(encoding="utf-8")))
        return documents

    def search(self, query: str, limit: int = 3) -> list[KnowledgeHit]:
        tokens = {token.lower() for token in WORD_RE.findall(query)}
        if not tokens:
            return []

        hits: list[KnowledgeHit] = []
        for source, content in self._documents:
            lowered = content.lower()
            score = sum(1 for token in tokens if token in lowered)
            if score:
                hits.append(KnowledgeHit(source=source, score=score, content=content.strip()))

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]
