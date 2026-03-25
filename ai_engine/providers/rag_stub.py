"""RAG retrieval — Phase 1 stub; Phase 2: FAISS + embeddings."""
from __future__ import annotations

from typing import Any


def call_rag(query: str, **_: Any) -> dict[str, Any]:
    """
    Return empty context until Phase 2 implements FAISS ingestion.

    Shape is stable for orchestrator integration.
    """
    return {
        "query": query,
        "chunks": [],
        "context": "",
    }
