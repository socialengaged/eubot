"""
RAG retrieval: FAISS + embeddings when index is present; otherwise delegates to rag_stub.
"""
from __future__ import annotations

from typing import Any

from ai_engine.config.settings import get_settings
from ai_engine.providers.rag_stub import call_rag as call_rag_stub


def call_rag(query: str, **kw: Any) -> dict[str, Any]:
    """
    Return chunks + concatenated context for orchestrator.

    Shape matches rag_stub: query, chunks, context.
    If the index was built with ingest_sacred, context includes source/topic labels
    and chunks are text-only strings; chunks_detail holds full dicts when present.
    """
    settings = get_settings()
    if not settings.rag_use_real:
        return call_rag_stub(query, **kw)

    top_k = int(kw.get("top_k", 3))
    path = (settings.rag_index_path or "").strip()
    if path:
        from ai_engine.rag.sacred_retriever import SacredRAGRetriever, format_sacred_context

        sr = SacredRAGRetriever(path)
        if sr.load(path):
            rich = sr.retrieve(query, top_k=top_k)
            if rich:
                ctx = format_sacred_context(rich)
                return {
                    "query": query,
                    "chunks": [c.get("text", "") for c in rich],
                    "chunks_detail": rich,
                    "context": ctx,
                }

    from ai_engine.rag.retriever import ensure_loaded, retrieve

    if not ensure_loaded():
        return call_rag_stub(query, **kw)

    chunks = retrieve(query, top_k=top_k)
    context = "\n\n".join(chunks) if chunks else ""
    return {
        "query": query,
        "chunks": chunks,
        "context": context,
    }
