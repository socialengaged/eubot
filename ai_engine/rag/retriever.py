"""
Lazy-loaded FAISS retriever (CPU). Uses embedding settings from index metadata (must match ingest).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_engine.config.settings import get_settings
from ai_engine.rag.embeddings import embed_text
from ai_engine.rag.index import load_index, load_metadata

_index: Any = None
_metadata: dict[str, Any] | None = None
_loaded_dir: str | None = None


def _index_dir() -> Path | None:
    raw = (get_settings().rag_index_path or "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_dir() else None


def _paths_for_dir(d: Path) -> tuple[Path, Path]:
    return d / "index.faiss", d / "metadata.pkl"


def load_retriever_index(index_dir: Path | str | None = None) -> bool:
    """
    Load index + metadata from disk. Returns False if missing or invalid.
    If index_dir is None, uses AI_ENGINE_RAG_INDEX_PATH.
    """
    global _index, _metadata, _loaded_dir
    base = Path(index_dir) if index_dir is not None else _index_dir()
    if base is None:
        return False
    idx_path, meta_path = _paths_for_dir(base)
    if not idx_path.is_file() or not meta_path.is_file():
        return False
    idx = load_index(idx_path)
    meta = load_metadata(meta_path)
    ed = meta.get("embedding_dim")
    if ed is not None and int(ed) != idx.d:
        return False
    _index = idx
    _metadata = meta
    _loaded_dir = str(base.resolve())
    return True


def ensure_loaded() -> bool:
    """Load once per process if RAG index path is set and files exist."""
    global _loaded_dir
    d = _index_dir()
    if d is None:
        return False
    cur = str(d.resolve())
    if _index is not None and _metadata is not None and _loaded_dir == cur:
        return True
    return load_retriever_index(d)


def reset_cache() -> None:
    """Test helper: unload index."""
    global _index, _metadata, _loaded_dir
    _index = None
    _metadata = None
    _loaded_dir = None


def _query_embedding(query: str) -> np.ndarray:
    assert _metadata is not None
    be = (_metadata.get("embed_backend") or "local").strip().lower()
    local_m = _metadata.get("rag_embed_model") or "sentence-transformers/all-MiniLM-L6-v2"
    oai_m = _metadata.get("openai_embed_model") or "text-embedding-3-small"
    if be == "openai":
        v = embed_text(query, backend="openai", openai_model=oai_m)
    else:
        v = embed_text(query, backend="local", model_id=local_m)
    return np.asarray(v, dtype=np.float32).reshape(1, -1)


def retrieve(query: str, top_k: int = 3) -> list[str]:
    """
    Return top_k text chunks for query. Empty list if index unavailable or on error.
    """
    if not ensure_loaded() or _index is None or _metadata is None:
        return []
    texts: list[str] = _metadata.get("texts") or []
    if not texts:
        return []
    k = min(max(1, top_k), len(texts))
    try:
        q = _query_embedding(query)
    except Exception:
        return []
    scores, idxs = _index.search(q, k)
    out: list[str] = []
    for j in idxs[0]:
        j = int(j)
        if 0 <= j < len(texts):
            out.append(texts[j])
    return out


def retrieve_with_metadata(query: str, top_k: int = 3) -> tuple[list[dict[str, str]], str]:
    """
    When the index directory (AI_ENGINE_RAG_INDEX_PATH) was built with ingest_sacred,
    return (chunk dicts with source/topic/text, formatted context). Otherwise ([], "").
    Uses SacredRAGRetriever for a single code path with rich metadata.
    """
    from ai_engine.rag.sacred_retriever import SacredRAGRetriever, format_sacred_context

    d = _index_dir()
    if d is None:
        return [], ""
    sr = SacredRAGRetriever(d)
    if not sr.load(d):
        return [], ""
    chunks = sr.retrieve(query, top_k=top_k)
    if not chunks:
        return [], ""
    return chunks, format_sacred_context(chunks)
