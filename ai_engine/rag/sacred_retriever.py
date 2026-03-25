"""
Sacred RAG: FAISS + metadata chunks {source, topic, text} from ingest_sacred.
Query embedding must match ingest (OpenAI text-embedding-3-small by default).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_engine.rag.embeddings import embed_text
from ai_engine.rag.index import load_index, load_metadata


def format_sacred_context(chunks: list[dict[str, str]]) -> str:
    """Human-readable block for system / user augmentation."""
    lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        src = (c.get("source") or "").strip()
        topic = (c.get("topic") or "").strip()
        text = (c.get("text") or "").strip()
        head = f"[{i}]"
        if src or topic:
            head += f" source={src!r} topic={topic!r}"
        lines.append(f"{head}\n{text}")
    return "\n\n".join(lines)


class SacredRAGRetriever:
    """Lazy-loadable retriever for a directory built by ingest_sacred."""

    def __init__(self, index_dir: Path | str | None = None):
        self.index_dir = Path(index_dir) if index_dir else None
        self._index: Any = None
        self._meta: dict[str, Any] | None = None

    def load(self, index_dir: Path | str | None = None) -> bool:
        base = Path(index_dir) if index_dir is not None else self.index_dir
        if base is None:
            return False
        self.index_dir = base
        idx_path = base / "index.faiss"
        meta_path = base / "metadata.pkl"
        if not idx_path.is_file() or not meta_path.is_file():
            return False
        idx = load_index(idx_path)
        meta = load_metadata(meta_path)
        ed = meta.get("embedding_dim")
        if ed is not None and int(ed) != idx.d:
            return False
        self._index = idx
        self._meta = meta
        return True

    def _chunks_list(self) -> list[dict[str, str]]:
        assert self._meta is not None
        chunks = self._meta.get("chunks")
        if isinstance(chunks, list) and chunks:
            out: list[dict[str, str]] = []
            for c in chunks:
                if isinstance(c, dict) and str(c.get("text") or "").strip():
                    out.append(
                        {
                            "source": str(c.get("source") or ""),
                            "topic": str(c.get("topic") or ""),
                            "text": str(c.get("text") or "").strip(),
                        }
                    )
            return out
        texts = self._meta.get("texts") or []
        return [{"source": "", "topic": "", "text": str(t)} for t in texts]

    def _texts_parallel(self) -> list[str]:
        return [c["text"] for c in self._chunks_list()]

    def _query_embedding(self, query: str) -> np.ndarray:
        assert self._meta is not None
        be = (self._meta.get("embed_backend") or "openai").strip().lower()
        oai_m = self._meta.get("openai_embed_model") or "text-embedding-3-small"
        local_m = self._meta.get("rag_embed_model") or "sentence-transformers/all-MiniLM-L6-v2"
        if be == "openai":
            v = embed_text(query, backend="openai", openai_model=oai_m)
        else:
            v = embed_text(query, backend="local", model_id=local_m)
        return np.asarray(v, dtype=np.float32).reshape(1, -1)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
        """Return up to top_k chunk dicts with source, topic, text."""
        if self._index is None or self._meta is None:
            return []
        chunks = self._chunks_list()
        if not chunks:
            return []
        k = min(max(1, top_k), len(chunks))
        try:
            q = self._query_embedding(query)
        except Exception:
            return []
        scores, idxs = self._index.search(q, k)
        out: list[dict[str, str]] = []
        for j in idxs[0]:
            j = int(j)
            if 0 <= j < len(chunks):
                out.append(dict(chunks[j]))
        return out


def retrieve_with_metadata(
    query: str,
    top_k: int = 3,
    *,
    index_dir: Path | str | None = None,
) -> tuple[list[dict[str, str]], str]:
    """
    Load once from index_dir (or use env in caller), return chunks + formatted context string.
    Returns ([], "") if index missing or error.
    """
    r = SacredRAGRetriever(index_dir)
    if not r.load(index_dir):
        return [], ""
    chunks = r.retrieve(query, top_k=top_k)
    return chunks, format_sacred_context(chunks)
