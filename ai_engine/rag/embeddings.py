"""
Embedding backends for RAG (CPU). Local: sentence-transformers; optional: OpenAI API.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import httpx
import numpy as np

from ai_engine.config.settings import get_settings


def _l2_normalize_rows(vecs: np.ndarray) -> np.ndarray:
    """Normalize each row to unit length (float32)."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vecs / norms).astype(np.float32, copy=False)


def embed_text_local(text: str, model_id: str) -> np.ndarray:
    """Single text -> (dim,) float32 L2-normalized."""
    from sentence_transformers import SentenceTransformer

    model = _get_st_model(model_id)
    v = model.encode(
        text or "",
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(v, dtype=np.float32).reshape(-1)


def embed_batch_local(texts: list[str], model_id: str) -> np.ndarray:
    """Batch encode -> (n, dim) float32, row L2-normalized."""
    from sentence_transformers import SentenceTransformer

    model = _get_st_model(model_id)
    vecs = model.encode(
        texts or [""],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return _l2_normalize_rows(arr)


@lru_cache(maxsize=4)
def _get_st_model(model_id: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device="cpu")


def embed_text_openai(text: str, model_name: str) -> np.ndarray:
    """OpenAI embeddings -> L2-normalized float32 vector."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for AI_ENGINE_RAG_EMBED_BACKEND=openai")
    url = settings.openai_base_url.rstrip("/") + "/embeddings"
    payload = {"model": model_name, "input": text or ""}
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = json.loads(r.text)
    emb = data["data"][0]["embedding"]
    v = np.asarray(emb, dtype=np.float32).reshape(1, -1)
    return _l2_normalize_rows(v)[0]


def embed_batch_openai(texts: list[str], model_name: str) -> np.ndarray:
    """Batch OpenAI embeddings (single request with list input)."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for AI_ENGINE_RAG_EMBED_BACKEND=openai")
    url = settings.openai_base_url.rstrip("/") + "/embeddings"
    payload = {"model": model_name, "input": texts or [""]}
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=300.0) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = json.loads(r.text)
    rows = sorted(data["data"], key=lambda x: x["index"])
    mat = np.stack([np.asarray(x["embedding"], dtype=np.float32) for x in rows], axis=0)
    return _l2_normalize_rows(mat)


def embed_text(
    text: str,
    *,
    backend: str | None = None,
    model_id: str | None = None,
    openai_model: str | None = None,
) -> np.ndarray:
    """
    Embed a single string to a L2-normalized float32 vector (cosine-ready for inner product index).
    Backend: local | openai. If backend/model_id omitted, uses get_settings().
    """
    settings = get_settings()
    be = (backend or settings.rag_embed_backend or "local").strip().lower()
    if be == "openai":
        om = openai_model or settings.openai_embed_model
        return embed_text_openai(text, om)
    mid = model_id or settings.rag_embed_model
    return embed_text_local(text, mid)


def embed_batch(
    texts: list[str],
    *,
    backend: str | None = None,
    model_id: str | None = None,
    openai_model: str | None = None,
) -> np.ndarray:
    """Embed many strings -> (n, dim) float32, row-normalized."""
    settings = get_settings()
    be = (backend or settings.rag_embed_backend or "local").strip().lower()
    if be == "openai":
        om = openai_model or settings.openai_embed_model
        return embed_batch_openai(texts, om)
    mid = model_id or settings.rag_embed_model
    return embed_batch_local(texts, mid)
