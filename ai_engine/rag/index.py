"""
FAISS index helpers: inner product on L2-normalized vectors (= cosine similarity).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _faiss():
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError(
            "faiss-cpu is required for RAG index. Install: pip install faiss-cpu"
        ) from e
    return faiss


def _l2_normalize_rows(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vecs / norms).astype(np.float32, copy=False)


def build_index(vectors: np.ndarray) -> Any:
    """
    Build IndexFlatIP for cosine-style search (vectors must be L2-normalized rows).
    vectors: (n, dim) float32
    """
    faiss = _faiss()
    if vectors.dtype != np.float32:
        vectors = np.asarray(vectors, dtype=np.float32)
    vectors = _l2_normalize_rows(np.asarray(vectors, dtype=np.float32))
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(int(dim))
    index.add(vectors)
    return index


def save_index(index: Any, path: Path | str) -> None:
    """Write FAISS index to a single file."""
    faiss = _faiss()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p))


def load_index(path: Path | str) -> Any:
    """Load FAISS index from file."""
    faiss = _faiss()
    return faiss.read_index(str(path))


def save_metadata(metadata: dict[str, Any], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_metadata(path: Path | str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)
