"""FAISS index roundtrip (no embedding models)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from ai_engine.rag.index import build_index, load_index, save_index


def test_build_search_roundtrip():
    rng = np.random.default_rng(0)
    dim = 32
    n = 50
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    index = build_index(vecs)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "idx.faiss"
        save_index(index, p)
        idx2 = load_index(p)
        norms = np.linalg.norm(vecs[0:1], axis=1, keepdims=True)
        q = (vecs[0:1] / np.maximum(norms, 1e-12)).astype(np.float32)
        scores, idxs = idx2.search(q, 3)
        assert int(idxs[0][0]) == 0
