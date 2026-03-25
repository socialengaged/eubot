"""
Build FAISS index + metadata from train.jsonl (or any JSONL with {"text": "..."} per line).

CPU only. Example:

  python -m ai_engine.rag.ingest \\
    --jsonl eurobot_baby/data/processed/train.jsonl \\
    --out-dir eurobot_baby/data/processed/rag_index \\
    --chunk-size 400 --chunk-overlap 80 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ai_engine.rag.embeddings import embed_batch
from ai_engine.rag.index import build_index, save_index, save_metadata


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Character-level chunks with overlap."""
    if chunk_size <= 0:
        return [text] if text.strip() else []
    t = text.strip()
    if not t:
        return []
    if len(t) <= chunk_size:
        return [t]
    out: list[str] = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(t):
        piece = t[i : i + chunk_size]
        if piece:
            out.append(piece)
        if i + chunk_size >= len(t):
            break
        i += step
    return out


def load_jsonl_texts(path: Path) -> list[str]:
    lines: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                lines.append(t)
    return lines


def run_ingest(
    jsonl_path: Path,
    out_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    backend: str,
    local_model: str,
    openai_model: str,
    max_chunks: int | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_texts = load_jsonl_texts(jsonl_path)
    chunks: list[str] = []
    for doc in raw_texts:
        parts = chunk_text(doc, chunk_size, chunk_overlap)
        chunks.extend(parts)

    if max_chunks is not None and max_chunks > 0:
        chunks = chunks[:max_chunks]

    if not chunks:
        print("No chunks produced; check JSONL format.", file=sys.stderr)
        sys.exit(1)

    be = backend.strip().lower()
    all_vecs: list[np.ndarray] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if be == "openai":
            vecs = embed_batch(
                batch,
                backend="openai",
                openai_model=openai_model,
            )
        else:
            vecs = embed_batch(batch, backend="local", model_id=local_model)
        all_vecs.append(vecs)
    mat = np.vstack(all_vecs)
    dim = mat.shape[1]

    index = build_index(mat)
    index_path = out_dir / "index.faiss"
    meta_path = out_dir / "metadata.pkl"
    save_index(index, index_path)

    meta = {
        "texts": chunks,
        "embedding_dim": dim,
        "embed_backend": be,
        "rag_embed_model": local_model,
        "openai_embed_model": openai_model,
        "source_jsonl": str(jsonl_path.resolve()),
        "num_vectors": len(chunks),
    }
    save_metadata(meta, meta_path)
    print(f"Wrote {index_path} and {meta_path} ({len(chunks)} vectors, dim={dim})")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest JSONL into FAISS RAG index (CPU).")
    p.add_argument(
        "--jsonl",
        type=Path,
        default=Path("eurobot_baby/data/processed/train.jsonl"),
        help='JSONL with {"text": "..."} per line',
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eurobot_baby/data/processed/rag_index"),
        help="Output directory for index.faiss + metadata.pkl",
    )
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--chunk-overlap", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--backend", choices=("local", "openai"), default="local")
    p.add_argument(
        "--local-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model id when backend=local",
    )
    p.add_argument(
        "--openai-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model when backend=openai",
    )
    p.add_argument("--max-chunks", type=int, default=0, help="Limit chunks (0 = no limit), for smoke tests")
    args = p.parse_args()
    max_c = args.max_chunks if args.max_chunks > 0 else None
    run_ingest(
        args.jsonl,
        args.out_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        backend=args.backend,
        local_model=args.local_model,
        openai_model=args.openai_model,
        max_chunks=max_c,
    )


if __name__ == "__main__":
    main()
