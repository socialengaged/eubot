"""
Build FAISS index from JSONL where each line is:
  {"text": "<long text>", "source": "...", "topic": "medicine|conversational|..."}

Uses OpenAI text-embedding-3-small (same as ingest_sacred). Chunks by words.

  python -m ai_engine.rag.ingest_multidomain_jsonl \\
    --jsonl ai_engine/data/rag_expansion/combined_rag.jsonl \\
    --out-dir eurobot_baby/vector_db/rag_expansion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ai_engine.rag.embeddings import embed_batch
from ai_engine.rag.index import build_index, save_index, save_metadata
from ai_engine.rag.ingest_sacred import chunk_by_words


def load_jsonl_docs(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(obj.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "source": str(obj.get("source") or "unknown").strip() or "unknown",
                    "topic": str(obj.get("topic") or "general").strip() or "general",
                }
            )
    return rows


def run_ingest_multidomain_jsonl(
    jsonl_path: Path,
    out_dir: Path,
    *,
    max_words: int,
    overlap: int,
    batch_size: int,
    openai_model: str,
    max_chunks: int | None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_jsonl_docs(jsonl_path)
    if not docs:
        print("No documents in JSONL.", file=sys.stderr)
        sys.exit(1)

    chunks_meta: list[dict[str, str]] = []
    for doc in docs:
        for piece in chunk_by_words(doc["text"], max_words, overlap):
            if not piece.strip():
                continue
            chunks_meta.append(
                {
                    "source": doc["source"],
                    "topic": doc["topic"],
                    "text": piece,
                }
            )

    if max_chunks is not None and max_chunks > 0:
        chunks_meta = chunks_meta[:max_chunks]

    if not chunks_meta:
        print("No chunks produced.", file=sys.stderr)
        sys.exit(1)

    texts = [c["text"] for c in chunks_meta]
    all_vecs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embed_batch(batch, backend="openai", openai_model=openai_model)
        all_vecs.append(vecs)
    mat = np.vstack(all_vecs)
    dim = mat.shape[1]

    index = build_index(mat)
    index_path = out_dir / "index.faiss"
    meta_path = out_dir / "metadata.pkl"
    save_index(index, index_path)

    meta = {
        "texts": texts,
        "chunks": chunks_meta,
        "embedding_dim": dim,
        "embed_backend": "openai",
        "rag_embed_model": "",
        "openai_embed_model": openai_model,
        "source_jsonl": str(jsonl_path.resolve()),
        "num_vectors": len(chunks_meta),
        "chunk_max_words": max_words,
        "chunk_overlap_words": overlap,
    }
    save_metadata(meta, meta_path)
    print(f"Wrote {index_path} and {meta_path} ({len(chunks_meta)} vectors, dim={dim})")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest multidomain JSONL into FAISS (OpenAI embeddings).")
    p.add_argument("--jsonl", type=Path, required=True, help="JSONL with text, source, topic")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eurobot_baby/vector_db/rag_expansion"),
        help="Output directory for index.faiss + metadata.pkl",
    )
    p.add_argument("--max-words", type=int, default=300)
    p.add_argument("--overlap", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--openai-model", default="text-embedding-3-small")
    p.add_argument("--max-chunks", type=int, default=0, help="0 = no limit")
    args = p.parse_args()
    max_c = args.max_chunks if args.max_chunks > 0 else None
    run_ingest_multidomain_jsonl(
        args.jsonl,
        args.out_dir,
        max_words=args.max_words,
        overlap=args.overlap,
        batch_size=args.batch_size,
        openai_model=args.openai_model,
        max_chunks=max_c,
    )


if __name__ == "__main__":
    main()
