"""
Build FAISS index for sacred texts: manifest JSONL, word chunks, OpenAI text-embedding-3-small.

Manifest: one JSON object per line:
  {"file": "path/to.txt", "source": "Title or citation", "topic": "stoicism"}

Paths in "file" are relative to the manifest's parent directory unless absolute.

Example:
  python -m ai_engine.rag.ingest_sacred \\
    --manifest ai_engine/data/sacred_sources.jsonl \\
    --out-dir eurobot_baby/vector_db/sacred \\
    --max-words 300 --overlap 30 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from ai_engine.rag.embeddings import embed_batch
from ai_engine.rag.index import build_index, save_index, save_metadata


def chunk_by_words(text: str, max_words: int, overlap: int) -> list[str]:
    """Non-overlapping sliding windows of `max_words` with `overlap` word carry."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    words = re.findall(r"\S+", text)
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]
    step = max(1, max_words - overlap)
    out: list[str] = []
    i = 0
    while i < len(words):
        piece = words[i : i + max_words]
        out.append(" ".join(piece))
        if i + max_words >= len(words):
            break
        i += step
    return out


def load_manifest(path: Path) -> list[dict[str, str]]:
    base = path.parent
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
            file_p = str(obj.get("file") or "").strip()
            if not file_p:
                continue
            src = str(obj.get("source") or "unknown").strip() or "unknown"
            topic = str(obj.get("topic") or "general").strip() or "general"
            fp = Path(file_p)
            if not fp.is_absolute():
                fp = (base / fp).resolve()
            rows.append({"path": str(fp), "source": src, "topic": topic})
    return rows


def run_ingest_sacred(
    manifest: Path,
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

    manifest_rows = load_manifest(manifest)
    if not manifest_rows:
        print("No rows in manifest or no readable files.", file=sys.stderr)
        sys.exit(1)

    chunks_meta: list[dict[str, str]] = []
    for row in manifest_rows:
        p = Path(row["path"])
        if not p.is_file():
            print(f"WARN: skip missing file {p}", file=sys.stderr)
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        for piece in chunk_by_words(raw, max_words, overlap):
            if not piece.strip():
                continue
            chunks_meta.append(
                {
                    "source": row["source"],
                    "topic": row["topic"],
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
        "source_manifest": str(manifest.resolve()),
        "num_vectors": len(chunks_meta),
        "chunk_max_words": max_words,
        "chunk_overlap_words": overlap,
    }
    save_metadata(meta, meta_path)
    print(f"Wrote {index_path} and {meta_path} ({len(chunks_meta)} vectors, dim={dim})")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest sacred manifest into FAISS (OpenAI embeddings).")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("ai_engine/data/sacred_sources.jsonl"),
        help="JSONL: file, source, topic per line",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eurobot_baby/vector_db/sacred"),
        help="Output dir (index.faiss + metadata.pkl)",
    )
    p.add_argument("--max-words", type=int, default=300)
    p.add_argument("--overlap", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--openai-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model id",
    )
    p.add_argument("--max-chunks", type=int, default=0, help="Limit chunks (0 = no limit)")
    args = p.parse_args()
    max_c = args.max_chunks if args.max_chunks > 0 else None
    run_ingest_sacred(
        args.manifest,
        args.out_dir,
        max_words=args.max_words,
        overlap=args.overlap,
        batch_size=args.batch_size,
        openai_model=args.openai_model,
        max_chunks=max_c,
    )


if __name__ == "__main__":
    main()
