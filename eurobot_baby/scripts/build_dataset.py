#!/usr/bin/env python3
"""
Merge raw text files, normalize, chunk into JSONL with {\"text\": \"...\"} per line.
Expects outputs from download_data.py, download_classics.py, download_sacred.py under data/raw/.

See docs/PIANO_DATASET_EXPANSION_v1.md and eurobot_baby/docs/RUNBOOK_PHASE0.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# Order matters slightly (wiki baseline first).
TRAIN_PARTS_REQUIRED = [
    "01_wikitext_train.txt",
    "02_wikipedia_it.txt",
    "03_gutenberg_en.txt",
    "04_gutenberg_it.txt",
    "05_sacred.txt",
    "07_esoteric_sage_corpus.txt",
    "06_gutenberg_it_extra.txt",
    # --- Fase 0: erano sul disco ma esclusi dalla lista originale ---
    "15_hf_vivechan.txt",
    "16_hf_sep_philosophy.txt",
    "17_wikipedia_esoteric_filtered.txt",
]

# Se il file manca, viene ignorato senza log (placeholder per fasi successive).
TRAIN_PARTS_OPTIONAL = [
    "08_esoteric_expanded.txt",
    "20_physics_corpus.txt",
    "21_physics_textbooks_hf.txt",
    "22_arxiv_physics.txt",
    "30_math_openwebmath_subset.txt",
    "31_math_classics.txt",
    "32_arxiv_math.txt",
    "40_astronomy_corpus.txt",
    "41_astronomy_astro_texts_hf.txt",
    "42_arxiv_astroph.txt",
    "50_pes2o_stem_subset.txt",
    # Teologia / informatica (generare con tools/scraping + merge_outputs_to_raw_names.py)
    "60_theology_corpus.txt",
    "70_informatics_corpus.txt",
    # Astrologia (sacred-texts/astrology/) e massoneria (sacred-texts/freemasonry/) — merge separato da 08
    "43_astrology_corpus.txt",
    "44_masonry_corpus.txt",
    # Informatica / storia del calcolo (Gutenberg + arxiv cs già in 70)
    "71_gutenberg_computing.txt",
]

VAL_FILE = "01_wikitext_val.txt"


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_chars, n)
        piece = text[i:end]
        if end < n:
            br = piece.rfind("\n\n")
            if br > chunk_chars // 4:
                piece = piece[: br + 2].strip()
                end = i + len(piece)
        if len(piece) >= 80:
            chunks.append(piece)
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks


def load_optional(path: Path, *, quiet_missing: bool = False) -> str:
    if not path.is_file():
        if not quiet_missing:
            print(f"  (missing, skip) {path.name}")
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_chars", type=int, default=4000, help="Target chunk size in characters")
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--train_out", type=Path, default=PROC / "train.jsonl")
    ap.add_argument("--val_out", type=Path, default=PROC / "val.jsonl")
    args = ap.parse_args()

    PROC.mkdir(parents=True, exist_ok=True)

    train_blob_parts: list[str] = []
    for name in TRAIN_PARTS_REQUIRED:
        p = RAW / name
        t = load_optional(p, quiet_missing=False)
        if t:
            train_blob_parts.append(t)
    for name in TRAIN_PARTS_OPTIONAL:
        p = RAW / name
        t = load_optional(p, quiet_missing=True)
        if t:
            train_blob_parts.append(t)
    train_blob = "\n\n".join(train_blob_parts)

    val_blob = load_optional(RAW / VAL_FILE)
    if not train_blob.strip():
        raise SystemExit(
            "No training text found. Run download_data.py, download_classics.py, download_sacred.py first."
        )

    train_chunks = chunk_text(train_blob, args.chunk_chars, args.overlap)
    val_chunks = chunk_text(val_blob, args.chunk_chars, args.overlap) if val_blob.strip() else train_chunks[:256]

    if not val_chunks:
        val_chunks = train_chunks[: min(512, len(train_chunks))]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_out, "w", encoding="utf-8") as f:
        for c in train_chunks:
            f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")

    with open(args.val_out, "w", encoding="utf-8") as f:
        for c in val_chunks:
            f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")

    print(f"train: {len(train_chunks)} chunks -> {args.train_out}")
    print(f"val:   {len(val_chunks)} chunks -> {args.val_out}")


if __name__ == "__main__":
    main()
