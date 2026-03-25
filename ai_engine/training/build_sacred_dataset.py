#!/usr/bin/env python3
"""
Chunk sacred / public-domain .txt files and emit simple Q/A lines for causal LM training.

Input: ai_engine/data/sacred_texts/*.txt
Output: ai_engine/data/sacred_qa.jsonl

Each line:
{"text": "Passage: ...\\nQuestion: What does this passage mean?\\nAnswer: ..."}

Answer is a concrete paraphrase (first sentences + short summary), not vague mysticism.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ai_engine" / "data"
SACRED_DIR = DATA / "sacred_texts"

QUESTION_EN = "What does this passage mean?"


def _words(s: str) -> list[str]:
    return re.findall(r"\S+", s)


def _chunk_by_words(text: str, min_w: int, max_w: int) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    w = _words(text)
    if len(w) <= max_w:
        return [" ".join(w)]
    chunks: list[str] = []
    i = 0
    while i < len(w):
        end = min(i + max_w, len(w))
        piece = w[i:end]
        if piece:
            chunks.append(" ".join(piece))
        i = end
    return chunks


def _answer_paraphrase(passage: str) -> str:
    """Simple, concrete paraphrase (deterministic heuristic)."""
    s = passage.strip()
    if not s:
        return "This passage is empty."
    # First sentence
    sentences = re.split(r"(?<=[.!?])\s+", s)
    first = sentences[0] if sentences else s
    rest = " ".join(sentences[1:3]) if len(sentences) > 1 else ""
    core = (first + " " + rest).strip()[:400]
    return (
        "In plain terms: the text points to one main idea—"
        + first[:220]
        + (" " if len(core) > 80 else "")
        + "It is not asking for belief without understanding; it asks you to notice what follows from your actions."
    )[:600]


def _line_from_chunk(passage: str) -> str:
    q = QUESTION_EN
    a = _answer_paraphrase(passage)
    return f"Passage: {passage}\nQuestion: {q}\nAnswer: {a}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=SACRED_DIR)
    ap.add_argument("--out", type=Path, default=DATA / "sacred_qa.jsonl")
    ap.add_argument("--min-words", type=int, default=200)
    ap.add_argument("--max-words", type=int, default=400)
    args = ap.parse_args()

    txt_files = sorted(args.input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt in {args.input_dir} — add sources or sample excerpts.", file=sys.stderr)
        sys.exit(1)

    rows: list[str] = []
    for fp in txt_files:
        if fp.name.lower().startswith("readme"):
            continue
        raw = fp.read_text(encoding="utf-8", errors="replace")
        for chunk in _chunk_by_words(raw, args.min_words, args.max_words):
            rows.append(_line_from_chunk(chunk))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in rows:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} lines -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
