#!/usr/bin/env python3
"""Clean raw text, chunk into line-based JSONL with field 'text' for LM training."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def clean_block(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def chunk_text(s: str, max_chars: int) -> list[str]:
    """Split long documents into chunks of roughly max_chars (word boundaries)."""
    s = clean_block(s)
    if not s:
        return []
    if len(s) <= max_chars:
        return [s]
    chunks = []
    start = 0
    while start < len(s):
        end = min(start + max_chars, len(s))
        if end < len(s):
            sp = s.rfind(" ", start, end)
            if sp > start + max_chars // 4:
                end = sp
        piece = s[start:end].strip()
        if len(piece) > 50:
            chunks.append(piece)
        start = end if end == len(s) else end + 1
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_chars", type=int, default=4000, help="Max chars per JSONL text field")
    ap.add_argument("--train_in", type=Path, default=ROOT / "data" / "raw" / "train.txt")
    ap.add_argument("--val_in", type=Path, default=ROOT / "data" / "raw" / "val.txt")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def process(inp: Path, outp: Path) -> int:
        if not inp.is_file():
            raise SystemExit(f"Missing {inp} — run download_data.py first.")
        raw = inp.read_text(encoding="utf-8", errors="ignore")
        parts = raw.split("\n\n")
        n = 0
        with open(outp, "w", encoding="utf-8") as f:
            for block in parts:
                for chunk in chunk_text(block, args.max_chars):
                    f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                    n += 1
        return n

    nt = process(args.train_in, args.out_dir / "train.jsonl")
    nv = process(args.val_in, args.out_dir / "val.jsonl")
    print(f"Wrote train.jsonl ({nt} lines), val.jsonl ({nv} lines) -> {args.out_dir}")


if __name__ == "__main__":
    main()
