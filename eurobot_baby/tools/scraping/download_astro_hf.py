#!/usr/bin/env python3
"""Fase 8 — patrickfleith/astro_texts_dataset -> JSONL text."""
from __future__ import annotations

import json
import os
from pathlib import Path

OUTPUT = Path(os.environ.get("EUROBOT_ASTRO_OUT", str(Path(__file__).resolve().parent / "output" / "astro")))


def row_to_text(row: dict) -> str:
    for k in ("text", "content", "body"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return json.dumps(row, ensure_ascii=False)


def main() -> None:
    from datasets import load_dataset

    OUTPUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("patrickfleith/astro_texts_dataset", split="train")
    out_path = OUTPUT / "astro_texts_hf.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            t = row_to_text(row if isinstance(row, dict) else {})
            if len(t) < 30:
                continue
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} lines -> {out_path}")


if __name__ == "__main__":
    main()
