#!/usr/bin/env python3
"""
Fase 6 — Download physics-related HuggingFace datasets to JSONL (field \"text\").

Env:
  EUROBOT_PHYSICS_OUT  Output directory (default: ./output/physics)

Requires: pip install datasets

Note: flappingairplanes/physics-textbooks-gpt2 may be unavailable on Hub from some regions;
we fall back to Feynman lectures + optional IUTVanguard/PhysicsEval.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

OUTPUT = Path(os.environ.get("EUROBOT_PHYSICS_OUT", str(Path(__file__).resolve().parent / "output" / "physics")))
OUTPUT.mkdir(parents=True, exist_ok=True)


def row_to_text(row: dict) -> str:
    for key in ("text", "content", "body", "article"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()
    # Feynman lectures
    if "section_text" in row and isinstance(row["section_text"], str):
        parts = [
            str(row.get("book_title", "")),
            str(row.get("chapter_title", "")),
            str(row.get("section_title", "")),
            row["section_text"],
        ]
        return "\n\n".join(p for p in parts if p and str(p).strip())
    return json.dumps(row, ensure_ascii=False)


def dump_dataset(name: str, split: str = "train") -> None:
    from datasets import load_dataset

    print(f"Loading {name} ...")
    ds = load_dataset(name, split=split)
    out_path = OUTPUT / (name.replace("/", "__") + ".jsonl")
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row_to_text(row if isinstance(row, dict) else row)
            if len(text) < 50:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
    print(f"  Wrote {n} lines -> {out_path}")


def main() -> None:
    # Primary: Feynman (always try)
    try:
        dump_dataset("enesxgrahovac/the-feynman-lectures-on-physics", "train")
    except Exception as e:
        print(f"SKIP Feynman: {e}")

    # Optional large physics textbooks (may not exist on Hub)
    for name in ("flappingairplanes/physics-textbooks-gpt2",):
        try:
            dump_dataset(name, "train")
        except Exception as e:
            print(f"SKIP {name}: {e}")

    # Optional: physics problems with solutions (smaller)
    try:
        dump_dataset("IUTVanguard/PhysicsEval", "train")
    except Exception as e:
        print(f"SKIP PhysicsEval: {e}")


if __name__ == "__main__":
    main()
