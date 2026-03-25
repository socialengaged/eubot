#!/usr/bin/env python3
"""
Fase 7 — OpenWebMath subset (streaming) -> JSONL con campo \"text\".

Env:
  EUROBOT_MATH_OUT      Directory output (default: ./output/math)
  EUROBOT_OPENWEBMATH_MAX_DOCS  Cap documenti (default: 500000)
  EUROBOT_HF_TOKEN      Opzionale per gated datasets

Requires: pip install datasets
"""
from __future__ import annotations

import json
import os
from pathlib import Path

OUTPUT = Path(os.environ.get("EUROBOT_MATH_OUT", str(Path(__file__).resolve().parent / "output" / "math")))
OUTPUT.mkdir(parents=True, exist_ok=True)
MAX_DOCS = int(os.environ.get("EUROBOT_OPENWEBMATH_MAX_DOCS", "500000"))


def main() -> None:
    from datasets import load_dataset

    name = "open-web-math/open-web-math"
    print(f"Streaming {name}, max_docs={MAX_DOCS} ...")
    ds = load_dataset(name, split="train", streaming=True)
    out_path = OUTPUT / "openwebmath_subset.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) < 50:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
            if n % 10000 == 0:
                print(f"  ... {n} docs", flush=True)
            if n >= MAX_DOCS:
                break
    print(f"Wrote {n} lines -> {out_path}")


if __name__ == "__main__":
    main()
