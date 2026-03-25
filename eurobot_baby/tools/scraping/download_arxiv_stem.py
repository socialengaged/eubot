#!/usr/bin/env python3
"""
Fasi 6/7/8 — Subset da KiteFishAI/arxiv-tex-corpus-full.

Il dataset Hub espone tipicamente solo \"id\" e \"text\" (senza \"categories\").
Se \"categories\" e assente, si usa un filtro euristico sulle prime N righe di testo.

Env:
  EUROBOT_ARXIV_OUT     Directory output (default: ./output/arxiv)
  EUROBOT_ARXIV_MODE    physics | math | astro | cs (default: physics)
  EUROBOT_ARXIV_MAX     Max righe JSONL (default: 100000)

Requires: pip install datasets
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

OUTPUT = Path(os.environ.get("EUROBOT_ARXIV_OUT", str(Path(__file__).resolve().parent / "output" / "arxiv")))
OUTPUT.mkdir(parents=True, exist_ok=True)
MODE = os.environ.get("EUROBOT_ARXIV_MODE", "physics").lower().strip()
MAX_ROWS = int(os.environ.get("EUROBOT_ARXIV_MAX", "100000"))

# Se il campo categories esiste (dataset varianti)
PATTERNS: dict[str, re.Pattern[str]] = {
    "physics": re.compile(
        r"(astro-ph|hep-th|hep-ph|gr-qc|quant-ph|cond-mat|nucl-th|nucl-ex|physics\.|hep-ex)",
        re.I,
    ),
    "math": re.compile(r"math\.[A-Z]{2}", re.I),
    "astro": re.compile(r"astro-ph", re.I),
    "cs": re.compile(r"cs\.[A-Z]{2}", re.I),
}

# Fallback euristico sul testo (prime ~4k char)
KW: dict[str, re.Pattern[str]] = {
    "physics": re.compile(
        r"\b(quantum|electromagnetic|thermodynamic|relativity|particle|superconduct|"
        r"condensed matter|gravitat|laser|plasma|semiconductor|optics)\b",
        re.I,
    ),
    "math": re.compile(
        r"(\\begin\{theorem\}|\\begin\{proof\}|\btheorem\b|\blemma\b|homology|manifold|"
        r"functor|category theory|\balgebra\b|\btopology\b)",
        re.I,
    ),
    "astro": re.compile(
        r"\b(galaxy|galaxies|cosmolog|stellar|supernova|black hole|dark matter|"
        r"neutron star|exoplanet|astrophys)\b",
        re.I,
    ),
    "cs": re.compile(
        r"\b(algorithm|complexity|computability|compiler|programming language|software engineering|"
        r"database|distributed system|cryptograph|operating system|machine learning|neural network|"
        r"graph theory|formal verification|lambda calculus)\b",
        re.I,
    ),
}


def row_text(row: dict) -> str:
    for k in ("text", "content", "body"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def match_row(row: dict, mode: str) -> bool:
    cats = str(row.get("categories", "") or "")
    if cats and mode in PATTERNS and PATTERNS[mode].search(cats):
        return True
    t = row_text(row)
    head = t[:8000]
    return bool(KW[mode].search(head)) if mode in KW else False


def main() -> None:
    from datasets import load_dataset

    if MODE not in PATTERNS:
        raise SystemExit(f"Unknown EUROBOT_ARXIV_MODE={MODE!r}; use physics|math|astro|cs")

    ds_name = "KiteFishAI/arxiv-tex-corpus-full"
    print(f"Streaming {ds_name}, mode={MODE}, max={MAX_ROWS} (categories+keyword fallback) ...")
    ds = load_dataset(ds_name, split="train", streaming=True)
    out_path = OUTPUT / f"arxiv_{MODE}.jsonl"
    n = 0
    scanned = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            scanned += 1
            if not match_row(row, MODE):
                continue
            t = row_text(row)
            if len(t) < 80:
                continue
            cats = str(row.get("categories", "") or "")
            f.write(json.dumps({"text": t, "categories": cats}, ensure_ascii=False) + "\n")
            n += 1
            if n % 2000 == 0:
                print(f"  ... kept {n} (scanned {scanned})", flush=True)
            if n >= MAX_ROWS:
                break
            if scanned > MAX_ROWS * 200 and n == 0:
                print("WARN: no matches in first scans; relax keywords or check dataset")
                break
    print(f"Wrote {n} lines -> {out_path} (scanned {scanned} rows)")


if __name__ == "__main__":
    main()
