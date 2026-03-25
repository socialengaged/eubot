#!/usr/bin/env python3
"""
Produce blob .txt files expected by build_dataset TRAIN_PARTS_OPTIONAL from scraping output dirs.

Usage (on pod or scraper):
  python merge_outputs_to_raw_names.py --out /workspace/eurobot_baby/data/raw
  # Durante training attivo: preferire --out .../data/raw_staging (vedi SETUP_SCRAPER.md)

Reads:
  output/gutenberg_esoteric/*.txt
  output/sacred_texts/**/*.txt (08: tutto tranne sottocartelle astrology/ e freemasonry/; quelle -> 43 e 44)
  output/gnosis_retry/*.txt
  output/physics/*.jsonl
  output/math/openwebmath_subset.jsonl
  output/arxiv/arxiv_physics.jsonl, arxiv_math.jsonl, arxiv_astro.jsonl, arxiv_cs.jsonl
  output/gutenberg_science/*.txt
  output/gutenberg_theology/*.txt
  output/astro/astro_texts_hf.jsonl

Writes:
  08_esoteric_expanded.txt, 20_physics_corpus.txt, ...
  60_theology_corpus.txt (da output/gutenberg_theology/), 70_informatics_corpus.txt (da arxiv_cs.jsonl)
  43_astrology_corpus.txt, 44_masonry_corpus.txt (sottosezioni sacred-texts)
  71_gutenberg_computing.txt (da output/gutenberg_computing/)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl_texts(p: Path) -> str:
    parts: list[str] = []
    if not p.is_file():
        return ""
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                t = o.get("text", "")
                if len(t) > 20:
                    parts.append(t)
            except json.JSONDecodeError:
                continue
    return "\n\n".join(parts)


def cat_txt_glob(root: Path, pattern: str) -> str:
    texts: list[str] = []
    for fp in sorted(root.glob(pattern)):
        if fp.is_file() and fp.suffix.lower() == ".txt":
            try:
                texts.append(fp.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    return "\n\n".join(texts)


def cat_sacred_subdir(st: Path, sub: str) -> str:
    d = st / sub
    if not d.is_dir():
        return ""
    texts: list[str] = []
    for fp in sorted(d.rglob("*.txt")):
        try:
            texts.append(fp.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n\n".join(texts)


def cat_sacred_excluding_subdirs(st: Path, exclude: frozenset[str]) -> str:
    """Concatena sacred_texts/*.txt escludendo le sottocartelle in exclude (primo livello)."""
    if not st.is_dir():
        return ""
    texts: list[str] = []
    for fp in sorted(st.rglob("*.txt")):
        try:
            rel = fp.relative_to(st)
        except ValueError:
            continue
        top = rel.parts[0] if rel.parts else ""
        if top in exclude:
            continue
        try:
            texts.append(fp.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n\n".join(texts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("."), help="Scraping output base (e.g. /workspace/eurobot_scraping_run)")
    ap.add_argument("--out", type=Path, required=True, help="data/raw destination")
    args = ap.parse_args()
    base = args.base
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # Esoteric blob
    guten = cat_txt_glob(base / "output" / "gutenberg_esoteric", "*.txt")
    st = base / "output" / "sacred_texts"
    _excl = frozenset({"astrology", "freemasonry"})
    sacred = cat_sacred_excluding_subdirs(st, _excl)
    astro_only = cat_sacred_subdir(st, "astrology")
    mason_only = cat_sacred_subdir(st, "freemasonry")
    if astro_only.strip():
        (out / "43_astrology_corpus.txt").write_text(astro_only, encoding="utf-8", errors="replace")
        print("Wrote 43_astrology_corpus.txt (sacred-texts astrology/)")
    if mason_only.strip():
        (out / "44_masonry_corpus.txt").write_text(mason_only, encoding="utf-8", errors="replace")
        print("Wrote 44_masonry_corpus.txt (sacred-texts freemasonry/)")
    gnosis = cat_txt_glob(base / "output" / "gnosis_retry", "*.txt")
    blob08 = "\n\n".join(x for x in (guten, sacred, gnosis) if x.strip())
    if blob08.strip():
        (out / "08_esoteric_expanded.txt").write_text(blob08, encoding="utf-8", errors="replace")
        print(f"Wrote 08_esoteric_expanded.txt ({len(blob08)//1024} KB)")

    phys_dir = base / "output" / "physics"
    phys_texts: list[str] = []
    if phys_dir.is_dir():
        for jp in sorted(phys_dir.glob("*.jsonl")):
            phys_texts.append(read_jsonl_texts(jp))
    phys_arxiv = read_jsonl_texts(base / "output" / "arxiv" / "arxiv_physics.jsonl")
    ptxt = "\n\n".join(x for x in phys_texts + [phys_arxiv] if x.strip())
    if ptxt.strip():
        (out / "20_physics_corpus.txt").write_text(ptxt, encoding="utf-8", errors="replace")
    if phys_texts:
        (out / "21_physics_textbooks_hf.txt").write_text(phys_texts[0], encoding="utf-8", errors="replace")
    if phys_arxiv.strip():
        (out / "22_arxiv_physics.txt").write_text(phys_arxiv, encoding="utf-8", errors="replace")
    if ptxt.strip() or phys_arxiv.strip():
        print("Wrote physics blobs")

    mtxt = read_jsonl_texts(base / "output" / "math" / "openwebmath_subset.jsonl")
    if mtxt.strip():
        (out / "30_math_openwebmath_subset.txt").write_text(mtxt, encoding="utf-8", errors="replace")
    mclassic = cat_txt_glob(base / "output" / "gutenberg_science", "*.txt")
    if mclassic.strip():
        (out / "31_math_classics.txt").write_text(mclassic, encoding="utf-8", errors="replace")
    marxiv = read_jsonl_texts(base / "output" / "arxiv" / "arxiv_math.jsonl")
    if marxiv.strip():
        (out / "32_arxiv_math.txt").write_text(marxiv, encoding="utf-8", errors="replace")

    atxt = read_jsonl_texts(base / "output" / "astro" / "astro_texts_hf.jsonl")
    if atxt.strip():
        (out / "41_astronomy_astro_texts_hf.txt").write_text(atxt, encoding="utf-8", errors="replace")
    aph = read_jsonl_texts(base / "output" / "arxiv" / "arxiv_astro.jsonl")
    if aph.strip():
        (out / "42_arxiv_astroph.txt").write_text(aph, encoding="utf-8", errors="replace")
    acorp = "\n\n".join(x for x in (atxt, aph) if x.strip())
    if acorp.strip():
        (out / "40_astronomy_corpus.txt").write_text(acorp, encoding="utf-8", errors="replace")

    theo = cat_txt_glob(base / "output" / "gutenberg_theology", "*.txt")
    if theo.strip():
        (out / "60_theology_corpus.txt").write_text(theo, encoding="utf-8", errors="replace")
        print("Wrote 60_theology_corpus.txt")

    cs_arxiv = read_jsonl_texts(base / "output" / "arxiv" / "arxiv_cs.jsonl")
    if cs_arxiv.strip():
        (out / "70_informatics_corpus.txt").write_text(cs_arxiv, encoding="utf-8", errors="replace")
        print("Wrote 70_informatics_corpus.txt (arxiv cs)")

    gcomp = cat_txt_glob(base / "output" / "gutenberg_computing", "*.txt")
    if gcomp.strip():
        (out / "71_gutenberg_computing.txt").write_text(gcomp, encoding="utf-8", errors="replace")
        print("Wrote 71_gutenberg_computing.txt")

    print("Done merge_outputs_to_raw_names.py ->", out)


if __name__ == "__main__":
    main()
