#!/usr/bin/env python3
"""Download training corpora via HuggingFace datasets; save plain text under data/raw/.

Modes:
  --mode wiki      WikiText-103 only (~15 MB, fast)
  --mode large     WikiText-103 + OpenWebText subset (~500 MB–1 GB)
  --mode full      WikiText-103 + full OpenWebText (~6 GB)

Italian (raw text for tokenizer / LM from scratch):
  --wikipedia_it N   Append N articles from Wikipedia IT (HF wikipedia 20220301.it)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from datasets import load_dataset


RAW_DIR = ROOT / "data" / "raw"


def save_texts(texts: list[str], path: Path) -> None:
    path.write_text("\n\n".join(texts), encoding="utf-8")
    mb = path.stat().st_size / 1e6
    print(f"  {path.name}: {len(texts)} docs, {mb:.1f} MB")


def download_wikitext() -> tuple[list[str], list[str]]:
    print("Downloading WikiText-103 ...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train, val = [], []
    for ex in ds["train"]:
        t = (ex.get("text") or "").strip()
        if len(t) >= 30:
            train.append(t)
    for ex in ds["validation"]:
        t = (ex.get("text") or "").strip()
        if len(t) >= 30:
            val.append(t)
    print(f"  WikiText: {len(train)} train, {len(val)} val docs")
    return train, val


def download_wikipedia_it(max_articles: int) -> list[str]:
    """Italian Wikipedia articles (streaming) for bilingual LM training."""
    print(f"Downloading Wikipedia IT (max_articles={max_articles}) …")
    ds = load_dataset("wikipedia", "20220301.it", split="train", streaming=True)
    texts: list[str] = []
    for i, ex in enumerate(ds):
        if i >= max_articles:
            break
        t = (ex.get("text") or "").strip()
        if len(t) >= 100:
            texts.append(t)
        if (i + 1) % 20_000 == 0:
            print(f"    ... {i + 1} scanned, {len(texts)} kept")
    print(f"  Wikipedia IT: {len(texts)} docs kept")
    return texts


def download_openwebtext(max_docs: int | None = None) -> list[str]:
    print(f"Downloading OpenWebText (max_docs={max_docs or 'all'}) ...")
    ds = load_dataset("openwebtext", split="train", streaming=True)
    texts = []
    for i, ex in enumerate(ds):
        if max_docs and i >= max_docs:
            break
        t = (ex.get("text") or "").strip()
        if len(t) >= 50:
            texts.append(t)
        if (i + 1) % 50_000 == 0:
            print(f"    ... {i + 1} rows processed, {len(texts)} kept")
    print(f"  OpenWebText: {len(texts)} docs kept")
    return texts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["wiki", "large", "full"], default="large",
                   help="wiki=WikiText only, large=+OWT 200k docs, full=+OWT all")
    p.add_argument(
        "--wikipedia_it",
        type=int,
        default=0,
        help="If >0, append N Italian Wikipedia articles (adds Italian comprehension for brain-zero)",
    )
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    wiki_train, wiki_val = download_wikitext()

    owt_texts: list[str] = []
    if args.mode == "large":
        owt_texts = download_openwebtext(max_docs=200_000)
    elif args.mode == "full":
        owt_texts = download_openwebtext(max_docs=None)

    all_train = wiki_train + owt_texts

    if args.wikipedia_it > 0:
        all_train = all_train + download_wikipedia_it(args.wikipedia_it)

    save_texts(all_train, RAW_DIR / "train.txt")
    save_texts(wiki_val, RAW_DIR / "val.txt")

    total_mb = sum(f.stat().st_size for f in RAW_DIR.glob("*.txt")) / 1e6
    print(f"\nTotal raw data: {total_mb:.1f} MB in {RAW_DIR}")
    print("Done. Next: python scripts/build_dataset.py")


if __name__ == "__main__":
    main()
