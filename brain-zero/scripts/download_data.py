#!/usr/bin/env python3
"""Download WikiText-103 (raw) via HuggingFace datasets; save plain text files under data/raw/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split_train_rows",
        type=int,
        default=50_000,
        help="Max training rows to keep (reduce for faster pipeline test).",
    )
    p.add_argument(
        "--split_val_rows",
        type=int,
        default=5_000,
        help="Max validation rows.",
    )
    args = p.parse_args()

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Loading wikitext-103-raw-v1 …")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    def write_split(name: str, split_name: str, max_rows: int) -> None:
        rows = ds[split_name]
        lines = []
        for i, ex in enumerate(rows):
            if i >= max_rows:
                break
            t = (ex.get("text") or "").strip()
            if len(t) < 20:
                continue
            lines.append(t)
        path = raw_dir / f"{name}.txt"
        path.write_text("\n\n".join(lines), encoding="utf-8")
        print(f"Wrote {len(lines)} docs -> {path} ({path.stat().st_size / 1e6:.2f} MB)")

    write_split("train", "train", args.split_train_rows)
    write_split("val", "validation", args.split_val_rows)
    print("Done.")


if __name__ == "__main__":
    main()
