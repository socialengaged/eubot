#!/usr/bin/env python3
"""Concatenate JSONL lines {\"text\": ...} into a single .txt (blank line between docs)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_jsonl", type=Path)
    ap.add_argument("output_txt", type=Path)
    args = ap.parse_args()

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input_jsonl, encoding="utf-8") as fin, open(args.output_txt, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text", "")
            if len(t) < 20:
                continue
            fout.write(t.rstrip() + "\n\n")
    print(f"OK -> {args.output_txt}")


if __name__ == "__main__":
    main()
