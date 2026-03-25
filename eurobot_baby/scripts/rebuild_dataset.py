#!/usr/bin/env python3
"""
Convenience wrapper: rebuild processed JSONL after adding raw files.
Delegates to build_dataset.py.

Usage:
  python scripts/rebuild_dataset.py
  python scripts/rebuild_dataset.py --chunk_chars 4000 --overlap 200
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    import build_dataset as bd

    sys.argv = [bd.__file__] + sys.argv[1:]
    bd.main()


if __name__ == "__main__":
    main()
