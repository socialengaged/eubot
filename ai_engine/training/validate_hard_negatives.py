#!/usr/bin/env python3
"""
Validate hard_negatives.jsonl: no generic assistant phrases, no empty answers.
Exit 1 if validation fails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "ai_engine" / "data" / "hard_negatives.jsonl"

BAD_GENERIC = re.compile(
    r"i can help you with (many|various|a lot)|\bask me anything\b|"
    r"\bas an ai language model\b|how can i assist you today",
    re.IGNORECASE,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=DEFAULT)
    args = ap.parse_args()
    if not args.path.is_file():
        print(f"Missing {args.path}", file=sys.stderr)
        return 1
    ok = 0
    bad = 0
    with open(args.path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {i}: JSON error {e}", file=sys.stderr)
                bad += 1
                continue
            t = str(o.get("text") or "")
            if "Assistant:" not in t:
                print(f"Line {i}: missing Assistant:", file=sys.stderr)
                bad += 1
                continue
            m = re.search(r"Assistant:\s*(.+)$", t, re.DOTALL)
            ans = (m.group(1) if m else "").strip()
            if len(ans) < 15:
                print(f"Line {i}: answer too short", file=sys.stderr)
                bad += 1
                continue
            if BAD_GENERIC.search(ans):
                print(f"Line {i}: generic phrase in answer", file=sys.stderr)
                bad += 1
                continue
            ok += 1
    print(f"OK: {ok} rows, bad: {bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
