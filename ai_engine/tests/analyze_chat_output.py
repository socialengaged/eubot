#!/usr/bin/env python3
"""
Analyze saved chat preview for theatrical tone, repetition, length, structure.

  set PYTHONPATH=.
  python -m ai_engine.tests.analyze_chat_output

Reads ai_engine/tests/chat_preview.txt (generate with run_chat_preview --save).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT = _TESTS_DIR / "chat_preview.txt"

_THEATRICAL_HINTS = ("imagine", "what if", "think about")


def _flags_for_text(text: str) -> list[str]:
    flags: list[str] = []
    t = text.strip()
    if not t:
        return flags
    if len(t) > 300:
        flags.append("TOO_LONG")
    low = t.lower()
    if any(x in low for x in _THEATRICAL_HINTS):
        flags.append("THEATRICAL")
    if t.count("\n") > 5:
        flags.append("OVERSTRUCTURED")
    words = t.split()
    if words:
        ratio = len(set(words)) / len(words)
        if ratio < 0.5:
            flags.append("REPETITIVE")
    return flags


def parse_bot_blocks(content: str) -> list[tuple[int, str, int | None]]:
    """Return list of (index, bot_text, len_from_line_or_None)."""
    out: list[tuple[int, str, int | None]] = []
    # [n] BOT:\n ... \n\n[LEN:nnn | ...
    for m in re.finditer(
        r"\[(\d+)\] BOT:\n(.*?)\n\n\[LEN:(\d+)",
        content,
        re.DOTALL,
    ):
        idx = int(m.group(1))
        bot = m.group(2).strip()
        ln = int(m.group(3))
        out.append((idx, bot, ln))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze chat_preview.txt for style issues.")
    p.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="Path to chat_preview.txt",
    )
    args = p.parse_args()

    path = args.input
    if not path.is_file():
        print(f"FILE NOT FOUND: {path}", file=sys.stderr, flush=True)
        print("Run: python -m ai_engine.tests.run_chat_preview --save", file=sys.stderr, flush=True)
        raise SystemExit(1)

    content = path.read_text(encoding="utf-8", errors="replace")
    blocks = parse_bot_blocks(content)
    if not blocks:
        print("No BOT blocks parsed. Check format (expect [n] BOT: ... [LEN:...]).", flush=True)
        raise SystemExit(1)

    flag_counter: Counter[str] = Counter()

    for i, (_idx, text, _ln) in enumerate(blocks, start=1):
        flags = _flags_for_text(text)
        for f in flags:
            flag_counter[f] += 1

        print(f"RESPONSE {i}:")
        print(f"FLAGS: {', '.join(flags) if flags else '(none)'}")
        preview = text.replace("\n", " ")[:400]
        if len(text) > 400:
            preview += "..."
        print(f"TEXT: {preview}\n")

    print("================ GLOBAL FLAG COUNTS ================")
    for key in ("THEATRICAL", "REPETITIVE", "TOO_LONG", "OVERSTRUCTURED"):
        print(f"{key}: {flag_counter.get(key, 0)}")
    print(f"TOTAL_FLAG_HITS: {sum(flag_counter.values())}")


if __name__ == "__main__":
    main()
