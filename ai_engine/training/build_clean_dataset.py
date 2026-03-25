#!/usr/bin/env python3
"""
Build cleaned JSONL for Eurobot Baby causal LM from:
- Original JSONL (lines with {\"text\": ...} or {\"messages\": [...]})
- Optional conversation logs (JSONL with messages or user/assistant fields)

Filters: repeated sentences, duplicate answers across inputs, server fallback templates.
Output: ai_engine/data/clean_dataset_v2.jsonl (one response per normalized user input).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ai_engine" / "data"

TEMPLATE_PATTERNS = re.compile(
    r"let me give you a simple and clear answer",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _norm_sentence(s: str) -> str:
    return _norm(s)


def _has_repeated_sentence(text: str) -> bool:
    """Same sentence (normalized) appears more than once in the same blob."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    seen: set[str] = set()
    for p in parts:
        n = _norm_sentence(p)
        if len(n) < 12:
            continue
        if n in seen:
            return True
        seen.add(n)
    return False


def _assistant_from_messages(obj: dict) -> str:
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if not isinstance(m, dict):
                continue
            if (m.get("role") or "").lower() == "assistant":
                return str(m.get("content") or "")
    return str(obj.get("assistant") or obj.get("output") or "")


def _user_from_messages(obj: dict) -> str:
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if not isinstance(m, dict):
                continue
            if (m.get("role") or "").lower() == "user":
                return str(m.get("content") or "")
    return str(obj.get("user") or obj.get("input") or "")


def _row_to_training_text(obj: dict) -> str | None:
    if "text" in obj and isinstance(obj["text"], str) and obj["text"].strip():
        t = obj["text"].strip()
        if "User:" in t or "Assistant:" in t:
            return t
        # plain text chunk — keep as single-turn style
        return f"User: .\nAssistant: {t}"
    u = _user_from_messages(obj).strip()
    a = _assistant_from_messages(obj).strip()
    if not u or not a:
        return None
    return f"User: {u}\nAssistant: {a}"


def _hash_input(user_text: str) -> str:
    return hashlib.sha256(_norm(user_text).encode()).hexdigest()[:24]


def _hash_answer(ans: str) -> str:
    return hashlib.sha256(_norm(ans).encode()).hexdigest()[:24]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_json_array(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(raw, list):
        out: list[dict] = []
        for x in raw:
            if isinstance(x, dict):
                out.append(x)
            # Skip bare prompt strings (no assistant) — not usable here
        return out
    return []


def _split_user_assistant_from_text(text: str) -> tuple[str, str]:
    m = re.search(
        r"User:\s*(.+?)\nAssistant:\s*(.+)$",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text


def clean_and_merge(
    originals: list[dict],
    logs: list[dict],
    *,
    max_chars: int,
) -> list[str]:
    # Parse all into (text, user_key, answer_only)
    candidates: list[tuple[str, str, str]] = []
    for obj in originals + logs:
        text = _row_to_training_text(obj)
        if not text:
            continue
        u = ""
        a = ""
        if "messages" in obj or "user" in obj:
            u = _user_from_messages(obj).strip()
            a = _assistant_from_messages(obj).strip()
        if not u and not a and "User:" in text:
            u, a = _split_user_assistant_from_text(text)
        elif not a and "Assistant:" in text:
            u, a = _split_user_assistant_from_text(text)
        if not u:
            u = "(prompt)"
        if TEMPLATE_PATTERNS.search(text) or TEMPLATE_PATTERNS.search(a):
            continue
        if _has_repeated_sentence(a or text):
            continue
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        candidates.append((text, _hash_input(u or text), a or text))

    # One answer per input (first wins)
    by_input: dict[str, tuple[str, str]] = {}
    for text, uk, ans in candidates:
        if uk not in by_input:
            by_input[uk] = (text, ans)

    # Dedup: same answer hash -> keep one (first input order)
    seen_ans: set[str] = set()
    out: list[str] = []
    for uk in sorted(by_input.keys()):
        text, ans = by_input[uk]
        ah = _hash_answer(ans)
        if ah in seen_ans:
            continue
        seen_ans.add(ah)
        out.append(text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--original",
        type=Path,
        action="append",
        default=[],
        help="JSONL or JSON array file (repeatable)",
    )
    ap.add_argument("--logs", type=Path, default=None, help="Optional JSONL conversation logs")
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA / "clean_dataset_v2.jsonl",
    )
    ap.add_argument("--max-chars", type=int, default=8000)
    args = ap.parse_args()

    originals: list[dict] = []
    for p in args.original:
        p = p.resolve()
        if p.suffix.lower() == ".jsonl":
            originals.extend(load_jsonl(p))
        else:
            originals.extend(load_json_array(p))

    logs = load_jsonl(args.logs.resolve()) if args.logs else []

    texts = clean_and_merge(originals, logs, max_chars=args.max_chars)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(texts)} lines -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
