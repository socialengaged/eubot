#!/usr/bin/env python3
"""
Generate a JSONL-style prompt list (array of strings) for a topic.

Examples: mindset, crypto, productivity, philosophy

  set PYTHONPATH=.
  python -m ai_engine.tests.generate_dataset --topic mindset --n 10
  python -m ai_engine.tests.generate_dataset --topic productivity --out custom.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_DATASETS_DIR = _TESTS_DIR / "datasets"

TEMPLATES: list[str] = [
    "Give a deep but simple explanation of {topic}",
    "What is a non-obvious truth about {topic}?",
    "How do people misunderstand {topic}?",
    "What is a practical tip about {topic}?",
    "Why does {topic} matter in daily life?",
    "What is one mistake beginners make with {topic}?",
    "Explain {topic} to someone in a hurry",
    "What would change your view of {topic}?",
    "What is a counterintuitive idea about {topic}?",
    "How can someone get better at {topic} step by step?",
    "Explain {topic} in a counterintuitive way",
    "What is a hard truth about {topic}?",
    "How does {topic} fail in real life?",
    "What separates good vs great in {topic}?",
    "Explain {topic} with a clear real example",
    "Break down {topic} step by step",
    "What is the real mechanism behind {topic}?",
    "Explain cause and effect in {topic}",
    "Explain {topic} in a direct and practical way",
    "Give a concise explanation of {topic}",
]


def _slug(topic: str) -> str:
    s = topic.strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    return s.strip("_") or "topic"


def slug_topic(topic: str) -> str:
    """Public filename slug for a topic (used by pipeline and CLI)."""
    return _slug(topic)


def generate_dataset(
    topic: str,
    n: int = 10,
    pattern_hints: list[str] | None = None,
) -> list[str]:
    """
    Return up to n unique-ish prompt strings from randomized templates.
    If pattern_hints is set (e.g. from last BEST PATTERN), templates that match those
    prefixes are oversampled to bias the next generation.
    """
    topic = topic.strip()
    if not topic:
        raise ValueError("topic must be non-empty")
    n = max(1, min(100, n))
    tpl_pool = list(TEMPLATES)
    hints = [h.strip() for h in (pattern_hints or []) if h and str(h).strip()]
    if hints:
        expanded: list[str] = []
        for tpl in tpl_pool:
            tf = tpl.format(topic=topic)
            w = 1
            for h in hints:
                hl = min(len(h), 16)
                if hl >= 6 and (tf[:hl].lower() == h[:hl].lower() or h[:8].lower() in tf[:32].lower()):
                    w = 3
                    break
            expanded.extend([tpl] * w)
        tpl_pool = expanded
    if n <= len(tpl_pool):
        chosen = random.sample(tpl_pool, k=n)
    else:
        chosen = random.sample(tpl_pool, k=len(tpl_pool))
        chosen.extend(random.choices(tpl_pool, k=n - len(tpl_pool)))
    random.shuffle(chosen)
    return [tpl.format(topic=topic) for tpl in chosen]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate convo_auto_<topic>.json under tests/datasets/")
    p.add_argument("--topic", required=True, help="Topic word or short phrase (e.g. mindset, crypto)")
    p.add_argument("--n", type=int, default=10, help="Number of prompts (default 10)")
    p.add_argument("--out", type=Path, default=None, help="Output path (default: datasets/convo_auto_<topic>.json)")
    args = p.parse_args()

    prompts = generate_dataset(args.topic, args.n)
    if args.out:
        out_path = args.out
    else:
        _DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _DATASETS_DIR / f"convo_auto_{_slug(args.topic)}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(prompts)} prompts to {out_path}", flush=True)


if __name__ == "__main__":
    main()
