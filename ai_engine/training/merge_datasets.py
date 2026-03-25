#!/usr/bin/env python3
"""
Weighted merge of clean + hard negatives + sacred QA JSONL for Eurobot Baby training.

Default mix: 50% clean, 30% hard negatives, 20% sacred.
Shuffle + seed. Optional word-drop augmentation (training noise).

Drops lines matching server fallback templates (anti-template).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ai_engine" / "data"

TEMPLATE_DROP = re.compile(
    r"let me give you a simple and clear answer|"
    r"focus on one small action you can take right now",
    re.IGNORECASE,
)


def load_lines(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        print(f"WARN: missing {path}", file=sys.stderr)
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


def _has_template(text: str) -> bool:
    return bool(TEMPLATE_DROP.search(text or ""))


def word_dropout(text: str, p: float, rng: random.Random) -> str:
    """Randomly drop words with probability p (data aug)."""
    words = text.split()
    if not words or p <= 0:
        return text
    kept = [w for w in words if rng.random() > p]
    if len(kept) < max(3, len(words) // 4):
        return text
    return " ".join(kept)


def stratified_sample(
    pool: list[dict],
    n: int,
    *,
    rng: random.Random,
) -> list[dict]:
    if not pool:
        print("WARN: empty pool — add JSONL sources or reduce --total-lines", file=sys.stderr)
        return []
    if n <= len(pool):
        return [json.loads(json.dumps(x)) for x in rng.sample(pool, n)]
    print(f"WARN: need {n} rows but pool has {len(pool)} — sampling with replacement", file=sys.stderr)
    return [json.loads(json.dumps(rng.choice(pool))) for _ in range(n)]


def apply_augment(rows: list[dict], p: float, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        t = str(r.get("text", ""))
        if p > 0 and rng.random() < 0.5:
            t = word_dropout(t, p, rng)
        out.append({"text": t})
    return out


def filter_templates(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not _has_template(str(r.get("text", "")))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, default=DATA / "clean_dataset_v2.jsonl")
    ap.add_argument("--hard", type=Path, default=DATA / "hard_negatives.jsonl")
    ap.add_argument("--sacred", type=Path, default=DATA / "sacred_qa.jsonl")
    ap.add_argument("--out", type=Path, default=DATA / "final_dataset_v3.jsonl")
    ap.add_argument("--total-lines", type=int, default=10_000, help="Target total rows")
    ap.add_argument("--w-clean", type=float, default=0.5)
    ap.add_argument("--w-hard", type=float, default=0.3)
    ap.add_argument("--w-sacred", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--augment-noise",
        type=float,
        default=0.0,
        help="Word-drop probability (0–0.2 typical). TASK 9 optional.",
    )
    args = ap.parse_args()

    s = args.w_clean + args.w_hard + args.w_sacred
    if abs(s - 1.0) > 1e-6:
        print("Weights must sum to 1.0", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    clean = filter_templates(load_lines(args.clean))
    hard = filter_templates(load_lines(args.hard))
    sacred = filter_templates(load_lines(args.sacred))

    total = args.total_lines
    n_clean = round(total * args.w_clean)
    n_hard = round(total * args.w_hard)
    n_sacred = total - n_clean - n_hard

    merged: list[dict] = []
    merged.extend(stratified_sample(clean, n_clean, rng=rng))
    merged.extend(stratified_sample(hard, n_hard, rng=rng))
    merged.extend(stratified_sample(sacred, n_sacred, rng=rng))

    merged = apply_augment(merged, args.augment_noise, rng)
    rng.shuffle(merged)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(merged)} lines -> {args.out} "
        f"(clean={n_clean}, hard={n_hard}, sacred={n_sacred})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
