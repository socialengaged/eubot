#!/usr/bin/env python3
"""
Offline + optional HTTP quality metrics for Baby-style assistant text.

Metrics:
- repetition_score (max run length of same word)
- unique_ratio
- fallback_rate (substring match vs run_chat_preview markers)
- response_variance (same prompt, N HTTP calls — optional)

Usage:
  python -m ai_engine.tests.eval_quality
  python -m ai_engine.tests.eval_quality --url https://eubot.seo.srl/api/baby/v1/chat/completions --samples 5
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Keep in sync with run_chat_preview._FALLBACK_MARKERS
_FALLBACK_MARKERS = (
    "Ti rispondo in modo semplice",
    "Ecco una spiegazione chiara",
    "Puoi spiegarmi meglio",
    "Let me give you a simple",
    "Sto avendo un attimo",
    "focus on one small action",
    "Ok, dimmi meglio",
)

TEMPLATE_BAD = (
    "let me give you a simple and clear answer",
)

FIXED_PROMPTS = (
    "which action",
    "for example",
    "who are you",
)


def repetition_score(text: str) -> int:
    words = text.split()
    if len(words) < 2:
        return 0
    best = run = 1
    for i in range(1, len(words)):
        if words[i].lower() == words[i - 1].lower():
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def unique_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return len(set(w.lower() for w in words)) / len(words)


def fallback_rate_simulated(answers: list[str]) -> float:
    if not answers:
        return 0.0
    n = sum(1 for a in answers if any(m in (a or "") for m in _FALLBACK_MARKERS))
    return 100.0 * n / len(answers)


def _post_v1(url: str, user: str, timeout: float) -> str:
    import httpx

    payload: dict[str, Any] = {"messages": [{"role": "user", "content": user}]}
    with httpx.Client(timeout=timeout) as c:
        r = c.post(url, json=payload, headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            return ""
        data = r.json()
        ch = data.get("choices") or []
        if not ch:
            return ""
        msg = ch[0].get("message") or {}
        return str(msg.get("content") or "")


def response_variance(url: str, prompt: str, n: int, timeout: float) -> dict[str, float]:
    texts: list[str] = []
    for _ in range(n):
        t = _post_v1(url, prompt, timeout)
        if t:
            texts.append(t)
    if len(texts) < 2:
        return {"n": float(len(texts)), "len_std": 0.0, "pairwise_edit_avg": 0.0}
    lengths = [len(t) for t in texts]
    # Cheap "variance": std of lengths + char diversity
    lens = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    # Normalized Hamming-ish: average fraction of positions that differ min len
    def diff_ratio(a: str, b: str) -> float:
        if not a and not b:
            return 0.0
        m = min(len(a), len(b))
        if m == 0:
            return 1.0
        return sum(1 for i in range(m) if a[i] != b[i]) / m

    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs.append(diff_ratio(texts[i], texts[j]))
    ped = statistics.mean(pairs) if pairs else 0.0
    return {"n": float(len(texts)), "len_std": float(lens), "pairwise_edit_avg": float(ped)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="", help="Optional Baby v1 URL for live variance tests")
    ap.add_argument("--samples", type=int, default=5, help="Repeated calls per fixed prompt")
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()

    # Offline demo on fixed strings
    demo = [
        "Pick one task and start it for five minutes.",
        "Let me give you a simple and clear answer: focus on one small action.",
        "bad bad bad bad bad bad",
    ]
    print("=== Offline metrics (demo strings) ===", flush=True)
    for t in demo:
        print(
            f"  repetition_score={repetition_score(t)} unique_ratio={unique_ratio(t):.3f} "
            f"fallback_like={any(m in t for m in _FALLBACK_MARKERS) or any(x in t.lower() for x in TEMPLATE_BAD)}",
            flush=True,
        )

    print(f"  fallback_rate_simulated(demo)={fallback_rate_simulated(demo):.1f}%", flush=True)

    if not args.url.strip():
        print("\nNo --url: skip HTTP variance. Pass --url for live tests.", flush=True)
        return 0

    print("\n=== HTTP variance (fixed prompts) ===", flush=True)
    url = args.url.strip()
    for p in FIXED_PROMPTS:
        stats = response_variance(url, p, args.samples, args.timeout)
        print(f"  prompt={p!r} -> {stats}", flush=True)

    # Simple assertion: answers should not all be identical template
    bad = 0
    for p in FIXED_PROMPTS:
        texts = []
        for _ in range(min(3, args.samples)):
            t = _post_v1(url, p, args.timeout)
            if t:
                texts.append(t)
        if len(texts) >= 2:
            if all("Let me give you a simple" in x for x in texts):
                bad += 1
    if bad:
        print(f"WARN: {bad} prompts produced only template-like answers in quick check.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
