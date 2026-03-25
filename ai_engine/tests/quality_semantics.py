"""Lightweight answer semantics (no external AI). Shared by test_live_safe and test_quality_batch."""
from __future__ import annotations


def semantic_score(text: str) -> float:
    """Return 0..1 heuristic score."""
    score = 0
    if "." in text:
        score += 1
    if len(text.split()) > 20:
        score += 1
    low = text.lower()
    if any(w in low for w in ["because", "why", "how", "means", "example"]):
        score += 1
    if "\n" in text:
        score += 1
    words = text.split()
    denom = max(len(words), 1)
    if len(set(words)) / denom > 0.6:
        score += 1
    return score / 5.0
