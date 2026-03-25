#!/usr/bin/env python3
"""Prompt mutation for quality feedback loops (no training)."""
from __future__ import annotations

import random
from typing import Any

from ai_engine.tests.quality_semantics import semantic_score


def contextual_mutation(p: str) -> list[str]:
    p = p.strip()
    return [
        f"Give a real-world example of: {p}",
        f"Explain step by step: {p}",
        f"What most people misunderstand about: {p}",
        f"Explain like I'm smart but new to this: {p}",
        f"Break this down clearly: {p}",
    ]


def _basic_templates(p: str) -> list[str]:
    p = p.strip()
    return [
        f"Explain deeper: {p}",
        f"Make this more practical: {p}",
        f"Give a more insightful answer: {p}",
        f"What is the hidden truth behind: {p}",
        f"Simplify but keep depth: {p}",
    ]


def _concrete_templates(p: str) -> list[str]:
    p = p.strip()
    return [
        f"Give a direct and practical answer: {p}",
        f"Answer briefly and clearly: {p}",
        f"No fluff, explain: {p}",
    ]


def all_mutations(p: str) -> list[str]:
    return _basic_templates(p) + contextual_mutation(p) + _concrete_templates(p)


def mutate_prompt(p: str) -> str:
    return random.choice(all_mutations(p))


def _diversity_ok(prompts: list[str]) -> bool:
    if not prompts:
        return False
    return len(set(prompts)) >= len(prompts) * 0.7


def _effective_high(r: dict[str, Any]) -> bool:
    if not r.get("ok"):
        return False
    qlt = str(r.get("qlt", "LOW"))
    sem = float(r.get("semantic", semantic_score(str(r.get("a", "")))))
    if qlt == "HIGH" and sem < 0.4:
        return False
    return qlt == "HIGH"


def build_improved_dataset(rows: list[dict[str, Any]], *, max_prompts: int = 20) -> list[str]:
    """
    Use effective HIGH rows (discard fake HIGH: long but semantic<0.4).
    Prioritize prompts whose answer had semantic > 0.6, then mutate (basic + contextual).
    """
    high_samples = [r for r in rows if _effective_high(r)]
    if not high_samples:
        return []

    def sort_key(r: dict[str, Any]) -> tuple[int, float]:
        sem = float(r.get("semantic", semantic_score(str(r.get("a", "")))))
        pri = 1 if sem > 0.6 else 0
        return (pri, sem)

    high_samples.sort(key=sort_key, reverse=True)

    high_q = [str(r.get("q", "")).strip() for r in high_samples if str(r.get("q", "")).strip()]
    if not high_q:
        return []

    max_prompts = max(1, min(20, int(max_prompts)))
    merged: list[str] = []
    for attempt in range(8):
        mutated = [mutate_prompt(q) for q in high_q]
        merged = list(dict.fromkeys(high_q + mutated))[:max_prompts]
        if _diversity_ok(merged):
            break

    return merged[:max_prompts]
