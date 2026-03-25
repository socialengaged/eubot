"""Heuristic filters on API result rows (no training / model changes)."""
from __future__ import annotations

from typing import Any


def is_good_response(r: dict[str, Any]) -> bool:
    """Prefer direct, non-fluffy answers for batch metrics."""
    if not r.get("ok"):
        return False
    if float(r.get("semantic", 0.0)) < 0.5:
        return False
    a = str(r.get("a", ""))
    if len(a) > 400:
        return False
    if "imagine" in a.lower():
        return False
    return True
