"""
Lightweight heuristic: user message likely philosophical / wisdom / ethics.
Used by gateway router and Baby serve.py for sacred RAG retrieval.
"""
from __future__ import annotations

import re

# Minimum length so "hi" does not trigger
_MIN_LEN = 12

_PHILOSOPHY_RE = re.compile(
    r"\b("
    r"philosoph|philosophy|philosophical|stoic|stoicism|epicure|plato|aristotle|"
    r"socrates|kant|nietzsche|heidegger|existential|ethics|ethical|virtue|"
    r"meaning of life|free will|metaphys|ontology|epistemolog|theodicy|"
    r"vedanta|upanishad|buddhis|dharma|karma|nirvana|tao|confucius|"
    r"seneca|marcus aurelius|spinoza|descartes|libero arbitrio|senso della vita|"
    r"filosofia|etica|metafisica|essere e nulla|cosa è la virtù"
    r")\b",
    re.IGNORECASE,
)


def is_philosophical_query(text: str, *, min_len: int = _MIN_LEN) -> bool:
    t = (text or "").strip()
    if len(t) < min_len:
        return False
    return bool(_PHILOSOPHY_RE.search(t))
