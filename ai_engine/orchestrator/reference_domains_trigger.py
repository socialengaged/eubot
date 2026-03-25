"""
Heuristic: query likely benefits from domain RAG (medicine, psychiatry, psychology,
anatomy, clinical) — distinct from sacred/philosophy trigger.
"""
from __future__ import annotations

import re

_MIN_LEN = 10

_EXPANSION_RE = re.compile(
    r"\b("
    r"symptom|diagnos|therapy|therapist|psychiatr|psycholog|mental health|depression|anxiety|"
    r"trauma|ptsd|ocd|bipolar|schizophren|medication|prescription|dosage|side effect|"
    r"clinical trial|patient|anatomy|muscle|skeleton|heart|lung|liver|kidney|brain|neuron|"
    r"nerve|spine|artery|vein|cell|organ|patholog|infection|virus|bacteria|vaccine|"
    r"diabetes|hypertension|cancer|tumor|radiolog|surgery|emergency room|icu|"
    r"medical|medicina|farmaco|sintomo|psicologo|psichiatria|ansia|depressione|"
    r"anatomia|corpo umano|muscolo|osso"
    r")\b",
    re.IGNORECASE,
)


def is_domain_reference_query(text: str, *, min_len: int = _MIN_LEN) -> bool:
    t = (text or "").strip()
    if len(t) < min_len:
        return False
    return bool(_EXPANSION_RE.search(t))
