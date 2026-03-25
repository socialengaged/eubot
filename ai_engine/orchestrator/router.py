"""Rule-based routing (Phase 1: minimal; Phase 3: richer rules)."""
from __future__ import annotations

import re
from enum import Enum
from typing import Any

from ai_engine.config.settings import get_settings
from ai_engine.orchestrator.philosophy_trigger import is_philosophical_query


class RouteDecision(str, Enum):
    BABY = "baby"
    LLM = "llm"
    RAG = "rag"
    HYBRID = "hybrid"


def route_request(context: dict[str, Any]) -> RouteDecision:
    """
    Decide which backend to use. Phase 1: feature flag + simple heuristics.

    context may include:
      - messages: list[{"role","content"}] (OpenAI-style)
      - user_id, session_id (optional)
    """
    settings = get_settings()
    if not settings.ai_engine_enabled:
        try:
            return RouteDecision(settings.default_route.lower())
        except ValueError:
            return RouteDecision.BABY

    if not settings.use_orchestrator:
        dr = settings.default_route.lower()
        try:
            return RouteDecision(dr)
        except ValueError:
            return RouteDecision.BABY

    messages = context.get("messages") or []
    last = ""
    if messages and isinstance(messages[-1], dict):
        last = (messages[-1].get("content") or "").strip()

    # Simple heuristics (expand in Phase 3)
    factual = bool(re.search(r"\b(when|where|what year|who wrote|define|quantify|how many)\b", last, re.I))
    conversational = bool(re.search(r"^(hi|hello|ciao|thanks|thank you)\b", last, re.I)) or len(last) < 20

    if conversational:
        return RouteDecision.LLM
    if is_philosophical_query(last):
        return RouteDecision.RAG
    if factual:
        return RouteDecision.RAG
    return RouteDecision.BABY
