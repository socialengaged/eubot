"""
Eurobot Baby — modular AI engine (orchestrator, RAG, memory, providers).

Phase 1: stubs + HTTP client to existing Baby serve.py. Does not replace training/inference core.
"""

from ai_engine.config.settings import Settings, clear_settings_cache, get_settings
from ai_engine.orchestrator.router import RouteDecision, route_request
from ai_engine.orchestrator.providers import call_baby, call_llm, call_rag, run_orchestrated_chat

__all__ = [
    "Settings",
    "get_settings",
    "clear_settings_cache",
    "RouteDecision",
    "route_request",
    "call_baby",
    "call_llm",
    "call_rag",
    "run_orchestrated_chat",
]
