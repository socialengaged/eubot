"""Feature flags and URLs for ai_engine (env-based, no required .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration; override via environment variables."""

    ai_engine_enabled: bool = field(default_factory=lambda: _bool("AI_ENGINE_ENABLED", True))
    use_orchestrator: bool = field(default_factory=lambda: _bool("AI_ENGINE_USE_ORCHESTRATOR", False))
    default_route: str = field(default_factory=lambda: _str("AI_ENGINE_DEFAULT_ROUTE", "baby") or "baby")
    baby_base_url: str = field(
        default_factory=lambda: _str("BABY_BASE_URL", "http://127.0.0.1:8080") or "http://127.0.0.1:8080"
    )
    baby_model: str = field(default_factory=lambda: _str("BABY_MODEL", "eurobot-baby") or "eurobot-baby")
    openai_api_key: str = field(default_factory=lambda: _str("OPENAI_API_KEY", ""))
    openai_base_url: str = field(
        default_factory=lambda: _str("OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    )
    llm_stub_mode: bool = field(default_factory=lambda: _bool("AI_ENGINE_LLM_STUB_MODE", True))
    # Directory containing index.faiss + metadata.pkl (from ai_engine.rag.ingest)
    rag_index_path: str = field(default_factory=lambda: _str("AI_ENGINE_RAG_INDEX_PATH", ""))
    rag_use_real: bool = field(default_factory=lambda: _bool("AI_ENGINE_RAG_USE_REAL", True))
    # local = sentence-transformers; openai = OpenAI embeddings API (same key as LLM)
    rag_embed_backend: str = field(default_factory=lambda: _str("AI_ENGINE_RAG_EMBED_BACKEND", "local") or "local")
    rag_embed_model: str = field(
        default_factory=lambda: _str("AI_ENGINE_RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    openai_embed_model: str = field(
        default_factory=lambda: _str("OPENAI_EMBED_MODEL", "text-embedding-3-small") or "text-embedding-3-small"
    )
    # Who answers after retrieval: baby (GPU) or llm (OpenAI when stub off)
    rag_answer_target: str = field(default_factory=lambda: _str("AI_ENGINE_RAG_ANSWER_TARGET", "baby") or "baby")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    """Call from tests after changing os.environ."""
    get_settings.cache_clear()
