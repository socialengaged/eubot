"""rag_real falls back when index missing."""
from __future__ import annotations

from ai_engine.config.settings import clear_settings_cache
from ai_engine.rag.retriever import reset_cache


def test_rag_real_fallback_without_index(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_RAG_INDEX_PATH", "")
    monkeypatch.setenv("AI_ENGINE_RAG_USE_REAL", "1")
    clear_settings_cache()
    reset_cache()

    from ai_engine.providers.rag_real import call_rag

    out = call_rag("hello world")
    assert out["chunks"] == []
    assert out["context"] == ""
    assert out["query"] == "hello world"


def test_rag_real_stub_when_flag_off(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_RAG_USE_REAL", "0")
    clear_settings_cache()
    reset_cache()

    from ai_engine.providers.rag_real import call_rag

    out = call_rag("x")
    assert out["chunks"] == []
