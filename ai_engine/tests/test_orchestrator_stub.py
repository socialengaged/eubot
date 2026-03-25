"""Unit tests: router + providers with mocks (no real Baby/OpenAI)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_engine.config.settings import clear_settings_cache
from ai_engine.orchestrator.router import RouteDecision, route_request
from ai_engine.providers.rag_stub import call_rag


@pytest.fixture(autouse=True)
def _clear_settings():
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_route_request_orchestrator_off_defaults_baby(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_USE_ORCHESTRATOR", "0")
    monkeypatch.setenv("AI_ENGINE_DEFAULT_ROUTE", "baby")
    clear_settings_cache()
    d = route_request({"messages": [{"role": "user", "content": "What year was Rome founded?"}]})
    assert d == RouteDecision.BABY


def test_route_request_orchestrator_on_factual_rag(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_USE_ORCHESTRATOR", "1")
    clear_settings_cache()
    d = route_request({"messages": [{"role": "user", "content": "Who wrote the Republic?"}]})
    assert d == RouteDecision.RAG


def test_route_request_orchestrator_on_conversational_llm(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_USE_ORCHESTRATOR", "1")
    clear_settings_cache()
    d = route_request({"messages": [{"role": "user", "content": "hi"}]})
    assert d == RouteDecision.LLM


def test_call_rag_stub():
    out = call_rag("test query")
    assert out["chunks"] == []
    assert out["context"] == ""


@patch("ai_engine.providers.baby_client.httpx.Client")
def test_call_baby_http(mock_client_class, monkeypatch):
    monkeypatch.setenv("BABY_BASE_URL", "http://127.0.0.1:8080")
    clear_settings_cache()

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "model": "eurobot-baby",
        }
    )
    mock_instance = MagicMock()
    mock_instance.post.return_value = mock_resp
    mock_client_class.return_value.__enter__.return_value = mock_instance

    from ai_engine.providers.baby_client import call_baby

    out = call_baby([{"role": "user", "content": "ping"}])
    assert out["choices"][0]["message"]["content"] == "ok"
    mock_instance.post.assert_called_once()
