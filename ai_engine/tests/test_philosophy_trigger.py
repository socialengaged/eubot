"""Tests for philosophical query heuristic."""
from __future__ import annotations

from ai_engine.orchestrator.philosophy_trigger import is_philosophical_query


def test_short_message_not_philosophical():
    assert is_philosophical_query("hi") is False
    assert is_philosophical_query("thanks") is False


def test_philosophy_keywords():
    assert is_philosophical_query("What is stoicism in plain terms?") is True
    assert is_philosophical_query("Explain ethics and virtue for beginners.") is True
    assert is_philosophical_query("Cos'è la filosofia per te?") is True


def test_non_philosophy_long():
    assert is_philosophical_query("How do I reset my password on the website?") is False
