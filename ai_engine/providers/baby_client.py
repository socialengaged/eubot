"""HTTP client for existing Eurobot Baby OpenAI-compatible API."""
from __future__ import annotations

import json
from typing import Any

import httpx

from ai_engine.config.settings import get_settings


def call_baby(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 256,
    temperature: float = 0.7,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """
    POST /v1/chat/completions to Baby serve.py. Returns parsed JSON response.
    """
    settings = get_settings()
    url = settings.baby_base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": settings.baby_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return json.loads(r.text)
