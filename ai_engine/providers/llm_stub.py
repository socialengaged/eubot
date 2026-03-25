"""External LLM provider — Phase 1 stub; optional OpenAI call when key + stub mode off."""
from __future__ import annotations

import json
import os
from typing import Any

import httpx

from ai_engine.config.settings import get_settings


def call_llm(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """
    OpenAI-compatible chat completions. If AI_ENGINE_LLM_STUB_MODE=1 (default), returns a fixed stub.
    If stub mode off and OPENAI_API_KEY set, calls OpenAI-compatible API.
    """
    settings = get_settings()
    if settings.llm_stub_mode:
        return {
            "id": "ai_engine-llm-stub",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "[ai_engine] LLM stub: set AI_ENGINE_LLM_STUB_MODE=0 and OPENAI_API_KEY to use a real provider.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "model": "stub",
        }

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when AI_ENGINE_LLM_STUB_MODE=0")

    url = settings.openai_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return json.loads(r.text)
