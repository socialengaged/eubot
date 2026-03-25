#!/usr/bin/env python3
"""
Extend Eurobot Baby FastAPI app with POST /v2/chat (orchestrator + Baby fallback).

Import the existing `app` from scripts/serve.py — do not instantiate a second FastAPI.

Run on the pod (separate from legacy serve on :8080), default :8081:
  export PYTHONPATH=/workspace/eubot   # parent repo containing ai_engine/
  export PORT=8081
  python scripts/serve_v2_extension.py

VRAM: importing scripts.serve loads the model. Do not run serve.py :8080 and this process
together on one GPU unless you accept 2× memory.

Environment:
  EUBOT_ROOT — directory that contains ai_engine/ (default: parent of eurobot_baby/)
  PORT — listen port (default 8081)
  BABY_BASE_URL — for orchestrator Baby HTTP calls (default http://127.0.0.1:8080)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_EUBOT_ROOT = Path(os.environ.get("EUBOT_ROOT", str(_ROOT.parent))).resolve()
if (_EUBOT_ROOT / "ai_engine").is_dir() and str(_EUBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EUBOT_ROOT))

try:
    from ai_engine.orchestrator.providers import call_baby, run_orchestrated_chat
except ImportError as e:
    raise ImportError(
        "ai_engine is not importable. Set PYTHONPATH or EUBOT_ROOT to the monorepo root "
        "that contains the ai_engine/ package."
    ) from e

try:
    from scripts.serve import app
except ImportError as e:
    raise ImportError(
        "scripts.serve not found. Run from eurobot_baby repo root; scripts/serve.py "
        "must exist (on the pod / GitHub clone)."
    ) from e

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class V2ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(..., min_length=1)


def _messages_as_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _normalize_to_assistant_text(resp: dict[str, Any]) -> str:
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if content is not None:
            return str(content)
    if "error" in resp:
        return str(resp.get("error"))
    if "rag" in resp:
        try:
            return json.dumps(resp.get("rag"), ensure_ascii=False)
        except (TypeError, ValueError):
            return str(resp.get("rag"))
    try:
        return json.dumps(resp, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(resp)


def _openai_chat_completion(content: str, model_name: str | None) -> dict[str, Any]:
    mid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    return {
        "id": mid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name or "eurobot-baby-v2",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/v2/chat")
def v2_chat(body: V2ChatRequest) -> dict[str, Any]:
    msgs = body.messages
    if not msgs:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    payload = {"messages": _messages_as_dicts(msgs)}

    try:
        raw = run_orchestrated_chat(payload)
        text = _normalize_to_assistant_text(raw)
    except Exception as ex:
        logger.exception("orchestrator failed, falling back to Baby: %s", ex)
        try:
            raw = call_baby(payload["messages"])
            text = _normalize_to_assistant_text(raw)
        except Exception as ex2:
            logger.exception("Baby fallback failed: %s", ex2)
            raise HTTPException(status_code=502, detail="orchestrator and Baby fallback failed") from ex2

    return _openai_chat_completion(text, body.model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8081"))
    uvicorn.run(app, host=host, port=port)
