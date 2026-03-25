"""
Public entrypoints for backends: thin re-exports + orchestrated dispatch (Phase 1: direct calls).

Use this module when you need a single import path for `call_baby`, `call_llm`, `call_rag`.
"""
from __future__ import annotations

from typing import Any

from ai_engine.config.settings import get_settings
from ai_engine.orchestrator.router import RouteDecision, route_request
from ai_engine.providers.baby_client import call_baby as _call_baby
from ai_engine.providers.llm_stub import call_llm as _call_llm
from ai_engine.providers.rag_real import call_rag as _call_rag

SYSTEM_PERSONALITY = """
You are a friendly, intelligent assistant.
You speak clearly and naturally.
You are helpful and slightly conversational, not robotic.
""".strip()

STYLE_GUIDE = (
    "Keep sentences short and natural.\n"
    "Use a human tone.\n"
    "Do not repeat yourself."
)

RAG_SYSTEM_PROMPT = "You are a smart, helpful and natural assistant."


def _system_with_personality(extra: str | None = None) -> str:
    parts = [SYSTEM_PERSONALITY, STYLE_GUIDE]
    if extra:
        parts.append(extra.strip())
    return "\n\n".join(p for p in parts if p)


def _conversation_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip system messages; keep user/assistant/tool for micro memory."""
    roles = {"user", "assistant", "tool"}
    return [m for m in messages if str(m.get("role", "")).lower() in roles]


def _last_messages(messages: list[dict[str, Any]], n: int = 3) -> list[dict[str, Any]]:
    """Micro memory: last n turns (role/content)."""
    conv = _conversation_messages(messages)
    if not conv:
        return []
    return list(conv[-n:])


def _messages_with_personality(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """System personality + last 3 conversation turns."""
    out: list[dict[str, Any]] = [{"role": "system", "content": _system_with_personality()}]
    out.extend(dict(m) for m in _last_messages(messages, 3))
    return out


def _format_recent_for_rag(messages: list[dict[str, Any]]) -> str:
    """Prior turns only (up to 2 before current), so Question: is not duplicated."""
    conv = _conversation_messages(messages)
    if len(conv) < 2:
        return ""
    prior = conv[-3:-1]
    if not prior:
        return ""
    lines: list[str] = []
    for m in prior:
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _rag_user_prompt(retrieved_context: str, query: str, messages: list[dict[str, Any]]) -> str:
    recent = _format_recent_for_rag(messages)
    head = ""
    if recent:
        head = f"Recent conversation:\n{recent}\n\n"
    return (
        f"{head}"
        "Use this context to answer clearly.\n\n"
        f"Context:\n{retrieved_context}\n\n"
        "Instructions:\n\n"
        "* Be natural\n"
        "* Do not copy the text\n"
        "* Explain clearly\n"
        "* If unsure, say it\n"
        "* Reply with only your final answer — no preamble or bullet lists unless the question asks for them\n\n"
        f"Question:\n{query}"
    )


def _clean_final_answer_text(text: str) -> str:
    """Strip filler; keep a single clean answer string for OpenAI-shaped responses."""
    t = (text or "").strip()
    for prefix in (
        "Final answer:",
        "Final Answer:",
        "Answer:",
        "Risposta:",
        "Risposta finale:",
    ):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix) :].lstrip()
    # collapse excessive blank lines
    lines = [ln.rstrip() for ln in t.splitlines()]
    t = "\n".join(lines).strip()
    return t


def _clean_openai_response(resp: dict[str, Any]) -> dict[str, Any]:
    """Same structure as Baby/LLM; only choices[0].message.content is cleaned."""
    if not isinstance(resp, dict):
        return resp
    out = dict(resp)
    choices = out.get("choices")
    if not isinstance(choices, list) or not choices:
        return out
    first = choices[0]
    if not isinstance(first, dict):
        return out
    msg = first.get("message")
    if not isinstance(msg, dict):
        return out
    content = msg.get("content")
    if isinstance(content, str):
        msg = dict(msg)
        msg["content"] = _clean_final_answer_text(content)
        first = dict(first)
        first["message"] = msg
        out["choices"] = [first] + list(choices[1:])
    return out


def call_baby(messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    return _call_baby(messages, **kw)


def call_llm(messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    return _call_llm(messages, **kw)


def call_rag(query: str, **kw: Any) -> dict[str, Any]:
    return _call_rag(query, **kw)


def run_orchestrated_chat(context: dict[str, Any], **kw: Any) -> dict[str, Any]:
    """
    Route then execute. Phase 1: returns response dict from one backend.
    - RAG: returns stub context + optional Baby (not wired yet); for now returns RAG dict only.
    - HYBRID: reserved for Phase 3.
    """
    decision = route_request(context)
    messages = context.get("messages") or []

    if decision == RouteDecision.BABY:
        return _clean_openai_response(_call_baby(_messages_with_personality(messages), **kw))
    if decision == RouteDecision.LLM:
        return _clean_openai_response(_call_llm(_messages_with_personality(messages), **kw))
    if decision == RouteDecision.RAG:
        last = ""
        if messages and isinstance(messages[-1], dict):
            last = str(messages[-1].get("content") or "")
        rag = _call_rag(last, **kw)
        retrieved_context = str(rag.get("context") or "").strip()
        if not retrieved_context:
            return _clean_openai_response(_call_baby(_messages_with_personality(messages), **kw))
        augmented = [
            {
                "role": "system",
                "content": _system_with_personality(RAG_SYSTEM_PROMPT),
            },
            {"role": "user", "content": _rag_user_prompt(retrieved_context, last, messages)},
        ]
        target = (get_settings().rag_answer_target or "baby").strip().lower()
        if target == "llm":
            return _clean_openai_response(_call_llm(augmented, **kw))
        return _clean_openai_response(_call_baby(augmented, **kw))
    if decision == RouteDecision.HYBRID:
        return {"error": "hybrid not implemented in Phase 1", "decision": decision.value}

    return _clean_openai_response(_call_baby(_messages_with_personality(messages), **kw))
