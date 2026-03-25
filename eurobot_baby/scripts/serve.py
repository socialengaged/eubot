#!/usr/bin/env python3
"""
OpenAI-compatible HTTP server for Eurobot Baby (causal LM checkpoint).

Run on pod (example):
  python -u scripts/serve.py --checkpoint models/checkpoints/step_431240 --host 0.0.0.0 --port 8080

Environment:
  TOKENIZER_DIR — tokenizer dir if not inside checkpoint (default: <repo>/models/tokenizer).
  BABY_CHECKPOINT — required for `serve_v2_extension.py` (imports `app` without running `main()`).
  DEBUG_GENERATION=1 — same as --debug-generation (lifespan load only).
  SAFE_MODE=1 — same as --safe-mode (temperature=0.5, top_p=0.8).
  EUROBOT_SACRED_RAG_PATH — directory with index.faiss + metadata.pkl from ai_engine.rag.ingest_sacred
    (requires repo root on PYTHONPATH for ai_engine; OPENAI_API_KEY for query embeddings).
  EUROBOT_RAG_EXPANSION_PATH — second index (HF conversational/medical RAG from ingest_multidomain_jsonl).
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import uuid
from pathlib import Path

import torch
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# --- Generation defaults (ML / anti-loop) ---
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.6
TOP_P = 0.9
TOP_K = 50
TEMPERATURE_RETRY = 0.5
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 3

# Stop when decoded generation contains any of these (anti-drift / role bleed)
STOP_SEQUENCES = ("\n\n\n", "User:", "Assistant:")

MAX_RAW_CHARS_BEFORE_TRUNC = 300
HARD_CHAR_LIMIT = 350

SAFE_TEMPERATURE = 0.5
SAFE_TOP_P = 0.8

# Legacy / empty-text only (avoid silent failure)
FALLBACK_EMPTY = "Puoi spiegarmi meglio cosa intendi?"

SYSTEM_PROMPT = (
    "You are Eu girlfriend and secretary, a helpful assistant. Answer clearly, directly, "
    "and simply. Avoid repetition, avoid storytelling, avoid nonsense. Always answer "
    "the user's question in a practical way.\n"
    "No dialogue, no scripts, no characters, no abstract writing."
)

BAD_SUBSTRINGS = (
    "baby:",
    "from shree",
    "abstract:",
    "theorem",
)

_model = None
_tokenizer = None
_debug_gen = False
_safe_mode = False

# Optional sacred RAG (lazy): set when ingest_sacred index loads successfully
_sacred_bundle: tuple | None = None
_sacred_init_attempted = False
# Optional domain expansion RAG (medicine/psych/etc.): ingest_multidomain_jsonl
_expansion_bundle: tuple | None = None
_expansion_init_attempted = False


def load_checkpoint(checkpoint_dir: Path, tokenizer_dir: Path | None) -> tuple:
    tok_path = str(tokenizer_dir) if tokenizer_dir and tokenizer_dir.is_dir() else str(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to(torch.device("cpu"))
    model.eval()
    return model, tokenizer


@asynccontextmanager
async def _lifespan(app_: FastAPI):
    """Load weights when `app` is imported without running `main()` (e.g. serve_v2_extension)."""
    global _model, _tokenizer, _debug_gen, _safe_mode
    if _model is None:
        ckpt = os.environ.get("BABY_CHECKPOINT", "").strip()
        if ckpt:
            tdir = os.environ.get("TOKENIZER_DIR", "").strip()
            tok_path = Path(tdir) if tdir else (ROOT / "models" / "tokenizer")
            print(f"[serve] lifespan: loading from BABY_CHECKPOINT={ckpt}", flush=True)
            _model, _tokenizer = load_checkpoint(Path(ckpt), tok_path if tok_path.is_dir() else None)
            _debug_gen = os.environ.get("DEBUG_GENERATION", "").lower() in ("1", "true", "yes")
            _safe_mode = os.environ.get("SAFE_MODE", "").lower() in ("1", "true", "yes")
            if _safe_mode:
                print("SAFE MODE ENABLED", flush=True)
            if _debug_gen:
                print("DEBUG_GENERATION ENABLED", flush=True)
            print("[serve] lifespan: model ready.", flush=True)
    yield


app = FastAPI(title="Eurobot Baby", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = Field(default="eurobot-baby")
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, description="Capped by server default (150)")
    temperature: float | None = None
    top_p: float | None = None


def _last_user_text(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if (m.role or "user").strip().lower() == "user":
            return (m.content or "").strip()
    return (messages[-1].content or "").strip() if messages else ""


def _get_sacred_rag_context(user_text: str) -> str:
    """Top-3 sacred chunks when EUROBOT_SACRED_RAG_PATH is set and query looks philosophical."""
    global _sacred_bundle, _sacred_init_attempted
    path = os.environ.get("EUROBOT_SACRED_RAG_PATH", "").strip()
    if not path:
        return ""
    if not _sacred_init_attempted:
        _sacred_init_attempted = True
        try:
            monorepo = Path(__file__).resolve().parent.parent.parent
            if (monorepo / "ai_engine").is_dir() and str(monorepo) not in sys.path:
                sys.path.insert(0, str(monorepo))
            from ai_engine.orchestrator.philosophy_trigger import is_philosophical_query
            from ai_engine.rag.sacred_retriever import SacredRAGRetriever, format_sacred_context

            r = SacredRAGRetriever(path)
            if r.load(Path(path)):
                _sacred_bundle = (r, format_sacred_context, is_philosophical_query)
                print(f"[serve] sacred RAG loaded from {path}", flush=True)
        except Exception as e:
            logger.warning("sacred RAG init failed: %s", e)
    if not _sacred_bundle:
        return ""
    r, format_ctx, is_phi = _sacred_bundle
    if not is_phi(user_text):
        return ""
    chunks = r.retrieve(user_text, top_k=3)
    if not chunks:
        return ""
    return format_ctx(chunks)


def _get_expansion_rag_context(user_text: str) -> str:
    """Top-3 chunks from HF/medical expansion index when EUROBOT_RAG_EXPANSION_PATH + domain trigger."""
    global _expansion_bundle, _expansion_init_attempted
    path = os.environ.get("EUROBOT_RAG_EXPANSION_PATH", "").strip()
    if not path:
        return ""
    if not _expansion_init_attempted:
        _expansion_init_attempted = True
        try:
            monorepo = Path(__file__).resolve().parent.parent.parent
            if (monorepo / "ai_engine").is_dir() and str(monorepo) not in sys.path:
                sys.path.insert(0, str(monorepo))
            from ai_engine.orchestrator.reference_domains_trigger import is_domain_reference_query
            from ai_engine.rag.sacred_retriever import SacredRAGRetriever, format_sacred_context

            r = SacredRAGRetriever(path)
            if r.load(Path(path)):
                _expansion_bundle = (r, format_sacred_context, is_domain_reference_query)
                print(f"[serve] expansion RAG loaded from {path}", flush=True)
        except Exception as e:
            logger.warning("expansion RAG init failed: %s", e)
    if not _expansion_bundle:
        return ""
    r, format_ctx, is_domain = _expansion_bundle
    if not is_domain(user_text):
        return ""
    chunks = r.retrieve(user_text, top_k=3)
    if not chunks:
        return ""
    return format_ctx(chunks)


def _combined_rag_context(user_text: str) -> str:
    """Sacred (philosophy) + expansion (medical/psych/...) — non-empty parts joined."""
    parts: list[str] = []
    s = _get_sacred_rag_context(user_text)
    if s:
        parts.append(s)
    e = _get_expansion_rag_context(user_text)
    if e:
        parts.append("--- Additional reference (domain) ---\n" + e)
    return "\n\n".join(parts)


def _messages_to_prompt(messages: list[ChatMessage], *, rag_context: str = "") -> str:
    """System prompt is always our behavioral anchor; client system lines are ignored."""
    sys_p = SYSTEM_PROMPT
    if rag_context:
        sys_p = (
            sys_p
            + "\n\nRelevant reference excerpts (use for ideas only; answer in your own words, "
            "do not copy long passages):\n"
            + rag_context
        )
    lines: list[str] = [f"System: {sys_p}"]
    for m in messages:
        r = (m.role or "user").strip().lower()
        c = (m.content or "").strip()
        if r == "system":
            continue
        if r == "assistant":
            lines.append(f"Assistant: {c}")
        else:
            lines.append(f"User: {c}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _last_20_words_repeat(words: list[str]) -> bool:
    """TASK 6: last 20 words duplicated immediately before (loop)."""
    if len(words) < 40:
        return False
    return words[-20:] == words[-40:-20]


class WordLoopStoppingCriteria(StoppingCriteria):
    """Stop when the last 20 words repeat the previous 20 words (generation loop)."""

    def __init__(self, tokenizer, prompt_token_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        row = input_ids[0]
        gen_only = row[self.prompt_token_len :]
        if gen_only.numel() < 8:
            return False
        text = self.tokenizer.decode(gen_only, skip_special_tokens=True)
        words = text.split()
        return _last_20_words_repeat(words)


class StopSequenceStoppingCriteria(StoppingCriteria):
    """Stop generation when any stop substring appears in decoded new tokens."""

    def __init__(self, tokenizer, prompt_token_len: int, stop_strings: tuple[str, ...]):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_token_len = prompt_token_len
        self.stop_strings = stop_strings

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        row = input_ids[0]
        gen_only = row[self.prompt_token_len :]
        if gen_only.numel() < 2:
            return False
        text = self.tokenizer.decode(gen_only, skip_special_tokens=True)
        return any(s in text for s in self.stop_strings)


def _should_discard(text: str) -> tuple[bool, str]:
    """Hard blocklist only — no diversity reject (softened in scoring)."""
    low = text.lower()
    for bad in BAD_SUBSTRINGS:
        if bad in low:
            return True, f"blocked_pattern:{bad}"
    return False, ""


def _strip_at_first_stop(text: str) -> str:
    cut = len(text)
    for s in STOP_SEQUENCES:
        i = text.find(s)
        if i >= 0:
            cut = min(cut, i)
    return text[:cut].strip()


def _truncate_at_sentence_cap(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    last = max(chunk.rfind("."), chunk.rfind("!"), chunk.rfind("?"))
    if last > max_chars // 4:
        return chunk[: last + 1].strip()
    return chunk.rstrip()


def has_loop(text: str) -> bool:
    """Same word repeated >3 times in a row → loop (for scoring)."""
    words = text.split()
    if len(words) < 4:
        return False
    run = 1
    for i in range(1, len(words)):
        if words[i].lower() == words[i - 1].lower():
            run += 1
            if run > 3:
                return True
        else:
            run = 1
    return False


def score_output(text: str) -> int:
    """
    Soft quality score (0–4+). Penalties never hard-reject; they lower score.
    score >= 3 → PASS
    score == 2 → SHORTEN
    score <= 1 → use contextual fallback
    """
    t = (text or "").strip()
    words = t.split()
    if not words:
        return 0
    unique_ratio = len(set(words)) / len(words)
    score = 0
    if unique_ratio > 0.5:
        score += 1
    if 20 < len(words) < 120:
        score += 1
    if not has_loop(t):
        score += 1
    if t.count(",") < 10:
        score += 1
    # Soft penalties (no direct reject)
    if t.count(",") >= 15:
        score -= 1
    mw = max((len(s.split()) for s in re.split(r"[.!?]+", t) if s.strip()), default=0)
    if mw > 80:
        score -= 1
    return max(0, score)


def clean_output(text: str) -> str:
    """Improve text instead of dropping it: dedup, whitespace, soft length cap."""
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    words = t.split()
    out: list[str] = []
    for w in words:
        if len(out) >= 2 and w.lower() == out[-1].lower() == out[-2].lower():
            continue
        out.append(w)
    t = " ".join(out)
    if len(t) > HARD_CHAR_LIMIT:
        t = t[:HARD_CHAR_LIMIT].rstrip()
    return t.strip()


def shorten_output(text: str) -> str:
    """When score == 2: trim to ~2 sentences or ~70 words."""
    t = (text or "").strip()
    if not t:
        return t
    parts = re.split(r"(?<=[.!?])\s+", t)
    if len(parts) >= 2 and len(t) > 180:
        return " ".join(parts[:2]).strip()
    words = t.split()
    if len(words) > 70:
        return " ".join(words[:70]).strip() + "."
    return t


def _classify_question(q: str) -> str:
    t = (q or "").strip()
    if len(t) < 2:
        return "incomprehensible"
    w = t.split()
    tl = t.lower()
    if len(w) <= 5 and not any(x in tl for x in ("why", "how", "explain", "spiega", "cosa", "what")):
        return "simple"
    if len(w) >= 40 or t.count("?") >= 2:
        return "complex"
    return "simple"


def contextual_fallback(user_question: str) -> str:
    """Human-friendly fallback — not an error tone."""
    cat = _classify_question(user_question)
    if cat == "incomprehensible":
        return "Puoi spiegarmi meglio cosa intendi?"
    if cat == "complex":
        return (
            "Ecco una spiegazione chiara: scrivimi il punto che ti interessa di più "
            "e ti rispondo in modo ordinato."
        )
    return (
        "Ti rispondo in modo semplice: dimmi cosa vuoi ottenere e ti do un passo concreto."
    )


def _post_process_answer(answer: str) -> str:
    words = answer.split()
    if len(words) > 120:
        return " ".join(words[:120]).strip()
    return answer.strip()


def _effective_decode_params(
    temperature: float | None = None,
    top_p: float | None = None,
) -> tuple[float, float, int]:
    if _safe_mode:
        t = SAFE_TEMPERATURE
        p = SAFE_TOP_P
    else:
        t = TEMPERATURE
        p = TOP_P
    if temperature is not None:
        t = temperature
    if top_p is not None:
        p = top_p
    return t, p, TOP_K


def _generate_reply(
    prompt: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    assert _model is not None and _tokenizer is not None
    inputs = _tokenizer(prompt, return_tensors="pt")
    dev = next(_model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    criteria = StoppingCriteriaList(
        [
            WordLoopStoppingCriteria(_tokenizer, prompt_len),
            StopSequenceStoppingCriteria(_tokenizer, prompt_len, STOP_SEQUENCES),
        ]
    )
    t, p, k = _effective_decode_params(temperature, top_p)
    gen_kw: dict = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=t,
        top_p=p,
        top_k=k,
        repetition_penalty=REPETITION_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        pad_token_id=_tokenizer.pad_token_id,
        eos_token_id=_tokenizer.eos_token_id,
        stopping_criteria=criteria,
    )
    if _debug_gen:
        logger.info("generate kwargs: %s", gen_kw)

    with torch.no_grad():
        out = _model.generate(**inputs, **gen_kw)

    new_tokens = out[0][prompt_len:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    text = _strip_at_first_stop(text)
    if len(text) > MAX_RAW_CHARS_BEFORE_TRUNC:
        text = _truncate_at_sentence_cap(text, MAX_RAW_CHARS_BEFORE_TRUNC)
    return text


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "safe_mode": bool(_safe_mode),
        "debug_generation": bool(_debug_gen),
        "sacred_rag_path_set": bool(os.environ.get("EUROBOT_SACRED_RAG_PATH", "").strip()),
        "sacred_rag_loaded": _sacred_bundle is not None,
        "expansion_rag_path_set": bool(os.environ.get("EUROBOT_RAG_EXPANSION_PATH", "").strip()),
        "expansion_rag_loaded": _expansion_bundle is not None,
    }


@app.get("/docs")
def docs_redirect():
    return {"detail": "OpenAPI at /openapi.json"}


def _prepare_candidate(raw: str) -> str:
    raw = _strip_at_first_stop(raw)
    if len(raw) > MAX_RAW_CHARS_BEFORE_TRUNC:
        raw = _truncate_at_sentence_cap(raw, MAX_RAW_CHARS_BEFORE_TRUNC)
    raw = _post_process_answer(raw)
    return clean_output(raw)


def _complete_one_turn(prompt: str, user_question: str) -> str:
    """Dual generation + soft scoring; contextual fallback when quality is low."""
    raw1 = _generate_reply(prompt, temperature=None)
    raw2 = _generate_reply(prompt, temperature=TEMPERATURE_RETRY)

    d1, reason1 = _should_discard(raw1)
    d2, reason2 = _should_discard(raw2)
    if d1 and _debug_gen:
        logger.info("blocked O1: %s", reason1)
    if d2 and _debug_gen:
        logger.info("blocked O2: %s", reason2)

    c1 = _prepare_candidate(raw1) if not d1 else ""
    c2 = _prepare_candidate(raw2) if not d2 else ""

    s1 = score_output(c1) if c1 else -1
    s2 = score_output(c2) if c2 else -1

    if _debug_gen:
        print("[GEN_DEBUG]", flush=True)
        print(f"OUT1: {raw1[:800]!r}", flush=True)
        print(f"SCORE1: {s1}", flush=True)
        print(f"OUT2: {raw2[:800]!r}", flush=True)
        print(f"SCORE2: {s2}", flush=True)
        print(f"CHOSEN: {'O1' if s1 >= s2 else 'O2'}", flush=True)

    if not c1 and not c2:
        print("DISCARDED BAD OUTPUT", flush=True)
        return contextual_fallback(user_question)

    if max(s1, s2) <= 1:
        return contextual_fallback(user_question)

    best = c1 if s1 >= s2 else c2
    sc = max(s1, s2)

    if sc == 2:
        return shorten_output(best)
    return best


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={"error": "model not loaded"},
        )
    msgs = req.messages
    if not msgs:
        return JSONResponse(status_code=400, content={"error": "messages required"})

    user_question = _last_user_text(msgs)
    rag_context = _combined_rag_context(user_question)
    prompt = _messages_to_prompt(msgs, rag_context=rag_context)
    text = _complete_one_turn(prompt, user_question)
    if not (text or "").strip():
        text = FALLBACK_EMPTY

    mid = str(uuid.uuid4())
    return JSONResponse(
        {
            "id": f"chatcmpl-{mid}",
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": req.model or "eurobot-baby",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
    )


def main() -> None:
    global _model, _tokenizer, _debug_gen, _safe_mode
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True, help="HF checkpoint dir (step_*)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Tokenizer dir (default: TOKENIZER_DIR env or models/tokenizer)",
    )
    ap.add_argument(
        "--debug-generation",
        action="store_true",
        help="Log generation kwargs and discard reasons",
    )
    ap.add_argument(
        "--safe-mode",
        action="store_true",
        help="Lower randomness: temperature=0.5, top_p=0.8",
    )
    args = ap.parse_args()

    _debug_gen = args.debug_generation
    _safe_mode = args.safe_mode or os.environ.get("SAFE_MODE", "").lower() in ("1", "true", "yes")
    if _debug_gen:
        logging.basicConfig(level=logging.INFO)

    ckpt = args.checkpoint.resolve()
    if not ckpt.is_dir():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    tdir = args.tokenizer
    if tdir is None:
        env_t = os.environ.get("TOKENIZER_DIR", "").strip()
        tdir = Path(env_t) if env_t else (ROOT / "models" / "tokenizer")
    else:
        tdir = Path(tdir)

    print(f"Loading tokenizer from {tdir if tdir.is_dir() else ckpt}", flush=True)
    print(f"Loading model from {ckpt}", flush=True)
    _model, _tokenizer = load_checkpoint(ckpt, tdir if tdir.is_dir() else None)
    print("Model ready.", flush=True)
    if _safe_mode:
        print("SAFE MODE ENABLED", flush=True)
        print(f"safe-mode: temperature={SAFE_TEMPERATURE} top_p={SAFE_TOP_P}", flush=True)
    if _debug_gen:
        print("DEBUG_GENERATION ENABLED", flush=True)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
