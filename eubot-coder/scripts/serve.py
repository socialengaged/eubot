#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP server for the fine-tuned model."""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from eubot_coder_utils import load_yaml, repo_root

app = FastAPI(title="Eubot Coder")

_model = None
_tokenizer = None
_system_default = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "eubot-coder"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


def load_model(base_name: str, merged_dir: Path | None, adapter_dir: Path | None):
    if merged_dir is not None and merged_dir.is_dir():
        p = str(merged_dir)
        model = AutoModelForCausalLM.from_pretrained(
            p,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
        model.eval()
        return model, tokenizer
    tok_src = str(adapter_dir) if adapter_dir and adapter_dir.is_dir() else base_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_dir and adapter_dir.is_dir():
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


@app.on_event("startup")
def startup():
    global _model, _tokenizer, _system_default
    cfg = load_yaml(ROOT / "configs" / "finetune.yaml")
    root = repo_root()
    base = cfg["model_name"]
    merged = root / cfg.get("merged_output_dir", "models/eubot-coder-merged")
    adapter = root / cfg["output_dir"]
    _system_default = (cfg.get("system_prompt") or "").strip()
    if merged.is_dir():
        mdir, adir = merged, None
    elif adapter.is_dir():
        mdir, adir = None, adapter
    else:
        raise RuntimeError("No merged model or adapter found. Train or merge first.")
    _model, _tokenizer = load_model(base, mdir, adir)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    assert _model is not None and _tokenizer is not None
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    if _system_default and not any(m["role"] == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": _system_default})

    prompt = _tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=_tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    cid = str(uuid.uuid4())
    return JSONResponse(
        {
            "id": f"chatcmpl-{cid}",
            "object": "chat.completion",
            "model": req.model,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
