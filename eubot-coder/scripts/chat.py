#!/usr/bin/env python3
"""Interactive chat: merged model or base + LoRA adapter."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

from eubot_coder_utils import load_yaml, repo_root


def load_model(base_name: str, merged_dir: Path | None, adapter_dir: Path | None):
    if merged_dir is not None and merged_dir.is_dir():
        tok_path = str(merged_dir)
        model = AutoModelForCausalLM.from_pretrained(
            tok_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "finetune.yaml")
    ap.add_argument("--merged", type=Path, default=None)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    root = repo_root()
    base = cfg["model_name"]
    merged = args.merged or (root / cfg.get("merged_output_dir", "models/eubot-coder-merged"))
    adapter = args.adapter or (root / cfg["output_dir"])

    if merged.is_dir():
        mdir, adir = merged, None
    elif adapter.is_dir():
        mdir, adir = None, adapter
    else:
        raise SystemExit(
            f"No model at {merged} or {adapter}. Run finetune.py or merge_adapter.py first."
        )

    print(f"Loading (merged={mdir is not None}) …")
    model, tokenizer = load_model(base, mdir, adir)

    system = (cfg.get("system_prompt") or "").strip()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    print("Eubot chat. Commands: /reset , quit\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if user == "/reset":
            messages = [{"role": "system", "content": system}] if system else []
            print("(history cleared)\n")
            continue

        messages.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
        )
        t = Thread(target=model.generate, kwargs=gen_kwargs)
        t.start()
        print("Bot: ", end="", flush=True)
        parts: list[str] = []
        for text in streamer:
            print(text, end="", flush=True)
            parts.append(text)
        print("\n")
        reply = "".join(parts).strip()
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
