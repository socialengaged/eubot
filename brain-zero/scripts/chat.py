#!/usr/bin/env python3
"""Interactive chat with a brain-zero model. Keeps conversation history as context."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from brainzero_utils import device_auto


SEPARATOR = "\n"
MAX_HISTORY_TOKENS = 384  # tokens reserved for conversation history


def find_latest_checkpoint(base: Path) -> Path | None:
    ckpts = sorted(base.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return ckpts[-1] if ckpts else None


def generate(model, tokenizer, prompt_ids, device, max_new=60, temp=0.85, top_k=40, top_p=0.92):
    with torch.no_grad():
        out = model.generate(
            prompt_ids,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )
    new_tokens = out[0][prompt_ids.shape[1]:]
    return tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True).strip()


def build_prompt(history: list[tuple[str, str]], user_msg: str) -> str:
    """Build a flat text prompt with conversation turns."""
    parts = []
    for user, bot in history:
        parts.append(f"User: {user}")
        parts.append(f"Bot: {bot}")
    parts.append(f"User: {user_msg}")
    parts.append("Bot:")
    return SEPARATOR.join(parts)


def trim_history(history: list[tuple[str, str]], tokenizer, max_tokens: int):
    """Keep only as many recent turns as fit in max_tokens."""
    while history:
        test = build_prompt(history, "test")
        ids = tokenizer.encode(test)
        if len(ids) <= max_tokens:
            break
        history.pop(0)
    return history


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with brain-zero model")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Checkpoint dir (default: latest in models/checkpoints/)")
    ap.add_argument("--max_new", type=int, default=60, help="Max new tokens per reply")
    ap.add_argument("--temperature", type=float, default=0.85)
    args = ap.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = find_latest_checkpoint(ROOT / "models" / "checkpoints")
        if ckpt is None:
            print("No checkpoints found in models/checkpoints/. Train first.")
            return
    if not ckpt.is_dir():
        print(f"Checkpoint not found: {ckpt}")
        return

    device = device_auto()
    print(f"Loading model from {ckpt.name} on {device}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(ckpt))
    model = GPT2LMHeadModel.from_pretrained(str(ckpt)).to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded ({param_count:.0f}M params). Type 'quit' to exit.\n")

    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        history = trim_history(history, tokenizer, MAX_HISTORY_TOKENS)
        prompt_text = build_prompt(history, user_input)
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        reply = generate(model, tokenizer, prompt_ids, device,
                         max_new=args.max_new, temp=args.temperature)

        first_line = reply.split("User:")[0].strip()
        if not first_line:
            first_line = reply[:200].strip()

        print(f"Bot: {first_line}\n")
        history.append((user_input, first_line))


if __name__ == "__main__":
    main()
