#!/usr/bin/env python3
"""Load checkpoint + tokenizer; generate text from a prompt."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from brainzero_utils import device_auto


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True, help="Directory from train.py (e.g. models/checkpoints/step_1000)")
    ap.add_argument("--prompt", type=str, default="The")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    if not args.checkpoint.is_dir():
        raise SystemExit(f"Checkpoint dir not found: {args.checkpoint}")

    device = device_auto()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(args.checkpoint))
    model = GPT2LMHeadModel.from_pretrained(str(args.checkpoint)).to(device)
    model.eval()

    ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
