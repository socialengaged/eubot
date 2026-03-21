#!/usr/bin/env python3
"""
End-to-end micro test: train a tiny BPE tokenizer, random-init GPT-2-style model,
run ~50 optimization steps, generate text. Proves CUDA/PyTorch/transformers pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from brainzero_utils import device_auto, set_seed


CORPUS = """
The quick brown fox jumps over the lazy dog.
Language models learn patterns from text.
We train on many examples to predict the next token.
This is a minimal sanity check for the brain-zero pipeline.
""".strip().split("\n")


def train_micro_tokenizer(vocab_size: int = 512) -> PreTrainedTokenizerFast:
    bpe = models.BPE(unk_token="<unk>")
    tokenizer = Tokenizer(bpe)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<eos>"],
    )
    tokenizer.train_from_iterator(CORPUS, trainer)
    # EOS at end of each line for short sequences
    tok_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    return tok_fast


def main() -> None:
    set_seed(42)
    device = device_auto()
    print(f"[test_baby] device={device}")

    tok = train_micro_tokenizer(512)
    print(f"[test_baby] tokenizer vocab_size={tok.vocab_size}")

    # ~1M param model
    cfg = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=128,
        n_embd=128,
        n_layer=2,
        n_head=4,
        n_inner=512,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[test_baby] model parameters: {n_params:,}")

    # Build one long token sequence from corpus
    ids: list[int] = []
    for line in CORPUS:
        ids.extend(tok.encode(line + " "))
    if len(ids) < 32:
        ids = ids * 8
    input_ids = torch.tensor([ids[:256]], dtype=torch.long, device=device)
    labels = input_ids.clone()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(50):
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 10 == 0:
            print(f"  step {step:3d}  loss={loss.item():.4f}")

    model.eval()
    prompt = "The quick"
    prompt_ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            prompt_ids,
            max_new_tokens=40,
            do_sample=True,
            top_k=50,
            temperature=0.9,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(gen[0].tolist(), skip_special_tokens=True)
    print("[test_baby] sample generation:")
    print(text)
    print("[test_baby] OK — pipeline alive.")


if __name__ == "__main__":
    main()
