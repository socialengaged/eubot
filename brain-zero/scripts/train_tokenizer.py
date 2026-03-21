#!/usr/bin/env python3
"""Train a BPE tokenizer on processed JSONL and save to models/tokenizer/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


def iter_texts(jsonl_path: Path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=Path, default=ROOT / "data" / "processed" / "train.jsonl")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "models" / "tokenizer")
    args = ap.parse_args()

    if not args.train_jsonl.is_file():
        raise SystemExit(f"Missing {args.train_jsonl} — run build_dataset.py first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    bpe = models.BPE(unk_token="<unk>")
    tokenizer = Tokenizer(bpe)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<unk>", "<eos>", "<bos>"],
    )
    print("Training BPE …")
    tokenizer.train_from_iterator(iter_texts(args.train_jsonl), trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
    )
    fast.save_pretrained(str(args.out_dir))
    print(f"Saved tokenizer to {args.out_dir} (vocab_size={fast.vocab_size})")


if __name__ == "__main__":
    main()
