#!/usr/bin/env python3
"""Merge LoRA adapter into base weights and save full model for inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eubot_coder_utils import load_yaml, repo_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "finetune.yaml")
    ap.add_argument("--adapter", type=Path, default=None, help="Adapter dir (default: output_dir from config)")
    ap.add_argument("--out", type=Path, default=None, help="Merged model dir (default: merged_output_dir)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    root = repo_root()
    base = cfg["model_name"]
    adapter = args.adapter or (root / cfg["output_dir"])
    out = args.out or (root / cfg["merged_output_dir"])

    if not adapter.is_dir():
        raise SystemExit(f"Adapter not found: {adapter}")

    print(f"Loading base: {base}")
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading adapter: {adapter}")
    model = PeftModel.from_pretrained(model, str(adapter))
    print("Merging …")
    merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    tok = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=True)
    tok.save_pretrained(str(out))
    print(f"Merged model saved to {out}")


if __name__ == "__main__":
    main()
