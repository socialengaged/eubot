#!/usr/bin/env python3
"""QLoRA fine-tune Qwen2.5-Coder with TRL SFTTrainer (conversational JSONL)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from eubot_coder_utils import load_yaml, repo_root


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "finetune.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint in output_dir (if any)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    root = repo_root()
    model_name = cfg["model_name"]
    train_path = root / cfg["data_train"]
    val_path = root / cfg["data_val"]
    out_dir = root / cfg["output_dir"]

    if not train_path.is_file():
        raise SystemExit(f"Missing {train_path} — run scripts/prepare_data.py first.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    target_modules = list(cfg.get("target_modules") or [])
    peft_config = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    val_ds = load_dataset("json", data_files=str(val_path), split="train") if val_path.is_file() else None

    max_seq = int(cfg.get("max_seq_length", 2048))
    max_steps = cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps not in (None, "null", "") else -1

    has_eval = val_ds is not None
    sft_kw: dict = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 500)),
        bf16=bool(cfg.get("bf16", True)) and torch.cuda.is_available(),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        optim=str(cfg.get("optim", "paged_adamw_8bit")),
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="none",
        max_length=max_seq,
        packing=False,
    )
    if has_eval:
        sft_kw["eval_strategy"] = cfg.get("eval_strategy", "steps")
        sft_kw["eval_steps"] = int(cfg.get("eval_steps", 500))
    else:
        sft_kw["eval_strategy"] = "no"

    try:
        training_args = SFTConfig(**sft_kw)
    except TypeError:
        sft_kw.pop("max_length", None)
        sft_kw["max_seq_length"] = max_seq
        training_args = SFTConfig(**sft_kw)

    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
        )

    resume_ckpt = None
    if args.resume:
        ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            resume_ckpt = str(ckpts[-1])
            print(f"Resuming from {resume_ckpt}")
        else:
            print("No checkpoint found, starting from scratch.")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
