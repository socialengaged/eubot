#!/usr/bin/env python3
"""
Download open instruction-following coding datasets from HuggingFace and save
unified chat JSONL: one object per line {"messages": [...]}.

Datasets are normalized to Qwen-style chat (system / user / assistant).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from datasets import Dataset, concatenate_datasets, load_dataset

from eubot_coder_utils import load_yaml, repo_root


DEFAULT_SYSTEM = (
    "Sei Eubot, assistente di programmazione. Stile: diretto, pratico, strategico, "
    "mai verboso. Rispondi con codice quando serve, spiegazioni brevi quando basta."
)


def alpaca_to_messages(row: dict, system: str) -> dict | None:
    inst = (row.get("instruction") or row.get("prompt") or "").strip()
    inp = (row.get("input") or "").strip()
    out = (row.get("output") or row.get("response") or "").strip()
    if not inst or not out:
        return None
    user = inst if not inp else f"{inst}\n\n{inp}"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": out},
        ]
    }


def codefeedback_to_messages(row: dict, system: str) -> dict | None:
    # m-a-p/CodeFeedback-Filtered-Instruction: query + answer
    q = (row.get("query") or row.get("instruction") or row.get("prompt") or "").strip()
    a = (row.get("answer") or row.get("response") or row.get("output") or "").strip()
    if not q or not a:
        return None
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    }


def load_python_alpaca(system: str, max_rows: int | None) -> Dataset:
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))

    def _map(ex):
        return alpaca_to_messages(ex, system)

    rows = [_map(ex) for ex in ds]
    rows = [r for r in rows if r is not None]
    return Dataset.from_list(rows)


def load_code_instructions_alpaca(system: str, max_rows: int | None) -> Dataset:
    ds = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train")
    n = min(max_rows or len(ds), len(ds))
    ds = ds.shuffle(seed=42).select(range(n))

    def _map(ex):
        return alpaca_to_messages(ex, system)

    rows = [_map(ex) for ex in ds]
    rows = [r for r in rows if r is not None]
    return Dataset.from_list(rows)


def load_codefeedback(system: str, max_rows: int | None) -> Dataset:
    ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    n = min(max_rows or len(ds), len(ds))
    if n < len(ds):
        ds = ds.shuffle(seed=42).select(range(n))

    def _map(ex):
        return codefeedback_to_messages(ex, system)

    rows = [_map(ex) for ex in ds]
    rows = [r for r in rows if r is not None]
    return Dataset.from_list(rows)


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "finetune.yaml")
    ap.add_argument("--max_python_alpaca", type=int, default=None, help="Cap rows (default: all ~18k)")
    ap.add_argument("--max_code_instructions", type=int, default=30_000, help="Subset of 122k dataset")
    ap.add_argument("--max_codefeedback", type=int, default=10_000)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge_style", type=Path, default=None, help="Optional JSONL from generate_style.py")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    system = (cfg.get("system_prompt") or DEFAULT_SYSTEM).strip()

    random.seed(args.seed)

    parts = []
    print("Loading iamtarun/python_code_instructions_18k_alpaca …")
    try:
        parts.append(load_python_alpaca(system, args.max_python_alpaca))
        print(f"  -> {len(parts[-1])} rows")
    except Exception as e:
        print(f"  SKIP python_code_instructions_18k_alpaca: {e}")

    print("Loading TokenBender/code_instructions_122k_alpaca_style …")
    try:
        parts.append(load_code_instructions_alpaca(system, args.max_code_instructions))
        print(f"  -> {len(parts[-1])} rows")
    except Exception as e:
        print(f"  SKIP code_instructions_122k: {e}")

    print("Loading m-a-p/CodeFeedback-Filtered-Instruction …")
    try:
        parts.append(load_codefeedback(system, args.max_codefeedback))
        print(f"  -> {len(parts[-1])} rows")
    except Exception as e:
        print(f"  SKIP CodeFeedback: {e}")

    parts = [p for p in parts if len(p) > 0]
    if not parts:
        raise SystemExit("No datasets loaded. Check network / HF access.")

    full = concatenate_datasets(parts)
    full = full.shuffle(seed=args.seed)

    if args.merge_style and args.merge_style.is_file():
        print(f"Merging style file {args.merge_style} …")
        extra = []
        with open(args.merge_style, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                extra.append(json.loads(line))
        if extra:
            full = concatenate_datasets([full, Dataset.from_list(extra)])
            full = full.shuffle(seed=args.seed)

    n = len(full)
    n_val = max(1, int(n * args.val_ratio))
    indices = list(range(n))
    random.shuffle(indices)
    val_idx = set(indices[:n_val])
    train_rows = [full[i] for i in range(n) if i not in val_idx]
    val_rows = [full[i] for i in range(n) if i in val_idx]

    out_train = repo_root() / "data" / "processed" / "train.jsonl"
    out_val = repo_root() / "data" / "processed" / "val.jsonl"
    save_jsonl(out_train, train_rows)
    save_jsonl(out_val, val_rows)
    print(f"Wrote {len(train_rows)} train, {len(val_rows)} val -> {out_train.parent}")


if __name__ == "__main__":
    main()
