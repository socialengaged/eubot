#!/usr/bin/env python3
"""
Train GPT-2-style LM from random init. Reads configs/model.yaml + configs/training.yaml.
Tokenizer must exist at models/tokenizer (run train_tokenizer.py first).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, get_linear_schedule_with_warmup

from baby_utils import (
    gpt2_config_from_dict,
    load_yaml,
    model_config_dict,
    repo_root,
    set_seed,
)


def _dataloader_kwargs(device: torch.device, num_workers: int) -> dict[str, object]:
    """CPU-side loading: pin_memory + workers evita che il main thread sia il collo di bottiglia."""
    nw = max(0, int(num_workers))
    if nw > 0:
        return {
            "num_workers": nw,
            "pin_memory": device.type == "cuda",
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
    return {"num_workers": 0, "pin_memory": device.type == "cuda"}


def _move_batch_to_device(batch, device: torch.device, non_blocking: bool):
    """Sposta batch su device (tensore o dict da futuri collate)."""
    if isinstance(batch, dict):
        return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
    return batch.to(device, non_blocking=non_blocking)


class BlockDataset(Dataset):
    """Non-overlapping blocks of token ids from JSONL {text: ...}."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: PreTrainedTokenizerFast,
        block_size: int,
        max_blocks: int | None = None,
    ) -> None:
        self.block_size = block_size
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.unk_token_id or 0

        buffer: list[int] = []
        self.blocks: list[torch.Tensor] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line)["text"]
                ids = tokenizer.encode(text, add_special_tokens=False)
                if eos_id is not None:
                    ids.append(eos_id)
                buffer.extend(ids)
                while len(buffer) >= block_size:
                    chunk = buffer[:block_size]
                    buffer = buffer[block_size:]
                    self.blocks.append(torch.tensor(chunk, dtype=torch.long))
                    if max_blocks is not None and len(self.blocks) >= max_blocks:
                        break
                if max_blocks is not None and len(self.blocks) >= max_blocks:
                    break

        if buffer and len(self.blocks) < (max_blocks or 10**18):
            while len(buffer) < block_size:
                buffer.append(pad_id)
            self.blocks.append(torch.tensor(buffer[:block_size], dtype=torch.long))

        if not self.blocks:
            raise RuntimeError(f"No blocks built from {jsonl_path} — check text length vs block_size.")

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.blocks[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=str, default=None, help="Override model.yaml profile (small)")
    ap.add_argument("--max_train_blocks", type=int, default=None, help="Cap training blocks for quick tests")
    ap.add_argument("--training_config", type=str, default="training.yaml", help="Training config filename inside configs/")
    ap.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint dir (e.g. models/checkpoints/step_3000)")
    args = ap.parse_args()

    root = repo_root()
    train_cfg = load_yaml(root / "configs" / args.training_config)
    mdict = model_config_dict(args.profile)
    tok_dir = root / train_cfg["tokenizer_dir"]
    if not (tok_dir / "tokenizer.json").is_file() and not (tok_dir / "tokenizer_config.json").is_file():
        raise SystemExit(f"Tokenizer not found in {tok_dir} — run train_tokenizer.py first.")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
    vocab_size = tokenizer.vocab_size
    mdict["vocab_size"] = vocab_size

    max_seq = int(train_cfg["max_seq_len"])
    mdict["n_positions"] = max(mdict.get("n_positions", max_seq), max_seq)

    cfg = gpt2_config_from_dict(mdict)
    set_seed(int(train_cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] device={device}")

    resume_step = 0
    if args.resume and args.resume.is_dir():
        print(f"Resuming from {args.resume} ...")
        model = GPT2LMHeadModel.from_pretrained(str(args.resume))
        name = args.resume.name
        if name.startswith("step_"):
            resume_step = int(name.split("_")[1])
        print(f"  resumed at step {resume_step}")
    else:
        model = GPT2LMHeadModel(cfg)

    model = model.to(device)
    if device.type == "cuda":
        assert next(model.parameters()).is_cuda, "MODEL NOT ON GPU"

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {param_count:.0f}M params")

    train_path = root / train_cfg["data_train"]
    val_path = root / train_cfg["data_val"]
    if not train_path.is_file():
        raise SystemExit(f"Missing {train_path}")

    print(f"Building train blocks from {train_path} (CPU, può richiedere molti minuti su JSONL grandi) ...", flush=True)
    train_ds = BlockDataset(
        train_path,
        tokenizer,
        block_size=max_seq,
        max_blocks=args.max_train_blocks,
    )
    try:
        print(f"Building val blocks ...", flush=True)
        val_ds = (
            BlockDataset(val_path, tokenizer, block_size=max_seq, max_blocks=500)
            if val_path.is_file() and val_path.stat().st_size > 0
            else None
        )
    except RuntimeError:
        val_ds = None
    if val_ds is None:
        n = min(256, len(train_ds))
        val_ds = torch.utils.data.Subset(train_ds, list(range(n)))

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 8))
    dl_kw = _dataloader_kwargs(device, num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw)
    _pin = bool(dl_kw.get("pin_memory", False))
    _nb = device.type == "cuda" and _pin
    print(
        f"DataLoader: num_workers={dl_kw.get('num_workers', 0)} "
        f"pin_memory={_pin} prefetch={dl_kw.get('prefetch_factor', '—')} "
        f"h2d_non_blocking={_nb}"
    )

    accum = int(train_cfg["gradient_accumulation_steps"])
    lr = float(train_cfg["learning_rate"])
    wd = float(train_cfg["weight_decay"])
    max_steps = int(train_cfg["max_steps"])
    warmup = int(train_cfg["warmup_steps"])

    remaining_steps = max_steps - resume_step
    if remaining_steps <= 0:
        print(f"Already at step {resume_step} >= max_steps {max_steps}. Nothing to do.")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = get_linear_schedule_with_warmup(opt, warmup, max_steps)
    for _ in range(resume_step):
        scheduler.step()

    use_fp16 = bool(train_cfg.get("use_fp16", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    ckpt_dir = root / train_cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_every = int(train_cfg["save_every"])
    eval_every = int(train_cfg["eval_every"])
    grad_clip = float(train_cfg.get("gradient_clip", 1.0))

    global_step = resume_step
    model.train()
    running_loss = 0.0
    t0 = time.time()
    pbar = tqdm(total=max_steps, initial=resume_step, desc="train")

    train_iter = iter(train_loader)
    while global_step < max_steps:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = _move_batch_to_device(batch, device, _nb)
            with torch.amp.autocast("cuda", enabled=use_fp16):
                if isinstance(batch, dict):
                    out = model(**batch)
                else:
                    out = model(input_ids=batch, labels=batch)
                loss = out.loss / accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        global_step += 1
        running_loss += accum_loss
        pbar.update(1)
        pbar.set_postfix(loss=f"{accum_loss:.4f}")

        if global_step % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"step {global_step}  loss={accum_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s elapsed"
            )

        if global_step % eval_every == 0:
            model.eval()
            val_loss = 0.0
            n = 0
            eval_batches = int(train_cfg.get("eval_batches", 20))
            with torch.no_grad():
                for i, vb in enumerate(val_loader):
                    if i >= eval_batches:
                        break
                    vb = _move_batch_to_device(vb, device, _nb)
                    with torch.amp.autocast("cuda", enabled=use_fp16):
                        if isinstance(vb, dict):
                            out = model(**vb)
                        else:
                            out = model(input_ids=vb, labels=vb)
                    val_loss += out.loss.item()
                    n += 1
            model.train()
            if n:
                print(f"  >> val_loss={val_loss / n:.4f}")

        if global_step % save_every == 0 or global_step == max_steps:
            save_path = ckpt_dir / f"step_{global_step}"
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  >> saved checkpoint -> {save_path}")

    pbar.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
