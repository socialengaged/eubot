#!/usr/bin/env python3
"""
Train GPT-2-style LM from random init. Reads configs/model.yaml + configs/training.yaml.
Tokenizer must exist at models/tokenizer (run train_tokenizer.py first).

Train data: streaming JSONL (no preload in RAM) via IterableDataset.
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
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, get_linear_schedule_with_warmup

from baby_utils import (
    gpt2_config_from_dict,
    load_yaml,
    model_config_dict,
    repo_root,
    set_seed,
)


def _dataloader_kwargs_streaming(device: torch.device) -> dict[str, object]:
    """IterableDataset: num_workers=0 (semplice e compatibile); pin_memory per H2D async."""
    return {"num_workers": 0, "pin_memory": device.type == "cuda"}


def _move_batch_to_device(batch, device: torch.device, non_blocking: bool):
    """Sposta batch su device (tensore o dict da collate)."""
    if isinstance(batch, dict):
        return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
    return batch.to(device, non_blocking=non_blocking)


def _count_tokens(batch: dict | torch.Tensor) -> int:
    """Token reali (non pad) nel batch."""
    if isinstance(batch, dict) and "attention_mask" in batch:
        return int(batch["attention_mask"].sum().item())
    if isinstance(batch, dict) and "input_ids" in batch:
        return int(batch["input_ids"].numel())
    return int(batch.numel())


def _make_collate_fn(pad_token_id: int):
    """Padding variabile nel batch; labels=-100 sul pad (loss HF)."""

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        ids = [b["input_ids"] for b in batch]
        max_len = max(x.size(0) for x in ids)
        rows: list[torch.Tensor] = []
        labs: list[torch.Tensor] = []
        for x in ids:
            pad_len = max_len - x.size(0)
            if pad_len > 0:
                pad = torch.full((pad_len,), pad_token_id, dtype=torch.long)
                inp_row = torch.cat([x, pad])
                lab_row = torch.cat([x.clone(), torch.full((pad_len,), -100, dtype=torch.long)])
            else:
                inp_row = x
                lab_row = x.clone()
            rows.append(inp_row)
            labs.append(lab_row)
        input_ids = torch.stack(rows, dim=0)
        labels = torch.stack(labs, dim=0)
        attention_mask = (input_ids != pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


class StreamingJsonlIterableDataset(IterableDataset):
    """Una passata sul file JSONL: tokenizza per riga, niente preload RAM."""

    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
        max_lines: int | None = None,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_lines = max_lines

    def __iter__(self):
        n = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text") or data.get("content") or ""
                    if not text:
                        continue
                    ids = self.tokenizer.encode(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.max_length,
                    )
                    if len(ids) < 2:
                        continue
                    t = torch.tensor(ids, dtype=torch.long)
                    yield {"input_ids": t, "labels": t.clone()}
                    n += 1
                    if self.max_lines is not None and n >= self.max_lines:
                        break
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=str, default=None, help="Override model.yaml profile (small)")
    ap.add_argument("--max_train_blocks", type=int, default=None, help="Deprecated (streaming); optional cap lines/epoch via max_lines")
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

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    cfg = gpt2_config_from_dict(mdict)
    set_seed(int(train_cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] device={device}")

    resume_step = 0
    resume_dir: Path | None = args.resume if args.resume and args.resume.is_dir() else None
    if resume_dir is not None:
        print(f"Resuming from {resume_dir} ...")
        model = GPT2LMHeadModel.from_pretrained(str(resume_dir))
        name = resume_dir.name
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

    train_max_lines = args.max_train_blocks  # riuso: limite righe per test rapidi
    print(f"[TRAIN] streaming JSONL (no RAM preload): {train_path}", flush=True)
    train_ds = StreamingJsonlIterableDataset(
        train_path,
        tokenizer,
        max_length=max_seq,
        max_lines=train_max_lines,
    )

    if val_path.is_file() and val_path.stat().st_size > 0:
        print(f"[TRAIN] val stream (max 500 lines): {val_path}", flush=True)
        val_ds = StreamingJsonlIterableDataset(val_path, tokenizer, max_length=max_seq, max_lines=500)
    else:
        print("[TRAIN] val: first 256 lines of train JSONL", flush=True)
        val_ds = StreamingJsonlIterableDataset(train_path, tokenizer, max_length=max_seq, max_lines=256)

    batch_size = int(train_cfg["batch_size"])
    dl_kw = _dataloader_kwargs_streaming(device)
    collate = _make_collate_fn(pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        **dl_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        **dl_kw,
    )
    _pin = bool(dl_kw.get("pin_memory", False))
    _nb = device.type == "cuda" and _pin
    print(
        f"DataLoader: IterableDataset stream num_workers=0 "
        f"pin_memory={_pin} h2d_non_blocking={_nb} batch_size={batch_size}",
        flush=True,
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
    for g in opt.param_groups:
        g.setdefault("initial_lr", g["lr"])

    opt_ckpt = resume_dir / "optimizer.pt" if resume_dir else None
    if opt_ckpt is not None and opt_ckpt.is_file():
        opt.load_state_dict(torch.load(opt_ckpt, map_location=device))
        print("[TRAIN] loaded optimizer.pt from checkpoint", flush=True)

    sched_ckpt = resume_dir / "scheduler.pt" if resume_dir else None
    if sched_ckpt is not None and sched_ckpt.is_file():
        scheduler = get_linear_schedule_with_warmup(opt, warmup, max_steps)
        scheduler.load_state_dict(torch.load(sched_ckpt, map_location="cpu"))
        print("[TRAIN] loaded scheduler.pt from checkpoint", flush=True)
    else:
        last_epoch = resume_step - 1 if resume_step > 0 else -1
        scheduler = get_linear_schedule_with_warmup(opt, warmup, max_steps, last_epoch=last_epoch)
        print(f"[TRAIN] scheduler aligned without warmup stepping loop (last_epoch={last_epoch})", flush=True)

    global_step = resume_step
    print(f"[TRAIN] resume global_step={global_step}", flush=True)

    use_fp16 = bool(train_cfg.get("use_fp16", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    ckpt_dir = root / train_cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_every = int(train_cfg["save_every"])
    eval_every = int(train_cfg["eval_every"])
    grad_clip = float(train_cfg.get("gradient_clip", 1.0))
    perf_every = int(train_cfg.get("perf_log_every", 20))
    ema_alpha = float(train_cfg.get("perf_ema_alpha", 0.1))

    model.train()
    t0 = time.time()
    pbar = tqdm(total=max_steps, initial=resume_step, desc="train")

    train_iter = iter(train_loader)
    ema_step_time: float | None = None
    ema_tok_s: float | None = None

    while global_step < max_steps:
        t_step = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        tokens_this_step = 0
        skip_optimizer = False
        did_backward = False

        for _ in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = _move_batch_to_device(batch, device, _nb)
            tokens_this_step += _count_tokens(batch)
            with torch.amp.autocast("cuda", enabled=use_fp16):
                if isinstance(batch, dict):
                    out = model(**batch)
                else:
                    out = model(input_ids=batch, labels=batch)
                loss = out.loss / accum

            if not torch.isfinite(loss).all():
                lv = loss.detach().float().item() if loss.numel() == 1 else float("nan")
                print(
                    f"[TRAIN][WARN] non-finite loss at step {global_step} (micro): {lv}",
                    flush=True,
                )
                skip_optimizer = True
                break

            scaler.scale(loss).backward()
            did_backward = True
            accum_loss += loss.item()

        if skip_optimizer:
            opt.zero_grad(set_to_none=True)
            if did_backward:
                scaler.update()
            continue

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        global_step += 1
        step_wall = time.perf_counter() - t_step
        if ema_step_time is None:
            ema_step_time = step_wall
            ema_tok_s = tokens_this_step / max(step_wall, 1e-9)
        else:
            ema_step_time = ema_alpha * step_wall + (1.0 - ema_alpha) * ema_step_time
            inst_tok_s = tokens_this_step / max(step_wall, 1e-9)
            ema_tok_s = ema_alpha * inst_tok_s + (1.0 - ema_alpha) * ema_tok_s

        pbar.update(1)
        pbar.set_postfix(loss=f"{accum_loss:.4f}")

        if global_step % perf_every == 0 and ema_step_time is not None and ema_tok_s is not None:
            print(
                f"[TRAIN][PERF] step={global_step} step_time={ema_step_time:.3f}s "
                f"tok/s={ema_tok_s:.0f} eff_batch={batch_size * accum} "
                f"tokens_last_step={tokens_this_step}",
                flush=True,
            )

        if global_step % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"step {global_step}  loss={accum_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s elapsed",
                flush=True,
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
                print(f"  >> val_loss={val_loss / n:.4f}", flush=True)

        if global_step % save_every == 0 or global_step == max_steps:
            save_path = ckpt_dir / f"step_{global_step}"
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            torch.save(opt.state_dict(), save_path / "optimizer.pt")
            torch.save(scheduler.state_dict(), save_path / "scheduler.pt")
            print(f"  >> saved checkpoint -> {save_path} (+ optimizer.pt, scheduler.pt)", flush=True)

    pbar.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
