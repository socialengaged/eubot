"""Shared helpers for brain-zero scripts."""
from __future__ import annotations

from pathlib import Path

import yaml
from transformers import GPT2Config


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_config_dict(profile: str | None = None) -> dict:
    cfg = load_yaml(repo_root() / "configs" / "model.yaml")
    name = profile or cfg.get("profile", "baby")
    if name not in cfg or not isinstance(cfg[name], dict):
        raise ValueError(f"Unknown model profile: {name}")
    return dict(cfg[name])


def gpt2_config_from_dict(d: dict) -> GPT2Config:
    """Build GPT2Config for random init (no pretrained weights)."""
    return GPT2Config(
        vocab_size=int(d["vocab_size"]),
        n_positions=int(d["n_positions"]),
        n_embd=int(d["n_embd"]),
        n_layer=int(d["n_layer"]),
        n_head=int(d["n_head"]),
        n_inner=int(d.get("n_inner", 4 * int(d["n_embd"]))),
        resid_pdrop=float(d.get("resid_pdrop", 0.1)),
        embd_pdrop=float(d.get("embd_pdrop", 0.1)),
        attn_pdrop=float(d.get("attn_pdrop", 0.1)),
        activation_function="gelu_new",
    )


def device_auto():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
