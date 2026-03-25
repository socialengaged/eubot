#!/usr/bin/env python3
"""Run on RunPod: add 07_esoteric_sage_corpus.txt to TRAIN_PARTS in build_dataset.py"""
from pathlib import Path

p = Path("/workspace/eurobot_baby/scripts/build_dataset.py")
t = p.read_text(encoding="utf-8")
if "07_esoteric_sage_corpus.txt" in t:
    print("already_patched")
else:
    t = t.replace(
        '"05_sacred.txt",',
        '"05_sacred.txt",\n    "07_esoteric_sage_corpus.txt",',
        1,
    )
    p.write_text(t, encoding="utf-8")
    print("patched_ok")
