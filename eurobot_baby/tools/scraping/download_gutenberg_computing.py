#!/usr/bin/env python3
"""
Testi storici su logica, calcolo e prime macchine analitiche (Project Gutenberg, PD).

Env: EUROBOT_GUTENBERG_COMPUTING_OUT (default ./output/gutenberg_computing)
Verificare gli ID su gutenberg.org prima della produzione.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Logica, calcolo, macchina analitica, economia delle macchine (Babbage).
# Evitare duplicati con download_stem_gutenberg_science.py (stesso merge → 31_math_classics).
COMPUTING_BOOKS: list[tuple[str, int]] = [
    ("boole_laws_of_thought.txt", 15114),
    ("menabrea_lovelace_analytical_engine.txt", 23432),
    ("babbage_economy_machinery.txt", 4238),
    ("venn_logic_of_chance.txt", 17384),
]


def main() -> None:
    out = Path(
        os.environ.get(
            "EUROBOT_GUTENBERG_COMPUTING_OUT",
            str(Path(__file__).resolve().parent / "output" / "gutenberg_computing"),
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    os.environ["EUROBOT_GUTENBERG_OUT"] = str(out)

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("gutenberg_esoteric", here / "download_gutenberg_esoteric.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.BOOKS = COMPUTING_BOOKS
    mod.OUTPUT_DIR = out
    mod.main()


if __name__ == "__main__":
    main()
