#!/usr/bin/env python3
"""
Fase 10 — Classici scientifici da Project Gutenberg (fisica, matematica, astronomia).

Nota: verificare ogni ID su gutenberg.org prima della produzione.

Env: EUROBOT_SCIENCE_OUT (default ./output/gutenberg_science)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

SCIENCE_BOOKS: list[tuple[str, int]] = [
    # Fisica
    ("newton_principia.txt", 28233),
    ("galileo_dialogue.txt", 46978),
    ("copernicus_revolutions.txt", 55504),
    ("newton_opticks.txt", 33504),
    ("maxwell_electricity_magnetism.txt", 55432),
    ("einstein_relativity.txt", 36276),
    ("faraday_experimental_researches.txt", 14986),
    # Matematica
    ("euclid_elements.txt", 21076),
    ("dudeney_amusements_math.txt", 16713),
    ("cajori_history_mathematics.txt", 31061),
    ("thompson_calculus_made_easy.txt", 33283),
    ("whitehead_intro_mathematics.txt", 39568),
    ("poincare_foundations_science.txt", 39713),
    ("russell_intro_math_philosophy.txt", 61684),
    # Astronomia
    ("flammarion_astronomy_amateurs.txt", 9866),
    ("clerke_popular_history_astronomy.txt", 10790),
    ("ball_story_of_heavens.txt", 27378),
    ("lodge_pioneers_of_science.txt", 45684),
    ("bryant_brief_history_astronomy.txt", 31624),
    ("dick_sidereal_heavens.txt", 48060),
    ("newcomb_astronomy_everyone.txt", 42141),
]


def main() -> None:
    out = Path(
        os.environ.get(
            "EUROBOT_SCIENCE_OUT",
            str(Path(__file__).resolve().parent / "output" / "gutenberg_science"),
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    os.environ["EUROBOT_GUTENBERG_OUT"] = str(out)

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("gutenberg_esoteric", here / "download_gutenberg_esoteric.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.BOOKS = SCIENCE_BOOKS
    mod.OUTPUT_DIR = out
    mod.main()


if __name__ == "__main__":
    main()
