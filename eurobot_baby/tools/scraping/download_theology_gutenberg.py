#!/usr/bin/env python3
"""
Teologia / filosofia della religione — testi Project Gutenberg (public domain).

Env:
  EUROBOT_GUTENBERG_THEOLOGY_OUT  Output (default: ./output/gutenberg_theology)
  EUROBOT_DELAY_SEC               Delay tra richieste (default: 2.0)

Verificare gli ID su https://www.gutenberg.org prima di produzione.
"""
from __future__ import annotations

import os
import re
import time

import requests
from pathlib import Path

DELAY = float(os.environ.get("EUROBOT_DELAY_SEC", "2.0"))
OUTPUT_DIR = Path(
    os.environ.get(
        "EUROBOT_GUTENBERG_THEOLOGY_OUT",
        str(Path(__file__).resolve().parent / "output" / "gutenberg_theology"),
    )
)

# (filename, gutenberg_id) — teologia / filosofia della religione (verificare su gutenberg.org)
BOOKS: list[tuple[str, int]] = [
    ("augustine_confessions.txt", 3296),
    ("james_varieties_religious_experience.txt", 621),
    ("bunyan_pilgrims_progress.txt", 674),
    ("butler_analogy_religion.txt", 5827),
    ("calvin_institutes.txt", 16478),
    ("kempis_imitation_of_christ.txt", 1653),
    ("paley_natural_theology.txt", 3328),
    ("kant_religion_within_bounds.txt", 5283),
    ("spinoza_theological_political_treatise.txt", 989),
]


def gutenberg_url(gid: int) -> str:
    return f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt"


def gutenberg_url_fallback(gid: int) -> str:
    return f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"


def strip_boilerplate(text: str) -> str:
    text = re.sub(
        r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG.*?\*\*\*",
        "",
        text,
        count=1,
        flags=re.I | re.S,
    )
    text = re.sub(r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG.*", "", text, flags=re.I | re.S)
    return text.strip()


def download_one(fname: str, gid: int) -> None:
    path = OUTPUT_DIR / fname
    if path.is_file() and path.stat().st_size > 512:
        print(f"SKIP {fname}")
        return
    for url in (gutenberg_url(gid), gutenberg_url_fallback(gid)):
        try:
            r = requests.get(url, timeout=90, headers={"User-Agent": "EurobotDataset/1.0 (research)"})
            if r.status_code != 200:
                continue
            text = strip_boilerplate(r.text)
            if len(text) < 200:
                continue
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", errors="replace")
            print(f"OK {fname} ({len(text)//1024} KB) <- {url}")
            time.sleep(DELAY)
            return
        except Exception as e:
            print(f"ERR {fname} {url}: {e}")
    print(f"FAIL {fname} gid={gid}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, gid in BOOKS:
        download_one(fname, gid)


if __name__ == "__main__":
    main()
