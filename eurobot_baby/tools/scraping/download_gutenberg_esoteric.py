#!/usr/bin/env python3
"""
Fase 1 — Download esoteric/occult books from Project Gutenberg (plain text).

Env:
  EUROBOT_GUTENBERG_OUT  Output directory (default: ./output/gutenberg_esoteric)
  EUROBOT_DELAY_SEC      Delay between requests (default: 2.0)

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
    os.environ.get("EUROBOT_GUTENBERG_OUT", str(Path(__file__).resolve().parent / "output" / "gutenberg_esoteric"))
)

# (filename, gutenberg_id) — verificare ID su gutenberg.org
BOOKS: list[tuple[str, int]] = [
    ("kybalion.txt", 14209),
    ("lesser_key_solomon.txt", 7242),
    ("blavatsky_secret_doctrine_v1.txt", 54962),
    ("blavatsky_secret_doctrine_v2.txt", 54968),
    ("blavatsky_isis_unveiled_v1.txt", 54500),
    ("blavatsky_isis_unveiled_v2.txt", 54501),
    ("blavatsky_key_to_theosophy.txt", 55834),
    ("steiner_occult_science.txt", 42053),
    ("steiner_mystics_renaissance.txt", 36487),
    ("underhill_practical_mysticism.txt", 23085),
    ("hall_initiates_of_flame.txt", 57474),
    ("blake_marriage_heaven_hell.txt", 45315),
    ("pike_morals_and_dogma.txt", 19447),
    ("silberer_hidden_symbolism_alchemy.txt", 55959),
    ("burgoyne_light_of_egypt.txt", 55879),
    ("besant_esoteric_christianity.txt", 54848),
    ("leadbeater_astral_plane.txt", 55131),
    ("hartmann_paracelsus.txt", 55941),
    ("levi_transcendental_magic.txt", 55685),
    ("levi_dogme_et_rituel_en.txt", 55686),
    ("lytton_zanoni.txt", 2083),
    ("lytton_strange_story.txt", 2084),
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
            if len(text) > 200:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8", errors="replace")
                print(f"OK   {fname} ({len(text) // 1024} KB) <- {url}")
                return
        except Exception as e:
            print(f"ERR  {fname} {url}: {e}")
    print(f"FAIL {fname} (gid={gid})")


def main() -> None:
    for fname, gid in BOOKS:
        download_one(fname, gid)
        time.sleep(DELAY)
    print(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
