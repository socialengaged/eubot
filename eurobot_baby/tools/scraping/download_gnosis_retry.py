#!/usr/bin/env python3
"""
Fase 3 — Retry download from gnosis.org / sacred-texts / gutenberg (same SOURCES as legacy script)
with browser-like User-Agent and longer delays.

Env:
  EUROBOT_RAW_DIR   Output directory for per-file downloads (default: ./output/gnosis_retry)
  EUROBOT_DELAY_SEC Default delay between requests (default: 5.0)
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("EUROBOT_RAW_DIR", str(SCRIPT_DIR / "output" / "gnosis_retry")))
DEFAULT_DELAY = float(os.environ.get("EUROBOT_DELAY_SEC", "5.0"))


def raw_path(name: str) -> Path:
    return RAW_DIR / name


SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
)


def fetch_bytes(url: str, timeout: int = 120) -> bytes:
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def html_to_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def save_text(path: Path, body: str, min_chars: int = 200) -> bool:
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if len(body) < min_chars:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", encoding="utf-8", errors="replace")
    return True


def should_skip(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 512


# Copied from backups/knowledge_pod.../download_esoteric.py
SOURCES: list[tuple[str, str, str]] = [
    ("01_nhl_gospel_of_truth.txt", "https://www.gnosis.org/nagham/got.htm", "html"),
    ("02_nhl_apocryphon_of_john.txt", "https://www.gnosis.org/nagham/apocjn.htm", "html"),
    ("03_nhl_gospel_of_thomas.txt", "https://www.gnosis.org/nagham/gosthom.htm", "html"),
    ("04_nhl_gospel_of_philip.txt", "https://www.gnosis.org/nagham/gosphil.htm", "html"),
    ("05_nhl_on_the_origin_of_the_world.txt", "https://www.gnosis.org/nagham/origin.htm", "html"),
    ("06_nhl_hypostasis_of_the_archons.txt", "https://www.gnosis.org/nagham/hypostas.htm", "html"),
    ("07_nhl_apocalypse_of_peter.txt", "https://www.gnosis.org/nagham/apopet.htm", "html"),
    ("08_nhl_thunder_perfect_mind.txt", "https://www.gnosis.org/nagham/thunder.htm", "html"),
    ("09_nhl_tripartite_tractate.txt", "https://www.gnosis.org/nagham/tripart.htm", "html"),
    ("10_nhl_sophia_of_jesus_christ.txt", "https://www.gnosis.org/nagham/soph.htm", "html"),
    ("11_nhl_acts_of_peter_and_the_twelve_apostles.txt", "https://www.gnosis.org/nagham/actpt12.htm", "html"),
    ("12_nhl_paraphrase_of_shem.txt", "https://www.gnosis.org/nagham/parashem.htm", "html"),
    ("13_nhl_second_treatise_of_the_great_seth.txt", "https://www.gnosis.org/nagham/greatset.htm", "html"),
    ("14_nhl_melchizedek.txt", "https://www.gnosis.org/nagham/melchiz.htm", "html"),
    ("15_nhl_testimony_of_truth.txt", "https://www.gnosis.org/nagham/testtrut.htm", "html"),
    ("16_nhl_discourse_on_the_eighth_and_ninth.txt", "https://www.gnosis.org/nagham/disc8-9.htm", "html"),
    ("17_nhl_prayer_of_the_apostle_paul.txt", "https://www.gnosis.org/nagham/prayerp.htm", "html"),
    ("18_nhl_authoritative_teaching.txt", "https://www.gnosis.org/nagham/authort.htm", "html"),
    ("19_nhl_concept_of_our_great_power.txt", "https://www.gnosis.org/nagham/greatpow.htm", "html"),
    ("20_nhl_thought_of_norea.txt", "https://www.gnosis.org/nagham/norea.htm", "html"),
    ("21_nhl_valentinian_exposition.txt", "https://www.gnosis.org/nagham/valent.htm", "html"),
    ("22_nhl_three_steles_of_seth.txt", "https://www.gnosis.org/nagham/steles.htm", "html"),
    ("23_nhl_zostrianos.txt", "https://www.gnosis.org/nagham/zost.htm", "html"),
    ("24_nhl_allogenes.txt", "https://www.gnosis.org/nagham/allogen.htm", "html"),
    ("30_hermetic_emerald_tablet.txt", "https://www.sacred-texts.com/eso/tab.htm", "html"),
    ("31_hermetic_corpus_hermeticum_pimander.txt", "https://www.sacred-texts.com/eso/pim/index.htm", "html"),
    ("32_alchemy_aurora_consurgens.txt", "https://www.sacred-texts.com/alc/aurora.txt", "plain"),
    ("33_paracelsus_hermetic_alchemy.txt", "https://www.sacred-texts.com/alc/para/index.htm", "html"),
    ("34_turba_philosophorum.txt", "https://www.sacred-texts.com/alc/turba.txt", "plain"),
    ("40_sefer_yetzirah_westcott.txt", "https://www.sacred-texts.com/jud/yetzirah.htm", "html"),
    ("41_kabbalah_unveiled_intro.txt", "https://www.sacred-texts.com/jud/kab/index.htm", "html"),
    ("50_plato_republic_gutenberg.txt", "https://www.gutenberg.org/files/1497/1497-0.txt", "gutenberg_txt"),
    ("51_marcus_aurelius_meditations_gutenberg.txt", "https://www.gutenberg.org/files/2680/2680-0.txt", "gutenberg_txt"),
    ("52_epictetus_enchiridion_gutenberg.txt", "https://www.gutenberg.org/files/1266/1266-0.txt", "gutenberg_txt"),
    ("53_plotinus_enneads_gutenberg.txt", "https://www.gutenberg.org/files/3800/3800-0.txt", "gutenberg_txt"),
    ("60_egyptian_book_of_dead_budge.txt", "https://www.sacred-texts.com/egy/ebod/index.htm", "html"),
    ("61_pistis_sophia_mead.txt", "https://www.sacred-texts.com/chr/ps/index.htm", "html"),
    ("62_book_of_enoch_r_h_charles.txt", "https://www.sacred-texts.com/bib/boe/index.htm", "html"),
    ("70_ptolemy_tetrabiblos.txt", "https://www.sacred-texts.com/astro/ptol/index.htm", "html"),
    ("80_tao_te_ching_legge.txt", "https://www.sacred-texts.com/tao/taote.txt", "plain"),
    ("81_upanishads_paramananda.txt", "https://www.sacred-texts.com/hin/upan/index.htm", "html"),
]


def process_one(fname: str, url: str, kind: str) -> str:
    path = raw_path(fname)
    if should_skip(path):
        return f"SKIP (exists) {fname}"
    data = fetch_bytes(url)
    if kind in ("gutenberg_txt", "plain"):
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = data.decode("latin-1", errors="replace")
        if kind == "gutenberg_txt":
            text = re.sub(r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG.*", "", text, flags=re.I | re.S)
        ok = save_text(path, text)
        return f"OK {fname}" if ok else f"WARN too_short {fname}"
    if kind == "html":
        text = html_to_text(data)
        ok = save_text(path, f"Source: {url}\n\n{text}")
        return f"OK {fname}" if ok else f"WARN too_short {fname}"
    return f"ERR unknown kind {kind}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between HTTP requests")
    ap.add_argument("--only", type=str, default="", help="Comma-separated output filenames to fetch only")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    ok = skipped = failed = 0
    for fname, url, kind in SOURCES:
        if only and fname not in only:
            continue
        try:
            msg = process_one(fname, url, kind)
            print(msg, flush=True)
            if msg.startswith("SKIP"):
                skipped += 1
            elif msg.startswith("OK"):
                ok += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAIL {fname}: {e}", flush=True)
            failed += 1
        time.sleep(args.delay)

    print(f"\nDone gnosis_retry: ok={ok} skipped={skipped} failed={failed} -> {RAW_DIR}", flush=True)


if __name__ == "__main__":
    main()
