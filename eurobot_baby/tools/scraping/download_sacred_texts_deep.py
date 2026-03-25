#!/usr/bin/env python3
"""
Fase 2 — Deep scrape of sacred-texts.com section indexes (polite delays).

Env:
  EUROBOT_SACRED_OUT   Output root (default: ./output/sacred_texts)
  EUROBOT_DELAY_SEC    Delay between page fetches (default: 2.5)

Eseguire con nohup/tmux; può richiedere ore.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

DELAY = float(os.environ.get("EUROBOT_DELAY_SEC", "2.5"))
BASE = "https://www.sacred-texts.com"
OUTPUT_DIR = Path(os.environ.get("EUROBOT_SACRED_OUT", str(Path(__file__).resolve().parent / "output" / "sacred_texts")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (compatible; EurobotResearch/1.0; academic corpus)",
        "Accept": "text/html,*/*",
    }
)

SECTIONS: list[tuple[str, str]] = [
    ("alchemy", "/alc/index.htm"),
    ("esoteric_western", "/eso/index.htm"),
    ("rosicrucian", "/eso/ros/index.htm"),
    ("theosophy", "/the/index.htm"),
    ("gnosticism", "/gno/index.htm"),
    ("grimoires", "/grim/index.htm"),
    ("freemasonry", "/mas/index.htm"),
    ("egyptian", "/egy/index.htm"),
    ("astrology", "/astro/index.htm"),
    ("hinduism_select", "/hin/index.htm"),
    ("buddhism_select", "/bud/index.htm"),
    ("taoism", "/tao/index.htm"),
    ("kabbalah", "/jud/index.htm"),
]


def fetch(url: str, timeout: int = 60) -> bytes:
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def html_to_text(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "iframe"]):
        tag.decompose()
    return soup.get_text(separator="\n").strip()


def scrape_section(name: str, index_path: str) -> int:
    index_url = BASE + index_path
    section_dir = OUTPUT_DIR / name
    section_dir.mkdir(parents=True, exist_ok=True)

    try:
        index_html = fetch(index_url)
    except Exception as e:
        print(f"FAIL index {name}: {e}")
        return 0

    soup = BeautifulSoup(index_html, "html.parser")
    base_url = index_url.rsplit("/", 1)[0] + "/"

    links: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and BASE not in href:
            continue
        if href.endswith((".htm", ".html", ".txt")):
            full = urljoin(base_url, href)
            if BASE in full:
                links.add(full)

    print(f"  {name}: {len(links)} pages found")
    count = 0
    for url in sorted(links):
        slug = hashlib.md5(url.encode()).hexdigest()[:10]
        fname = re.sub(r"[^\w]", "_", url.split("/")[-1].replace(".htm", "").replace(".html", ""))
        out_path = section_dir / f"{fname}_{slug}.txt"
        if out_path.is_file() and out_path.stat().st_size > 200:
            count += 1
            continue
        try:
            data = fetch(url)
            text = html_to_text(data)
            if len(text) > 100:
                out_path.write_text(f"Source: {url}\n\n{text}\n", encoding="utf-8", errors="replace")
                count += 1
        except Exception as e:
            print(f"  ERR {url}: {e}")
        time.sleep(DELAY)

    print(f"  {name}: {count} texts saved")
    return count


def main() -> None:
    total = 0
    for name, path in SECTIONS:
        print(f"\n--- Section: {name} ---")
        total += scrape_section(name, path)
        time.sleep(5)
    print(f"\nTotal section passes: {total}. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
