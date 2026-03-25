#!/usr/bin/env python3
"""
Console chat preview: real USER/BOT turns with LEN, QLT, semantic (from run_questions), LAT.

Default POST URL is production (OVH nginx → RunPod Baby):

  https://eubot.seo.srl/api/baby/v1/chat/completions

From repo root:

  set PYTHONPATH=.
  python -m ai_engine.tests.run_chat_preview
  python -m ai_engine.tests.run_chat_preview --dataset ai_engine/tests/datasets/convo_real_world.json --max 10
  python -m ai_engine.tests.run_chat_preview --url http://127.0.0.1:18080/v1/chat/completions   # local SSH tunnel

Uses r[\"semantic\"] from test_live_safe.run_questions (no extra semantic logic here).
--verbose: extra per-request lines from run_questions (Q/A + SEMANTIC).
--save: also writes ai_engine/tests/chat_preview.txt
--debug: raw I/O diagnostics (URL, httpx probe, full result dicts).
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx

# Heuristic: server-side contextual / legacy fallbacks (tune if copy changes)
_FALLBACK_MARKERS = (
    "Ti rispondo in modo semplice",
    "Ecco una spiegazione chiara",
    "Puoi spiegarmi meglio",
    "Let me give you a simple",
    "Sto avendo un attimo",
    "focus on one small action",
    "Ok, dimmi meglio",
)


def _is_fallback_like(answer: str) -> bool:
    a = (answer or "").strip()
    return any(m in a for m in _FALLBACK_MARKERS)


def _diversity_score(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _has_obvious_loop(answer: str) -> bool:
    """Same word repeated >5 times in a row → loop suspect."""
    w = answer.split()
    if len(w) < 6:
        return False
    run = 1
    for i in range(1, len(w)):
        if w[i].lower() == w[i - 1].lower():
            run += 1
            if run > 5:
                return True
        else:
            run = 1
    return False
from pathlib import Path
from typing import Any

_TESTS_DIR = Path(__file__).resolve().parent
_PREVIEW_OUT = _TESTS_DIR / "chat_preview.txt"
_DEFAULT_DATASET = _TESTS_DIR / "datasets" / "_best_dataset.json"

_REPO = _TESTS_DIR.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ai_engine.tests.test_live_safe import (  # noqa: E402
    MAX_QUESTIONS_CAP,
    MIN_DELAY_SEC,
    DEFAULT_TIMEOUT,
    LiveRunParams,
    resolve_url,
    run_questions,
)

# Production: nginx on OVH proxies /api/baby/ → tunnel → pod :8080 (see eubot-coder/deploy/nginx-eubot.seo.srl.conf)
DEFAULT_PRODUCTION_BABY_URL = "https://eubot.seo.srl/api/baby/v1/chat/completions"


def _resolve_dataset(arg: Path) -> Path:
    if arg.is_absolute() and arg.is_file():
        return arg
    for base in (Path.cwd(), _TESTS_DIR, _TESTS_DIR / "datasets"):
        c = base / arg
        if c.is_file():
            return c.resolve()
    return (Path.cwd() / arg).resolve()


def main() -> None:
    p = argparse.ArgumentParser(description="Print a readable chat preview (USER/BOT + metrics) to the console.")
    p.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help="JSON array of prompt strings (default: datasets/_best_dataset.json)",
    )
    p.add_argument("--max", type=int, default=5, help=f"How many prompts to send (capped at {MAX_QUESTIONS_CAP})")
    p.add_argument(
        "--only-high",
        action="store_true",
        dest="only_high",
        help="After the run, keep only rows with QLT HIGH and semantic >= 0.4",
    )
    p.add_argument(
        "--url",
        default=DEFAULT_PRODUCTION_BABY_URL,
        help=f"POST URL (default: production {DEFAULT_PRODUCTION_BABY_URL})",
    )
    p.add_argument("--model", default=None, help="Optional v1 model name")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Per-request debug from run_questions (Q/A, length, QLT, SEMANTIC, LAT) before the formatted block",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help=f"Also write the same output to {_PREVIEW_OUT.name}",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print URL, raw results list, direct httpx test POST, and per-row RAW/error/status (I/O debugging)",
    )
    args = p.parse_args()

    ds = _resolve_dataset(args.dataset)
    if not ds.is_file():
        print(f"DATASET NOT FOUND: {ds}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    raw = json.loads(ds.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        print("Dataset must be a non-empty JSON array of strings.", file=sys.stderr, flush=True)
        raise SystemExit(1)

    questions = [str(x).strip() for x in raw if str(x).strip()]
    max_q = max(1, min(MAX_QUESTIONS_CAP, int(args.max)))
    items = questions[:max_q]

    mode = "v1"
    final_url = resolve_url(mode, args.url)
    model = (args.model or "").strip() or None

    params = LiveRunParams(
        mode=mode,
        url=final_url,
        model=model,
        delay_sec=MIN_DELAY_SEC,
        timeout=DEFAULT_TIMEOUT,
        verbose=args.verbose,
        write_log=False,
    )

    if args.debug:
        print("DEBUG URL (arg):", args.url, flush=True)
        print("DEBUG URL (resolved):", final_url, flush=True)

    results = run_questions(items, params)

    if not results:
        print("NO RESULTS RETURNED", flush=True)
        raise SystemExit(1)

    if args.debug:
        print("DEBUG RESULTS:", flush=True)
        print(results, flush=True)

        test_payload: dict[str, Any] = {"messages": [{"role": "user", "content": "test"}]}
        if model:
            test_payload["model"] = model
        print("TEST CALL...", flush=True)
        try:
            r = httpx.post(
                final_url,
                json=test_payload,
                timeout=10.0,
                headers={"Content-Type": "application/json"},
            )
            print("STATUS:", r.status_code, flush=True)
            print("BODY:", r.text[:300], flush=True)
        except Exception as e:
            print("REQUEST ERROR:", str(e), flush=True)

    if args.only_high:
        results = [r for r in results if r.get("qlt") == "HIGH" and float(r.get("semantic", 0.0)) >= 0.4]

    log: list[str] = []

    def emit(s: str = "") -> None:
        print(s, end="", flush=True)
        log.append(s)

    emit("\n================ CHAT OUTPUT ================\n\n")

    ok_results: list[dict[str, Any]] = []
    for i, r in enumerate(results, 1):
        if args.debug:
            print(f"\n[{i}] RAW:", r, flush=True)
        if not r.get("ok"):
            err = r.get("error") or "request failed"
            if args.debug:
                print(f"[{i}] ERROR FULL:", r.get("error"), flush=True)
                print(f"[{i}] STATUS:", r.get("http_status"), flush=True)
            emit(f"[{i}] ERROR: {err}\n")
            emit("\n--------------------------------------------\n\n")
            continue

        ok_results.append(r)
        sem = float(r.get("semantic", 0.0))
        emit(f"[{i}] USER:\n{r['q']}\n\n")
        emit(f"[{i}] BOT:\n{r['a']}\n\n")
        emit(
            f"[LEN:{r['length']} | QLT:{r['qlt']} | SEM:{sem:.2f} | LAT:{float(r['lat']):.2f}s]\n"
        )
        emit("\n--------------------------------------------\n\n")

    if ok_results:
        avg_len = sum(int(r.get("length", 0)) for r in ok_results) / len(ok_results)
        avg_sem = sum(float(r.get("semantic", 0.0)) for r in ok_results) / len(ok_results)
        high = sum(1 for r in ok_results if r.get("qlt") == "HIGH")
        low = sum(1 for r in ok_results if r.get("qlt") == "LOW")

        n_ok = len(ok_results)
        fb_n = sum(1 for r in ok_results if _is_fallback_like(r.get("a") or ""))
        loop_n = sum(1 for r in ok_results if _has_obvious_loop(r.get("a") or ""))
        fallback_rate = 100.0 * fb_n / n_ok if n_ok else 0.0
        div_avg = sum(_diversity_score(r.get("a") or "") for r in ok_results) / n_ok

        emit("\n================ SUMMARY ================\n\n")
        emit(f"AVG_LEN: {avg_len:.1f}\n")
        emit(f"AVG_SEMANTIC: {avg_sem:.2f}\n")
        emit(f"FALLBACK_RATE: {fallback_rate:.1f}% ({fb_n}/{n_ok})\n")
        emit(f"AVG_DIVERSITY: {div_avg:.3f}\n")
        emit(f"LOOP_SUSPECT_ROWS: {loop_n}\n")
        emit(f"HIGH: {high}\n")
        emit(f"LOW: {low}\n")

    if args.save:
        _PREVIEW_OUT.parent.mkdir(parents=True, exist_ok=True)
        _PREVIEW_OUT.write_text("".join(log), encoding="utf-8")
        print(f"\nSaved: {_PREVIEW_OUT}", flush=True)


if __name__ == "__main__":
    main()
