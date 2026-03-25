#!/usr/bin/env python3
"""
Safe, sequential live check for Eurobot Baby (v1 production on :8080, optional v2).

Does not touch training or start/stop server processes. Blocking HTTP only (no threads/async).

v1 payload is messages-only by default; add --model if the API requires a model name.

  python ai_engine/tests/test_live_safe.py --mode v1 --max 3
  python ai_engine/tests/test_live_safe.py --mode v1 --model eurobot-baby --max 3

Production dataset:

  python ai_engine/tests/test_live_safe.py --mode v1 --dataset ai_engine/tests/datasets/convo_real_world.json --delay 3 --max 10

QA smoke (defaults: delay 3s, max 10 requests):
  python ai_engine/tests/test_live_safe.py --mode v1 --url https://eubot.seo.srl/api/baby/v1/chat/completions --delay 3 --max 10

Batch evaluation imports ``run_questions`` + ``LiveRunParams`` from this module.

Override URL (optional env BABY_CHAT_URL). Timeout default 10s; use --timeout 60 under heavy GPU load.

Do not run serve_v2_extension on the training pod until a separate inference GPU is available.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from ai_engine.tests.quality_semantics import semantic_score

_TESTS_DIR = Path(__file__).resolve().parent
DATASET_PATH = _TESTS_DIR / "datasets" / "convo_growth_set_01.json"
LOG_FILE = _TESTS_DIR / "output_live_safe.log"

DEFAULT_V1_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_V2_URL = "http://127.0.0.1:8081/v2/chat"

MIN_DELAY_SEC = 3.0
MAX_QUESTIONS_CAP = 10
DEFAULT_TIMEOUT = 10.0


@dataclass
class LiveRunParams:
    """Parameters for run_questions (no argparse dependency)."""

    mode: Literal["v1", "v2"]
    url: str
    model: str | None = None
    delay_sec: float = MIN_DELAY_SEC
    timeout: float = DEFAULT_TIMEOUT
    verbose: bool = True
    write_log: bool = True


def _resolve_dataset_path(p: Path) -> Path:
    """Resolve relative paths from cwd, tests dir, or tests/datasets/."""
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        Path.cwd() / p,
        _TESTS_DIR / p,
        _TESTS_DIR / "datasets" / p.name,
    ]
    if p.parts and p.parts[0] == "datasets":
        candidates.append(_TESTS_DIR / p)
    for c in candidates:
        try:
            r = c.resolve()
            if r.exists():
                return r
        except OSError:
            continue
    return (Path.cwd() / p).resolve()


def load_questions(dataset_path: Path) -> list[str]:
    """Load prompt list from JSON array of strings; fallback to ping if missing/invalid."""
    if dataset_path.exists():
        try:
            raw = json.loads(dataset_path.read_text(encoding="utf-8"))
            if isinstance(raw, list) and raw:
                return [str(x).strip() for x in raw if str(x).strip()]
        except (json.JSONDecodeError, OSError):
            pass
    return ["ping"]


def log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _one_line(s: str) -> str:
    return " ".join(s.splitlines())


def _quality_tag(answer: str) -> Literal["LOW", "MID", "HIGH"]:
    n = len(answer)
    if n < 50:
        return "LOW"
    if n < 150:
        return "MID"
    return "HIGH"


def _parse_answer(mode: Literal["v1", "v2"], data: dict[str, Any]) -> str:
    if mode == "v1":
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") or {}
            if isinstance(msg, dict):
                return str(msg.get("content") or "")
        return ""
    r = data.get("response")
    if isinstance(r, str) and r.strip():
        return r
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            return str(msg.get("content") or "")
    return ""


def _build_payload(mode: Literal["v1", "v2"], question: str, model: str | None) -> dict[str, Any]:
    if mode == "v1":
        p: dict[str, Any] = {"messages": [{"role": "user", "content": question}]}
        if model:
            p["model"] = model
        return p
    return {"messages": [{"role": "user", "content": question}]}


def _post_with_retry(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    quiet: bool = False,
) -> tuple[httpx.Response, float]:
    """POST once; on any failure retry after 2s (single retry, safe for flaky network/GPU)."""
    t0 = time.time()
    try:
        r = client.post(url, json=payload, headers={"Content-Type": "application/json"})
        return r, time.time() - t0
    except Exception as e:
        if not quiet:
            print(f"Request error (will retry once): {e}\n", flush=True)
        time.sleep(2)
        t0 = time.time()
        r = client.post(url, json=payload, headers={"Content-Type": "application/json"})
        return r, time.time() - t0


def resolve_url(mode: Literal["v1", "v2"], url_arg: str | None) -> str:
    env = (os.environ.get("BABY_CHAT_URL") or "").strip()
    if url_arg and url_arg.strip():
        return url_arg.strip()
    if env:
        return env
    return DEFAULT_V1_URL if mode == "v1" else DEFAULT_V2_URL


def run_questions(questions: list[str], params: LiveRunParams) -> list[dict[str, Any]]:
    """
    Run sequential POSTs for each question. Returns one dict per question:

    - q, a, length, lat, qlt (LOW|MID|HIGH)
    - ok: bool
    - http_status: int | None
    - error: str | None (if exception after retry)
    """
    results: list[dict[str, Any]] = []
    quiet = not params.verbose
    n = len(questions)
    with httpx.Client(timeout=params.timeout) as client:
        for i, question in enumerate(questions, start=1):
            payload = _build_payload(params.mode, question, params.model)
            if params.verbose:
                print(f"--- [{i}/{n}] ---", flush=True)
                print(f"Q: {question}\n", flush=True)

            t0 = time.time()
            answer = ""
            qlt: Literal["LOW", "MID", "HIGH"] = "LOW"
            ok = False
            http_status: int | None = None
            err_msg: str | None = None
            lat = 0.0

            try:
                r, lat = _post_with_retry(client, params.url, payload, quiet=quiet)
                http_status = r.status_code
                text_body = r.text
                if r.status_code != 200:
                    if params.verbose:
                        print(f"HTTP {r.status_code}\n{text_body[:1500]}\n", flush=True)
                    if params.write_log:
                        log(
                            f"{i} | MODE:{params.mode} | Q: {_one_line(question)} | A: HTTP_{r.status_code} | LEN: 0 | QLT:LOW | LAT:{lat:.2f}s"
                        )
                else:
                    try:
                        data = json.loads(text_body)
                    except json.JSONDecodeError:
                        data = {}
                    if not isinstance(data, dict):
                        data = {}
                    answer = _parse_answer(params.mode, data)
                    qlt = _quality_tag(answer)
                    ok = True
                    sem = float(semantic_score(answer))
                    if params.verbose:
                        print(f"A: {answer}\n", flush=True)
                        print(
                            f"Response length: {len(answer)} chars | QLT: {qlt} | LAT: {lat:.2f}s\n",
                            flush=True,
                        )
                        print(f"SEMANTIC: {sem:.2f}\n", flush=True)
                    if params.write_log:
                        if len(answer) < 50:
                            log(f"WARNING_SHORT | MODE:{params.mode} | {_one_line(question)}")
                        log(
                            f"{i} | MODE:{params.mode} | Q: {_one_line(question)} | A: {_one_line(answer)} | LEN: {len(answer)} | QLT:{qlt} | LAT:{lat:.2f}s"
                        )
            except Exception as e:
                lat = time.time() - t0
                err_msg = str(e)
                if params.verbose:
                    print(f"Request failed (after retry): {e}\n", flush=True)
                if params.write_log:
                    log(
                        f"{i} | MODE:{params.mode} | Q: {_one_line(question)} | A: REQUEST_ERROR | LEN: 0 | QLT:LOW | LAT:{lat:.2f}s"
                    )

            results.append(
                {
                    "q": question,
                    "a": answer,
                    "length": len(answer),
                    "lat": lat,
                    "qlt": qlt,
                    "ok": ok,
                    "http_status": http_status,
                    "error": err_msg,
                    "semantic": sem if ok else 0.0,
                }
            )

            if i < n:
                time.sleep(params.delay_sec)

    return results


def run_live_safe(
    url: str,
    *,
    mode: Literal["v1", "v2"],
    dataset_path: Path,
    model: str | None = None,
    delay_sec: float = MIN_DELAY_SEC,
    max_questions: int = MAX_QUESTIONS_CAP,
    timeout: float = DEFAULT_TIMEOUT,
) -> int:
    """
    Send sequential POST requests. Returns 0 if all HTTP 200, else 1.
    """
    questions = load_questions(dataset_path)
    cap = max(1, min(max_questions, MAX_QUESTIONS_CAP, len(questions)))
    items = questions[:cap]

    print(f"Target: {url}", flush=True)
    print(f"MODE: {mode}", flush=True)
    if model:
        print(f"Model: {model}", flush=True)
    print(f"Dataset: {dataset_path}", flush=True)
    print(f"Questions: {len(items)} | delay: {delay_sec}s | timeout: {timeout}s | sequential only\n", flush=True)

    params = LiveRunParams(
        mode=mode,
        url=url,
        model=model,
        delay_sec=delay_sec,
        timeout=timeout,
        verbose=True,
        write_log=True,
    )
    results = run_questions(items, params)
    ok = all(r.get("ok") for r in results)
    print("Done.", flush=True)
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(
        description="Safe sequential Baby chat live check (v1 /v1/chat/completions or v2 /v2/chat)."
    )
    p.add_argument("--mode", choices=["v1", "v2"], default="v1", help="API shape (default v1 production)")
    p.add_argument(
        "--url",
        default=None,
        help="Full POST URL (default: from BABY_CHAT_URL or mode default)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Optional model name for v1 payload (omit if server accepts messages-only)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=MIN_DELAY_SEC,
        help=f"Seconds between requests (minimum {MIN_DELAY_SEC} enforced for GPU safety)",
    )
    p.add_argument(
        "--max",
        type=int,
        default=10,
        dest="max_q",
        help=f"Max questions per run (≤{MAX_QUESTIONS_CAP})",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout seconds per request (default 10)",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="JSON file: array of prompt strings",
    )
    args = p.parse_args()

    mode = args.mode  # type: Literal["v1", "v2"]
    delay = float(args.delay)
    if delay < MIN_DELAY_SEC:
        print(
            f"Note: --delay {delay} clamped to {MIN_DELAY_SEC}s (GPU safety).",
            file=sys.stderr,
            flush=True,
        )
        delay = MIN_DELAY_SEC
    max_q = max(1, min(MAX_QUESTIONS_CAP, int(args.max_q)))
    timeout = max(1.0, float(args.timeout))

    final_url = resolve_url(mode, args.url)
    ds = _resolve_dataset_path(args.dataset)
    model = (args.model or "").strip() or None

    sys.exit(
        run_live_safe(
            final_url,
            mode=mode,
            dataset_path=ds,
            model=model,
            delay_sec=delay,
            max_questions=max_q,
            timeout=timeout,
        )
    )


if __name__ == "__main__":
    main()
