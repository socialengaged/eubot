#!/usr/bin/env python3
"""
Autonomous loop: generate → test → score (semantic + novelty + len) → mutate → retest → compare → promote.

  set PYTHONPATH=.
  python -m ai_engine.tests.run_quality_pipeline \\
    --url http://127.0.0.1:8080/v1/chat/completions \\
    --delay 3 --max 10

Promote only when score and avg_semantic do not regress (real quality ↑).
Constraints: max ≤ 10 per batch, delay ≥ 3, sequential HTTP only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ai_engine.tests.dataset_mutator import build_improved_dataset
from ai_engine.tests.generate_dataset import generate_dataset, slug_topic
from ai_engine.tests.test_quality_batch import (
    BEST_DATASET_PATH,
    BEST_OUTPUTS_PATH,
    IMPROVED_DATASET_PATH,
    MAX_QUESTIONS_CAP,
    MIN_DELAY_SEC,
    append_mutation_history,
    best_pattern_from_rows,
    best_outputs_from_rows,
    quality_targets_met,
    run_batch,
)

_TESTS_DIR = Path(__file__).resolve().parent
_DATASETS_DIR = _TESTS_DIR / "datasets"
PATTERN_HINTS_PATH = _DATASETS_DIR / "_pattern_hints.json"

DEFAULT_TOPICS = ["mindset", "decision making", "psychology"]


def _load_pattern_hints() -> list[str] | None:
    if not PATTERN_HINTS_PATH.is_file():
        return None
    try:
        raw = json.loads(PATTERN_HINTS_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
    except (json.JSONDecodeError, OSError):
        pass
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Generate datasets, batch test, mutate, compare, promote.")
    p.add_argument("--mode", choices=["v1", "v2"], default="v1")
    p.add_argument("--url", default=None, help="POST URL (default: BABY_CHAT_URL or v1 default)")
    p.add_argument("--model", default=None)
    p.add_argument("--delay", type=float, default=MIN_DELAY_SEC)
    p.add_argument("--max", type=int, default=10, dest="max_q")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--verbose", action="store_true", help="Print each Q/A and SEMANTIC during batch")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Stricter WEAK rules (same as test_quality_batch --strict)",
    )
    p.add_argument(
        "--use-patterns",
        action="store_true",
        help="Bias generation using datasets/_pattern_hints.json if present (from last BEST PATTERN)",
    )
    p.add_argument(
        "--topics",
        nargs="*",
        default=None,
        help="Override default topics (default: mindset, decision making, psychology)",
    )
    p.add_argument(
        "--skip-mutation",
        action="store_true",
        help="Only run generation + first batch (no improved dataset pass)",
    )
    args = p.parse_args()

    delay = max(MIN_DELAY_SEC, float(args.delay))
    max_q = max(1, min(MAX_QUESTIONS_CAP, int(args.max_q)))
    topics = list(args.topics) if args.topics else list(DEFAULT_TOPICS)

    hints: list[str] | None = _load_pattern_hints() if args.use_patterns else None
    if hints:
        print(f"Using pattern hints from {PATTERN_HINTS_PATH}: {hints[:2]}...", flush=True)

    _DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for topic in topics:
        prompts = generate_dataset(topic, n=10, pattern_hints=hints)
        out_path = _DATASETS_DIR / f"convo_auto_{slug_topic(topic)}.json"
        out_path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Generated {len(prompts)} prompts -> {out_path}", flush=True)
        paths.append(out_path)

    model = (args.model or "").strip() or None
    timeout = max(1.0, float(args.timeout))

    print("--- Phase 1: batch on generated datasets ---", flush=True)
    r1 = run_batch(
        paths,
        mode=args.mode,
        url=args.url,
        model=model,
        delay=delay,
        max_q=max_q,
        timeout=timeout,
        verbose=args.verbose,
        strict=args.strict,
    )

    best = r1.get("best_name")
    score1 = float(r1.get("best_score") or 0.0)
    report = r1.get("report") or {}
    print(f"BEST DATASET: {best} (score={score1})", flush=True)

    rows_best = (r1.get("rows_by_stem") or {}).get(best) or []
    if rows_best:
        pat = best_pattern_from_rows(rows_best)
        if pat:
            print(f"BEST PATTERN: {pat!r}", flush=True)
            PATTERN_HINTS_PATH.write_text(json.dumps([pat], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved pattern hint -> {PATTERN_HINTS_PATH}", flush=True)

    if best and best in report:
        ok, msg = quality_targets_met(report[best])
        print(f"Quality targets (monitoring): {'PASS' if ok else 'FAIL'} — {msg}", flush=True)

    if not best:
        print("No best dataset selected (check API and datasets).", flush=True)
        raise SystemExit(1)

    if args.skip_mutation:
        return

    improved_prompts = build_improved_dataset(rows_best, max_prompts=20)
    if not improved_prompts:
        print("Mutation skipped: no effective HIGH responses for _improved_dataset.json", flush=True)
        return

    IMPROVED_DATASET_PATH.write_text(
        json.dumps(improved_prompts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(improved_prompts)} prompts -> {IMPROVED_DATASET_PATH}", flush=True)

    print("--- Phase 2: batch on improved (mutated) dataset ---", flush=True)
    improved_stem = IMPROVED_DATASET_PATH.stem
    r2 = run_batch(
        [IMPROVED_DATASET_PATH],
        mode=args.mode,
        url=args.url,
        model=model,
        delay=delay,
        max_q=max_q,
        timeout=timeout,
        verbose=args.verbose,
        strict=args.strict,
        merge_report=dict(report),
        write_history=False,
        write_artifacts=False,
    )

    merged_report = r2.get("report") or {}
    imp_stats = merged_report.get(improved_stem) or {}
    score2 = float(imp_stats.get("score", 0.0))
    sem1 = float(report[best].get("avg_semantic", 0.0))
    sem2 = float(imp_stats.get("avg_semantic", 0.0))

    print(
        f"Improved: score={score2} avg_semantic={sem2} | baseline: score={score1} avg_semantic={sem1}",
        flush=True,
    )
    append_mutation_history(
        score1,
        score2,
        best_dataset=improved_stem,
        prev_avg_semantic=sem1,
        new_avg_semantic=sem2,
    )

    if score2 > score1 and sem2 >= sem1:
        print("IMPROVEMENT CONFIRMED", flush=True)
        imp_rows = (r2.get("rows_by_stem") or {}).get(improved_stem) or []
        prompts_use = improved_prompts[:max_q]
        BEST_DATASET_PATH.write_text(
            json.dumps(prompts_use, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with open(BEST_OUTPUTS_PATH, "w", encoding="utf-8") as f:
            json.dump(best_outputs_from_rows(imp_rows), f, ensure_ascii=False, indent=2)
        print(f"Promoted improved -> {BEST_DATASET_PATH} and {BEST_OUTPUTS_PATH}", flush=True)
    elif score2 > score1 and sem2 < sem1:
        print("FAKE IMPROVEMENT DETECTED", flush=True)
        print("No promotion (score up but avg_semantic down).", flush=True)
    else:
        print("Improved did not beat baseline on score+semantic; keeping prior _best_dataset.json", flush=True)


if __name__ == "__main__":
    main()
