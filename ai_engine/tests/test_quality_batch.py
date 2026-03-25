#!/usr/bin/env python3
"""
Batch quality evaluation across datasets (sequential HTTP via test_live_safe.run_questions).

Goals (monitoring, not automatic gates): HIGH share > 70%, avg_len > 120, stable latency — compare runs manually.

Composite score per dataset:
  avg_len + novelty*50 + avg_semantic*40 − LOW_count*20 − duplicate_answer_count*10 − style_penalty
(avg_len / avg_semantic computed on quality-filtered ok rows only; style_penalty +15 per answer containing \"imagine\").
HIGH length-only but semantic < 0.4 is treated as MID (fake HIGH discarded).

Usage (from repo root):

  set PYTHONPATH=.
  python -m ai_engine.tests.test_quality_batch
  python -m ai_engine.tests.test_quality_batch --datasets convo_growth_set_01.json convo_real_world.json

Writes ai_engine/tests/output_quality_report.json (gitignored by default).
Appends ai_engine/tests/output_quality_history.json per run.
Saves best prompts to datasets/_best_dataset.json, top-3 names to datasets/_top3_datasets.json,
and best Q/A pairs to datasets/_best_outputs.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Any, Literal

# Repo root: ai_engine/tests/ -> ai_engine -> parent = eubot
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ai_engine.tests.quality_filter import is_good_response  # noqa: E402
from ai_engine.tests.quality_semantics import semantic_score  # noqa: E402
from ai_engine.tests.test_live_safe import (  # noqa: E402
    MAX_QUESTIONS_CAP,
    MIN_DELAY_SEC,
    LiveRunParams,
    resolve_url,
    run_questions,
)

_TESTS_DIR = Path(__file__).resolve().parent
_DATASETS_DIR = _TESTS_DIR / "datasets"
_REPORT_PATH = _TESTS_DIR / "output_quality_report.json"
_HISTORY_PATH = _TESTS_DIR / "output_quality_history.json"
BEST_DATASET_PATH = _DATASETS_DIR / "_best_dataset.json"
BEST_OUTPUTS_PATH = _DATASETS_DIR / "_best_outputs.json"
TOP3_PATH = _DATASETS_DIR / "_top3_datasets.json"
IMPROVED_DATASET_PATH = _DATASETS_DIR / "_improved_dataset.json"
FALLBACK_DATASET_NAME = "convo_growth_set_02.json"


def _resolve_cli_dataset(name: str) -> Path:
    """Resolve a dataset path from CLI (filename or relative path)."""
    p = Path(name)
    if p.is_file():
        return p.resolve()
    for base in (_DATASETS_DIR, _TESTS_DIR, Path.cwd()):
        c = base / p.name
        if c.is_file():
            return c.resolve()
    return (_DATASETS_DIR / p.name).resolve()


def _try_load_dataset_list(path: Path) -> list[str] | None:
    """Return prompts or None if file invalid / empty."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw:
            out = [str(x).strip() for x in raw if str(x).strip()]
            return out if out else None
    except (json.JSONDecodeError, OSError, TypeError):
        pass
    return None


def enrich_rows_semantic(rows: list[dict[str, Any]], *, verbose: bool = False) -> None:
    """Set qlt_effective (fake HIGH → MID); semantic may already be set by run_questions."""
    for r in rows:
        if not r.get("ok"):
            r["semantic"] = 0.0
            r["qlt_effective"] = str(r.get("qlt", "LOW"))
            continue
        a = str(r.get("a", ""))
        sem = float(r.get("semantic", semantic_score(a)))
        r["semantic"] = sem
        qlt = str(r.get("qlt", "LOW"))
        if qlt == "HIGH" and sem < 0.4:
            r["qlt_effective"] = "MID"
        else:
            r["qlt_effective"] = qlt


def discover_datasets() -> list[Path]:
    if not _DATASETS_DIR.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(_DATASETS_DIR.glob("*.json")):
        if p.name.startswith("_"):
            continue
        out.append(p)
    return out


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("ok")]
    errors = sum(1 for r in rows if not r.get("ok"))
    filtered_ok = [r for r in ok_rows if is_good_response(r)]
    dist: dict[str, int] = {"LOW": 0, "MID": 0, "HIGH": 0}
    if not ok_rows:
        return {
            "avg_len": 0.0,
            "avg_latency": 0.0,
            "avg_semantic": 0.0,
            "quality_distribution": dist,
            "errors": errors,
            "novelty": 0.0,
            "duplicate_answers": 0,
            "score": 0.0,
            "filtered_count": 0,
            "style_penalty": 0,
        }
    for r in ok_rows:
        q = str(r.get("qlt_effective", r.get("qlt", "LOW")))
        if q in dist:
            dist[q] += 1
        else:
            dist["LOW"] += 1
    if filtered_ok:
        avg_len = sum(r.get("length", 0) for r in filtered_ok) / len(filtered_ok)
        avg_sem = sum(float(r.get("semantic", 0.0)) for r in filtered_ok) / len(filtered_ok)
    else:
        avg_len = 0.0
        avg_sem = 0.0
    avg_lat = sum(float(r.get("lat", 0)) for r in ok_rows) / len(ok_rows)
    low_c = int(dist.get("LOW", 0))
    answers = [str(r.get("a", "")).strip() for r in ok_rows]
    unique_answers = set(answers)
    n_ok = len(ok_rows)
    novelty = len(unique_answers) / max(n_ok, 1)
    dup_count = n_ok - len(unique_answers)
    style_penalty = sum(15 for r in ok_rows if "imagine" in str(r.get("a", "")).lower())
    score = (
        float(avg_len)
        + (novelty * 50.0)
        + (float(avg_sem) * 40.0)
        - (low_c * 20.0)
        - (dup_count * 10.0)
        - float(style_penalty)
    )
    return {
        "avg_len": round(avg_len, 4),
        "avg_latency": round(avg_lat, 4),
        "avg_semantic": round(avg_sem, 4),
        "quality_distribution": dist,
        "errors": errors,
        "novelty": round(novelty, 4),
        "duplicate_answers": int(dup_count),
        "score": round(score, 4),
        "filtered_count": len(filtered_ok),
        "style_penalty": int(style_penalty),
    }


def weak_dataset_warning(dist: dict[str, int], ok_count: int) -> bool:
    """True if >50% of successful responses are LOW quality (marked WEAK, excluded from ranking)."""
    if ok_count == 0:
        return True
    low = int(dist.get("LOW", 0))
    return (low / ok_count) > 0.5


def weak_semantic_hard(avg_semantic: float) -> bool:
    """TASK 6: avg_semantic < 0.4 → WEAK."""
    return float(avg_semantic) < 0.4


def strict_dataset_reject(stats: dict[str, Any]) -> bool:
    """TASK 15: LOW > 30% or avg_semantic < 0.5."""
    dist = stats.get("quality_distribution", {})
    n = sum(dist.values())
    if n == 0:
        return True
    low_pct = dist.get("LOW", 0) / n * 100
    if low_pct > 30.0:
        return True
    return float(stats.get("avg_semantic", 0.0)) < 0.5


def composite_score(stats: dict[str, Any]) -> float:
    return float(stats.get("score", 0.0))


def quality_targets_met(stats: dict[str, Any]) -> tuple[bool, str]:
    """
    Monitoring targets (not hard gates): HIGH >= 60% of successful answers,
    LOW <= 20%, avg_len > 100.
    """
    dist = stats.get("quality_distribution", {})
    n = sum(dist.values())
    if n == 0:
        return False, "no successful responses"
    high_pct = dist.get("HIGH", 0) / n * 100
    low_pct = dist.get("LOW", 0) / n * 100
    al = float(stats.get("avg_len", 0))
    issues: list[str] = []
    if high_pct < 60.0:
        issues.append(f"HIGH {high_pct:.1f}% < 60%")
    if low_pct > 20.0:
        issues.append(f"LOW {low_pct:.1f}% > 20%")
    if al <= 100.0:
        issues.append(f"avg_len {al:.1f} <= 100")
    if not issues:
        return True, "targets: HIGH>=60% LOW<=20% avg_len>100"
    return False, "; ".join(issues)


def print_dataset_block(name: str, stats: dict[str, Any], *, weak: bool) -> None:
    dist = stats["quality_distribution"]
    print(f"DATASET: {name}", flush=True)
    if weak:
        print("  MARKED: WEAK (excluded from ranking)", flush=True)
    print(f"AVG_LEN: {stats['avg_len']}", flush=True)
    print(f"AVG_SEMANTIC: {stats.get('avg_semantic', 0)}", flush=True)
    print(f"AVG_LAT: {stats['avg_latency']}s", flush=True)
    print(f"SCORE: {stats.get('score', 0)}", flush=True)
    print(f"NOVELTY: {stats.get('novelty', 0)} (unique answers / total)", flush=True)
    print(f"DUP_ANSWERS: {stats.get('duplicate_answers', 0)}", flush=True)
    print(
        f"QLT: HIGH {dist.get('HIGH', 0)} | MID {dist.get('MID', 0)} | LOW {dist.get('LOW', 0)}",
        flush=True,
    )
    print(f"ERRORS: {stats['errors']}", flush=True)
    print("", flush=True)


def _read_history_list() -> list[Any]:
    if not _HISTORY_PATH.is_file():
        return []
    try:
        raw = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def append_history_entry(entry: dict[str, Any]) -> None:
    """Append one record to output_quality_history.json."""
    history = _read_history_list()
    history.append(entry)
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def append_mutation_history(
    prev_score: float,
    new_score: float,
    best_dataset: str | None = None,
    *,
    prev_avg_semantic: float | None = None,
    new_avg_semantic: float | None = None,
) -> None:
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "mutation",
        "prev_score": float(prev_score),
        "new_score": float(new_score),
        "best_dataset": best_dataset,
    }
    if prev_avg_semantic is not None:
        entry["prev_avg_semantic"] = round(float(prev_avg_semantic), 4)
    if new_avg_semantic is not None:
        entry["new_avg_semantic"] = round(float(new_avg_semantic), 4)
    append_history_entry(entry)


def _append_history(
    best_name: str | None,
    best_score: float,
    avg_semantic: float | None = None,
) -> None:
    prev_score: float | None = None
    prev_sem: float | None = None
    history = _read_history_list()
    if history:
        last = history[-1]
        if isinstance(last, dict) and "score" in last:
            try:
                prev_score = float(last["score"])
            except (TypeError, ValueError):
                pass
        if isinstance(last, dict) and last.get("avg_semantic") is not None:
            try:
                prev_sem = float(last["avg_semantic"])
            except (TypeError, ValueError):
                pass

    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_dataset": best_name,
        "score": float(best_score),
    }
    if avg_semantic is not None:
        entry["avg_semantic"] = round(float(avg_semantic), 4)
    append_history_entry(entry)

    if prev_score is not None:
        if best_score > prev_score:
            print("QUALITY IMPROVED", flush=True)
        elif best_score < prev_score:
            print("QUALITY DROP", flush=True)
    if (
        prev_score is not None
        and prev_sem is not None
        and avg_semantic is not None
        and best_score > prev_score
        and avg_semantic < prev_sem
    ):
        print("FAKE IMPROVEMENT DETECTED", flush=True)
    print("", flush=True)


def best_outputs_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        if not r.get("ok"):
            continue
        a = str(r.get("a", ""))
        out.append(
            {
                "q": str(r.get("q", "")),
                "a": a,
                "qlt": str(r.get("qlt", "LOW")),
                "len": int(r.get("length", len(a))),
                "semantic": round(float(r.get("semantic", 0.0)), 4),
            }
        )
    return out


def best_pattern_from_rows(rows: list[dict[str, Any]]) -> str | None:
    """Most common ~16-char prefix among effective-HIGH prompts."""
    qs: list[str] = []
    for r in rows:
        if not r.get("ok"):
            continue
        if str(r.get("qlt_effective", r.get("qlt"))) != "HIGH":
            continue
        q = str(r.get("q", "")).strip()
        if q:
            qs.append(q)
    if not qs:
        return None
    prefixes = [q[:16] for q in qs if len(q) >= 8]
    if not prefixes:
        return None
    return Counter(prefixes).most_common(1)[0][0]


def run_batch(
    dataset_paths: list[Path],
    *,
    mode: Literal["v1", "v2"],
    url: str | None,
    model: str | None,
    delay: float,
    max_q: int,
    timeout: float,
    verbose: bool = False,
    strict: bool = False,
    merge_report: dict[str, dict[str, Any]] | None = None,
    write_history: bool = True,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """
    Run batch evaluation. Returns report, best_name, best_score, prompts_by_stem, rows_by_stem.
    merge_report: start from a prior report dict (e.g. merge improved run into same JSON).
    write_history: set False for auxiliary runs (mutation pass); pipeline may append mutation entry.
    write_artifacts: set False for auxiliary runs that should not overwrite best/top3/outputs.
    """
    final_url = resolve_url(mode, url)
    delay_use = max(MIN_DELAY_SEC, float(delay))
    cap = max(1, min(MAX_QUESTIONS_CAP, int(max_q)))
    timeout_use = max(1.0, float(timeout))

    report: dict[str, dict[str, Any]] = dict(merge_report) if merge_report else {}
    prompts_by_stem: dict[str, list[str]] = {}
    rows_by_stem: dict[str, list[dict[str, Any]]] = {}
    # (stem, stats, score, avg_semantic, low_count) — tie-break: score, avg_semantic, LOW asc
    ranking_eligible: list[tuple[str, dict[str, Any], float, float, int]] = []

    for path in dataset_paths:
        stem = path.stem
        prompts = _try_load_dataset_list(path)
        if prompts is None:
            print(f"SKIP (invalid or empty): {path}", file=sys.stderr, flush=True)
            continue

        items = prompts[:cap]
        prompts_by_stem[stem] = items
        params = LiveRunParams(
            mode=mode,
            url=final_url,
            model=model,
            delay_sec=delay_use,
            timeout=timeout_use,
            verbose=verbose,
            write_log=False,
        )
        rows = run_questions(items, params)
        enrich_rows_semantic(rows, verbose=verbose)
        rows_by_stem[stem] = rows
        stats = aggregate_metrics(rows)
        ok_count = sum(1 for r in rows if r.get("ok"))
        low_c = int(stats["quality_distribution"].get("LOW", 0))
        weak = (
            weak_dataset_warning(stats["quality_distribution"], ok_count)
            or weak_semantic_hard(stats["avg_semantic"])
            or (strict and strict_dataset_reject(stats))
        )

        report[stem] = {
            "avg_len": stats["avg_len"],
            "avg_latency": stats["avg_latency"],
            "avg_semantic": stats["avg_semantic"],
            "quality_distribution": stats["quality_distribution"],
            "errors": stats["errors"],
            "novelty": stats["novelty"],
            "duplicate_answers": stats["duplicate_answers"],
            "score": stats["score"],
            "filtered_count": stats.get("filtered_count", 0),
            "style_penalty": stats.get("style_penalty", 0),
            "weak": weak,
        }

        if weak:
            print(f"WARNING_DATASET_WEAK | {stem}", flush=True)

        print_dataset_block(stem, report[stem], weak=weak)

        if not weak:
            ranking_eligible.append(
                (
                    stem,
                    report[stem],
                    composite_score(report[stem]),
                    float(stats["avg_semantic"]),
                    low_c,
                )
            )

    ranking_eligible.sort(key=lambda x: (-x[2], -x[3], x[4]))

    best_name: str | None = None
    best_score = 0.0
    best_prompts: list[str] | None = None
    best_rows: list[dict[str, Any]] | None = None

    if ranking_eligible:
        best_name, best_stats, best_score, best_asem, best_low = ranking_eligible[0]
        best_prompts = prompts_by_stem.get(best_name)
        best_rows = rows_by_stem.get(best_name)
        print("BEST DATASET (rank: score, then avg_semantic, then LOW count; WEAK excluded):", flush=True)
        for rank, (name, _, sc, asem, lc) in enumerate(ranking_eligible[:3], start=1):
            print(f"  {rank}. {name} (score={sc}, avg_semantic={asem}, LOW={lc})", flush=True)
        if len(ranking_eligible) > 3:
            print("  ...", flush=True)
    else:
        print("All datasets marked WEAK — fallback to", FALLBACK_DATASET_NAME, flush=True)
        fb_path = _DATASETS_DIR / FALLBACK_DATASET_NAME
        if not fb_path.is_file():
            for base in (_TESTS_DIR, Path.cwd()):
                alt = base / "datasets" / FALLBACK_DATASET_NAME
                if alt.is_file():
                    fb_path = alt
                    break
        if fb_path.is_file():
            fb_prompts = _try_load_dataset_list(fb_path)
            if fb_prompts:
                best_name = fb_path.stem
                best_prompts = fb_prompts[:cap]
                best_score = 0.0
    print("", flush=True)

    top3_names = [name for name, _, _ in ranking_eligible[:3]]
    if not top3_names and best_name:
        top3_names = [best_name]

    _DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if write_artifacts and best_prompts is not None and best_name is not None:
        with open(BEST_DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(best_prompts, f, ensure_ascii=False, indent=2)
        print(f"Saved best prompts -> {BEST_DATASET_PATH}", flush=True)
        if best_rows is not None:
            with open(BEST_OUTPUTS_PATH, "w", encoding="utf-8") as f:
                json.dump(best_outputs_from_rows(best_rows), f, ensure_ascii=False, indent=2)
            print(f"Saved best Q/A -> {BEST_OUTPUTS_PATH}", flush=True)

    if write_artifacts:
        with open(TOP3_PATH, "w", encoding="utf-8") as f:
            json.dump(top3_names, f, indent=2, ensure_ascii=False)
        print(f"Saved top-3 names -> {TOP3_PATH}", flush=True)

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved: {_REPORT_PATH}", flush=True)

    if write_history:
        _append_history(best_name, best_score if best_name else 0.0)

    return {
        "report": report,
        "best_name": best_name,
        "best_score": best_score,
        "prompts_by_stem": prompts_by_stem,
        "rows_by_stem": rows_by_stem,
        "quality_targets_met": quality_targets_met(report[best_name])[0] if best_name and best_name in report else False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Batch quality evaluation (sequential, no parallelism).")
    p.add_argument("--mode", choices=["v1", "v2"], default="v1")
    p.add_argument("--url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--delay", type=float, default=MIN_DELAY_SEC)
    p.add_argument("--max", type=int, default=10, dest="max_q")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--verbose", action="store_true", help="Print each Q/A to stdout (no parallel)")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Mark WEAK if LOW>30%% or avg_semantic<0.5 (stricter than default)",
    )
    p.add_argument(
        "datasets",
        nargs="*",
        help="Optional dataset filenames (under tests/datasets/). If empty, all *.json are used.",
    )
    args = p.parse_args()

    if args.datasets:
        paths = [_resolve_cli_dataset(name) for name in args.datasets]
    else:
        paths = discover_datasets()

    if not paths:
        print("No datasets found.", file=sys.stderr, flush=True)
        raise SystemExit(1)

    model = (args.model or "").strip() or None
    run_batch(
        paths,
        mode=args.mode,
        url=args.url,
        model=model,
        delay=args.delay,
        max_q=args.max_q,
        timeout=args.timeout,
        verbose=args.verbose,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
