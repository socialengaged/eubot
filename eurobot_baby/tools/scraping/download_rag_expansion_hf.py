#!/usr/bin/env python3
"""
Download open / Hub datasets for (1) RAG JSONL with topic+source and (2) optional LM training JSONL.

Outputs (under repo ai_engine/data/):
  rag_expansion/*.jsonl          — one file per source; merge with merge_rag_expansion_jsonl.py
  rag_expansion/combined_rag.jsonl — optional: run merger
  expansion_training/lm_train.jsonl — {"text": "User: ...\\nAssistant: ..."} for future merge_datasets

Requires: pip install datasets (see requirements-ai-engine.txt)

Licenses vary by dataset — verify before production use (see docs/DATASET_EXPANSION_RAG_AND_TRAINING.md).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

EUROBOT_BABY = Path(__file__).resolve().parents[2]
REPO_ROOT = EUROBOT_BABY.parent
RAG_OUT = Path(os.environ.get("RAG_EXPANSION_OUT", str(REPO_ROOT / "ai_engine" / "data" / "rag_expansion")))
TRAIN_OUT = Path(os.environ.get("EXPANSION_TRAIN_OUT", str(REPO_ROOT / "ai_engine" / "data" / "expansion_training")))


def _write_rag_line(f, text: str, source: str, topic: str) -> None:
    text = (text or "").strip()
    if len(text) < 40:
        return
    f.write(
        json.dumps({"text": text, "source": source, "topic": topic}, ensure_ascii=False) + "\n"
    )


def _write_lm_train(f, user: str, assistant: str) -> None:
    u = (user or "").strip()
    a = (assistant or "").strip()
    if len(u) < 5 or len(a) < 10:
        return
    t = f"User: {u}\nAssistant: {a}"
    f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def export_oasst1(max_rows: int, rng: random.Random) -> tuple[int, int]:
    from datasets import load_dataset

    print("Loading OpenAssistant/oasst1 ...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    path = RAG_OUT / "oasst1.jsonl"
    path_lm = TRAIN_OUT / "oasst1_lm.jsonl"
    n_rag = n_lm = 0
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    with open(path, "w", encoding="utf-8") as fr, open(path_lm, "w", encoding="utf-8") as fl:
        for i in indices:
            if n_rag >= max_rows and n_lm >= max_rows:
                break
            row = ds[i]
            text = str(row.get("text") or "").strip()
            role = str(row.get("role") or "")
            if len(text) < 80:
                continue
            if n_rag < max_rows:
                _write_rag_line(fr, text, "OpenAssistant/oasst1", "conversational")
                n_rag += 1
            if n_lm < max_rows:
                _write_lm_train(fl, f"Role={role}. Expand or answer clearly.", text[:1500])
                n_lm += 1
    print(f"  oasst1 rag lines ~{n_rag} -> {path}")
    return n_rag, n_lm


def export_dolly(max_rows: int) -> int:
    from datasets import load_dataset

    print("Loading databricks/databricks-dolly-15k ...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    path = RAG_OUT / "dolly.jsonl"
    path_lm = TRAIN_OUT / "dolly_lm.jsonl"
    n = 0
    with open(path, "w", encoding="utf-8") as fr, open(path_lm, "w", encoding="utf-8") as fl:
        for i, row in enumerate(ds):
            if n >= max_rows:
                break
            instr = str(row.get("instruction") or "").strip()
            inp = str(row.get("input") or "").strip()
            out = str(row.get("response") or "").strip()
            if not out:
                continue
            ctx = f"{instr}\n{inp}".strip() if inp else instr
            block = f"Instruction: {ctx}\nResponse: {out}"
            _write_rag_line(fr, block, "databricks-dolly-15k", "conversational_instruction")
            _write_lm_train(fl, ctx[:1200], out[:2000])
            n += 1
    print(f"  dolly lines {n} -> {path}")
    return n


def export_medmcqa(max_rows: int) -> int:
    from datasets import load_dataset

    print("Loading medmcqa ...")
    try:
        ds = load_dataset("medmcqa", "default", split="train", trust_remote_code=True)
    except TypeError:
        ds = load_dataset("medmcqa", "default", split="train")
    path = RAG_OUT / "medmcqa.jsonl"
    path_lm = TRAIN_OUT / "medmcqa_lm.jsonl"
    n = 0
    with open(path, "w", encoding="utf-8") as fr, open(path_lm, "w", encoding="utf-8") as fl:
        for i, row in enumerate(ds):
            if n >= max_rows:
                break
            q = str(row.get("question") or "")
            opa, opb, opc, opd = (
                str(row.get("opa") or ""),
                str(row.get("opb") or ""),
                str(row.get("opc") or ""),
                str(row.get("opd") or ""),
            )
            cop = int(row.get("cop", 0))
            sub = str(row.get("subject_name") or row.get("topic_name") or row.get("subject") or "unknown").lower()
            topic = "medicine_general"
            if "anatom" in sub:
                topic = "anatomy"
            elif "pharma" in sub or "medicine" in sub:
                topic = "medicine_general"
            opts = f"A) {opa}\nB) {opb}\nC) {opc}\nD) {opd}"
            letter = ("A", "B", "C", "D")[cop] if 0 <= cop <= 3 else "?"
            ans = (opa, opb, opc, opd)[cop] if 0 <= cop <= 3 else ""
            block = f"Question: {q}\n{opts}\nCorrect: {letter}) {ans}"
            _write_rag_line(fr, block, "medmcqa", topic)
            _write_lm_train(
                fl,
                f"{q}\n{opts}",
                f"The best answer is {letter}: {ans}",
            )
            n += 1
    print(f"  medmcqa lines {n} -> {path}")
    return n


def export_pubmed_qa(max_rows: int) -> int:
    from datasets import load_dataset

    print("Loading pubmed_qa (pqa_labeled) ...")
    try:
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    except Exception as e:
        print(f"  SKIP pubmed_qa: {e}")
        return 0
    path = RAG_OUT / "pubmed_qa.jsonl"
    path_lm = TRAIN_OUT / "pubmed_qa_lm.jsonl"
    n = 0
    with open(path, "w", encoding="utf-8") as fr, open(path_lm, "w", encoding="utf-8") as fl:
        for i, row in enumerate(ds):
            if n >= max_rows:
                break
            ctx = str(row.get("context") or "").strip()
            q = str(row.get("question") or "").strip()
            la = row.get("long_answer")
            long_a = la.strip() if isinstance(la, str) else str(la or "").strip()
            if not long_a:
                continue
            block = f"Context: {ctx[:4000]}\nQuestion: {q}\nAnswer: {long_a}"
            _write_rag_line(fr, block, "pubmed_qa", "medicine_literature")
            _write_lm_train(fl, f"{q}\n\nContext: {ctx[:2000]}", long_a[:2000])
            n += 1
    print(f"  pubmed_qa lines {n} -> {path}")
    return n


def export_daily_dialog(max_rows: int, rng: random.Random) -> int:
    """Lightweight conversational turns (psychology / daily)."""
    from datasets import load_dataset

    print("Loading daily_dialog ...")
    try:
        ds = load_dataset("daily_dialog", split="train")
    except Exception as e:
        print(f"  SKIP daily_dialog: {e}")
        return 0
    path = RAG_OUT / "daily_dialog.jsonl"
    path_lm = TRAIN_OUT / "daily_dialog_lm.jsonl"
    n = 0
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    with open(path, "w", encoding="utf-8") as fr, open(path_lm, "w", encoding="utf-8") as fl:
        for i in idxs:
            if n >= max_rows:
                break
            row = ds[i]
            turns = row.get("dialog")
            if not turns:
                continue
            if isinstance(turns, list) and turns and isinstance(turns[0], list):
                flat: list[str] = []
                for t in turns:
                    flat.extend(str(x) for x in t)
                text = "\n".join(flat)
            elif isinstance(turns, list):
                text = "\n".join(str(t) for t in turns)
            else:
                text = str(turns)
            if len(text) < 50:
                continue
            _write_rag_line(fr, text, "daily_dialog", "psychology_conversational")
            parts = text.split("\n")
            if len(parts) >= 2:
                _write_lm_train(fl, parts[0][:500], "\n".join(parts[1:])[:1500])
            n += 1
    print(f"  daily_dialog lines {n} -> {path}")
    return n


def merge_rag_jsonl_files(pattern: str = "*.jsonl", out_name: str = "combined_rag.jsonl") -> Path:
    out = RAG_OUT / out_name
    n = 0
    with open(out, "w", encoding="utf-8") as fout:
        for p in sorted(RAG_OUT.glob(pattern)):
            if p.name == out_name or p.name.startswith("combined"):
                continue
            with open(p, encoding="utf-8", errors="replace") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        fout.write(line + "\n")
                        n += 1
    print(f"Merged {n} lines -> {out}")
    return out


def merge_lm_train(out_name: str = "lm_train_merged.jsonl") -> Path:
    out = TRAIN_OUT / out_name
    n = 0
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fout:
        for p in sorted(TRAIN_OUT.glob("*_lm.jsonl")):
            with open(p, encoding="utf-8", errors="replace") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        fout.write(line + "\n")
                        n += 1
    print(f"Merged LM lines {n} -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-dataset", type=int, default=5000, help="Cap rows per dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-oasst", action="store_true")
    ap.add_argument("--skip-dolly", action="store_true")
    ap.add_argument("--skip-med", action="store_true")
    ap.add_argument("--skip-pubmed", action="store_true")
    ap.add_argument("--skip-daily", action="store_true")
    ap.add_argument("--merge-only", action="store_true", help="Only merge existing rag jsonl")
    args = ap.parse_args()

    RAG_OUT.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    m = args.max_per_dataset

    if args.merge_only:
        merge_rag_jsonl_files()
        merge_lm_train()
        return 0

    if not args.skip_oasst:
        try:
            export_oasst1(m, rng)
        except Exception as e:
            print(f"SKIP oasst1: {e}", file=sys.stderr)

    if not args.skip_dolly:
        try:
            export_dolly(m)
        except Exception as e:
            print(f"SKIP dolly: {e}", file=sys.stderr)

    if not args.skip_med:
        try:
            export_medmcqa(m)
        except Exception as e:
            print(f"SKIP medmcqa: {e}", file=sys.stderr)

    if not args.skip_pubmed:
        try:
            export_pubmed_qa(m // 2)
        except Exception as e:
            print(f"SKIP pubmed: {e}", file=sys.stderr)

    if not args.skip_daily:
        try:
            export_daily_dialog(m // 3, rng)
        except Exception as e:
            print(f"SKIP daily_dialog: {e}", file=sys.stderr)

    merge_rag_jsonl_files()
    merge_lm_train()
    print(f"Done. RAG dir: {RAG_OUT}\nTraining LM dir: {TRAIN_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
