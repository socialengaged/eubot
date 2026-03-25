#!/usr/bin/env bash
# HF + Gutenberg STEM -> merge into eurobot_baby data raw (or staging). Run with nohup.
# Non usare "set -e": alcuni step HF possono terminare con abort/cleanup difettoso senza invalidare gli output.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WORK="${EUROBOT_SCRAPING_RUN:-/workspace/eurobot_scraping_run}"
MERGE_OUT="${EUROBOT_MERGE_OUT:-/workspace/eurobot_baby/data/raw}"

export EUROBOT_PHYSICS_OUT="$WORK/output/physics"
export EUROBOT_MATH_OUT="$WORK/output/math"
export EUROBOT_ARXIV_OUT="$WORK/output/arxiv"
export EUROBOT_ASTRO_OUT="$WORK/output/astro"
export EUROBOT_SCIENCE_OUT="$WORK/output/gutenberg_science"
export EUROBOT_GUTENBERG_THEOLOGY_OUT="$WORK/output/gutenberg_theology"
export EUROBOT_GUTENBERG_COMPUTING_OUT="$WORK/output/gutenberg_computing"
# Caps ridotti rispetto al piano pieno per completare in tempi ragionevoli (aumentare in produzione).
export EUROBOT_OPENWEBMATH_MAX_DOCS="${EUROBOT_OPENWEBMATH_MAX_DOCS:-10000}"
export EUROBOT_ARXIV_MAX="${EUROBOT_ARXIV_MAX:-8000}"
# arXiv cs: più righe per rinforzare sviluppo / informatica (override con EUROBOT_ARXIV_CS_MAX).
export EUROBOT_ARXIV_CS_MAX="${EUROBOT_ARXIV_CS_MAX:-12000}"
mkdir -p "$EUROBOT_PHYSICS_OUT" "$EUROBOT_MATH_OUT" "$EUROBOT_ARXIV_OUT" "$EUROBOT_ASTRO_OUT" "$EUROBOT_SCIENCE_OUT" "$EUROBOT_GUTENBERG_THEOLOGY_OUT" "$EUROBOT_GUTENBERG_COMPUTING_OUT"
mkdir -p "$MERGE_OUT"

PY="${PY:-/root/eurobot_baby_venv/bin/python}"
LOG="${LOG:-$WORK/pipeline_stem.log}"
mkdir -p "$(dirname "$LOG")"
echo "[stem] start $(date -u) WORK=$WORK MERGE_OUT=$MERGE_OUT" >> "$LOG"
$PY download_physics_hf.py >> "$LOG" 2>&1
EUROBOT_ARXIV_MODE=physics $PY download_arxiv_stem.py >> "$LOG" 2>&1
EUROBOT_ARXIV_MODE=math $PY download_arxiv_stem.py >> "$LOG" 2>&1
EUROBOT_ARXIV_MODE=astro $PY download_arxiv_stem.py >> "$LOG" 2>&1
EUROBOT_ARXIV_MODE=cs EUROBOT_ARXIV_MAX="$EUROBOT_ARXIV_CS_MAX" $PY download_arxiv_stem.py >> "$LOG" 2>&1
$PY download_math_hf.py >> "$LOG" 2>&1
$PY download_astro_hf.py >> "$LOG" 2>&1
$PY download_stem_gutenberg_science.py >> "$LOG" 2>&1
$PY download_theology_gutenberg.py >> "$LOG" 2>&1
$PY download_gutenberg_computing.py >> "$LOG" 2>&1
$PY merge_outputs_to_raw_names.py --base "$WORK" --out "$MERGE_OUT" >> "$LOG" 2>&1
echo "[stem] done $(date -u)" >> "$LOG"
date -u > "$WORK/stem_pipeline.done"
