#!/usr/bin/env bash
# Fase 1 — Gutenberg esoterico (CPU/rete). Può girare in parallelo al training; non lanciare due istanze.
set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
WORK="${EUROBOT_SCRAPING_RUN:-/workspace/eurobot_scraping_run}"
export EUROBOT_GUTENBERG_OUT="${EUROBOT_GUTENBERG_OUT:-$WORK/output/gutenberg_esoteric}"
mkdir -p "$EUROBOT_GUTENBERG_OUT"
LOG="${LOG:-$WORK/pipeline_gutenberg_esoteric.log}"
mkdir -p "$(dirname "$LOG")"
PY="${PY:-/root/eurobot_baby_venv/bin/python}"
echo "[gutenberg_esoteric] start $(date -u)" >> "$LOG"
$PY download_gutenberg_esoteric.py >> "$LOG" 2>&1
echo "[gutenberg_esoteric] done $(date -u)" >> "$LOG"
date -u > "$WORK/gutenberg_esoteric.done"
