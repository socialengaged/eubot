#!/usr/bin/env bash
# Sequential: sacred-texts deep -> gnosis retry. Run with nohup on scraper/pod.
# Non lanciare in parallelo a run_stem_pipeline.sh se entrambi saturano rete/disco; OK in parallelo al training GPU.
set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
WORK="${EUROBOT_SCRAPING_RUN:-/workspace/eurobot_scraping_run}"
export EUROBOT_SACRED_OUT="$WORK/output/sacred_texts"
mkdir -p "$EUROBOT_SACRED_OUT"
LOG="${LOG:-$WORK/pipeline_esoteric.log}"
mkdir -p "$(dirname "$LOG")"
PY="${PY:-/root/eurobot_baby_venv/bin/python}"
echo "[esoteric] start sacred $(date -u)" >> "$LOG"
$PY download_sacred_texts_deep.py >> "$LOG" 2>&1
export EUROBOT_RAW_DIR="$WORK/output/gnosis_retry"
mkdir -p "$EUROBOT_RAW_DIR"
echo "[esoteric] start gnosis $(date -u)" >> "$LOG"
$PY download_gnosis_retry.py --delay 5 >> "$LOG" 2>&1
echo "[esoteric] done $(date -u)" >> "$LOG"
date -u > "$WORK/esoteric_pipeline.done"
