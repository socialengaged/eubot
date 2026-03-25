#!/usr/bin/env bash
# Wrapper: scraping + merge mentre il training gira sulla GPU — merge verso raw_staging di default
# (non sovrascrive data/raw usato come riferimento finché non fai promote_raw_staging.sh).
#
# Uso (pod):
#   nohup bash run_scraping_safe.sh >> /workspace/eurobot_scraping_run/scraping_safe_outer.log 2>&1 &
#
# Env:
#   EUROBOT_MERGE_OUT   default: /workspace/eurobot_baby/data/raw_staging
#   EUROBOT_SCRAPING_RUN  base output (default /workspace/eurobot_scraping_run)
#   HF_HOME / HF_HUB_CACHE
#   LOG (file log per run_stem_pipeline interno; default sotto WORK)
set -u
export HF_HOME="${HF_HOME:-/root/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
mkdir -p "$HF_HOME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EUROBOT_MERGE_OUT="${EUROBOT_MERGE_OUT:-/workspace/eurobot_baby/data/raw_staging}"
mkdir -p "$EUROBOT_MERGE_OUT"

OUT_LOG="${SCRAPING_SAFE_LOG:-/workspace/eurobot_scraping_run/scraping_safe.log}"
mkdir -p "$(dirname "$OUT_LOG")"
export LOG="$OUT_LOG"
{
  echo "[scraping_safe] start $(date -u)"
  echo "[scraping_safe] EUROBOT_MERGE_OUT=$EUROBOT_MERGE_OUT EUROBOT_SCRAPING_RUN=${EUROBOT_SCRAPING_RUN:-/workspace/eurobot_scraping_run}"
} >> "$LOG"

bash "$SCRIPT_DIR/run_stem_pipeline.sh"

echo "[scraping_safe] done $(date -u)" >> "$LOG"
