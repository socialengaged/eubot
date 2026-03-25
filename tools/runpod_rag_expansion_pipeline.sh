#!/usr/bin/env bash
# Run from monorepo root on RunPod (or dev) after: pip install -r requirements-ai-engine.txt
# 1) Download HF shards to ai_engine/data/rag_expansion + expansion_training
# 2) Merge JSONL
# 3) Build FAISS under eurobot_baby/vector_db/rag_expansion (needs OPENAI_API_KEY)
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
export HF_HOME="${HF_HOME:-/root/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARN: OPENAI_API_KEY not set — ingest will fail until set." >&2
fi

python eurobot_baby/tools/scraping/download_rag_expansion_hf.py --max-per-dataset "${MAX_PER_DATASET:-5000}"

python -m ai_engine.rag.ingest_multidomain_jsonl \
  --jsonl ai_engine/data/rag_expansion/combined_rag.jsonl \
  --out-dir eurobot_baby/vector_db/rag_expansion \
  --max-words 300 --overlap 30 --batch-size 64

echo "Done. Set on serve:"
echo "  export EUROBOT_RAG_EXPANSION_PATH=$ROOT/eurobot_baby/vector_db/rag_expansion"
