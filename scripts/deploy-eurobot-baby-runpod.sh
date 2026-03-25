#!/usr/bin/env bash
# Eurobot Baby — full pipeline on RunPod (eubot pod).
# Run ON THE POD as root. Repo: https://github.com/socialengaged/eurobot_baby
set -euo pipefail

export HF_HOME="${HF_HOME:-/root/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/hf_home}"
mkdir -p "$HF_HOME"

WORKDIR="${WORKDIR:-/workspace/eurobot_baby}"
REPO="${REPO:-https://github.com/socialengaged/eurobot_baby.git}"

if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone "$REPO" "$WORKDIR"
fi
cd "$WORKDIR"

VENV="${VENV:-/root/eurobot_baby_venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
export TMPDIR="${TMPDIR:-/tmp}"
"$VENV/bin/pip" install --no-cache-dir -U pip
"$VENV/bin/pip" install --no-cache-dir -r requirements.txt

PY="$VENV/bin/python"

"$PY" scripts/download_data.py
"$PY" scripts/download_classics.py
"$PY" scripts/download_sacred.py

"$PY" scripts/build_dataset.py
"$PY" scripts/train_tokenizer.py
"$PY" scripts/train.py

echo "Done. Smoke test (after checkpoints exist):"
echo "  $PY scripts/inference.py"
echo "  $PY scripts/chat.py"
