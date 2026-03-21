#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Eubot Coding Assistant pipeline ==="
pip install -r requirements.txt

echo "[1/4] prepare_data.py"
python scripts/prepare_data.py

echo "[2/4] finetune.py (long run)"
python scripts/finetune.py

echo "[3/4] merge_adapter.py"
python scripts/merge_adapter.py

echo "[4/4] Done. Chat: python scripts/chat.py"
echo "        API:  python scripts/serve.py --host 0.0.0.0 --port 8080"
