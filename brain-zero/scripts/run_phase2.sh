#!/bin/bash
set -e

# Phase 2: Small model (125M), large dataset, 50k steps
# Run from brain-zero/ directory on RunPod

echo "=== PHASE 2: TRAINING MASSIVO ==="
echo ""

# 1. Download data (WikiText + OpenWebText ~200k docs)
echo "[1/5] Downloading data (WikiText + OpenWebText) ..."
python scripts/download_data.py --mode large

# 2. Build JSONL dataset
echo "[2/5] Building JSONL dataset ..."
python scripts/build_dataset.py

# 3. Train tokenizer with larger vocab
echo "[3/5] Training tokenizer (vocab 16384) ..."
python scripts/train_tokenizer.py --vocab_size 16384

# 4. Train model (125M params, 50k steps)
echo "[4/5] Training model (small, 125M params, 50k steps) ..."
echo "       This will take ~1-3 hours depending on GPU."
echo "       Checkpoints saved every 5000 steps."
echo ""
python scripts/train.py --profile small --training_config training_phase2.yaml

# 5. Test
echo "[5/5] Generating sample text ..."
LAST_CKPT=$(ls -d models/checkpoints/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -n "$LAST_CKPT" ]; then
    python scripts/inference.py --checkpoint "$LAST_CKPT" --prompt "The history of"
    echo ""
    echo "=== DONE. Launch chat: python scripts/chat.py ==="
else
    echo "No checkpoint found!"
fi
