#!/usr/bin/env bash
# Dual mode: training tiene la GPU; serve usa CPU (stesso processo non compete per VRAM).
# Per forzare GPU sullo stesso device del train: sconsigliato (OOM); export CUDA_VISIBLE_DEVICES=0 solo se accetti il rischio.
set -e

export CUDA_VISIBLE_DEVICES=""
unset TORCH_DEVICE 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export SERVE_MAX_CONCURRENT_CHAT="${SERVE_MAX_CONCURRENT_CHAT:-1}"
# CPU sotto carico train: timeout chat più lungo (default server 120s)
export SERVE_CHAT_REQUEST_TIMEOUT_SEC="${SERVE_CHAT_REQUEST_TIMEOUT_SEC:-300}"

cd /workspace/eurobot_baby

if [ ! -d "models/checkpoints/serve_checkpoint" ]; then
  echo "ERROR: models/checkpoints/serve_checkpoint missing — cp -r models/checkpoints/step_918672 models/checkpoints/serve_checkpoint"
  exit 1
fi

nohup /root/eurobot_baby_venv/bin/python -u scripts/serve.py \
  --checkpoint models/checkpoints/serve_checkpoint \
  --host 0.0.0.0 \
  --port 8080 \
  --safe-mode \
  >>/root/serve.log 2>&1 &

echo $! >/root/serve.pid
echo "[OK] serve started pid=$(cat /root/serve.pid) (CPU, CUDA_VISIBLE_DEVICES empty)"
