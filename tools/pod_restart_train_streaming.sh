#!/usr/bin/env bash
set -euo pipefail
for p in $(pgrep -f "eurobot_baby_venv/bin/python -u scripts/train.py" 2>/dev/null || true); do
  kill -9 "$p" 2>/dev/null || true
done
sleep 2
cd /workspace/eurobot_baby
: >/root/train_loop.log
export PYTHONUNBUFFERED=1
nohup /root/eurobot_baby_venv/bin/python -u scripts/train.py \
  --resume models/checkpoints/step_918672 \
  >>/root/train_loop.log 2>&1 &
echo "pid=$!"
sleep 15
nvidia-smi
echo "---LOG---"
tail -n 40 /root/train_loop.log
