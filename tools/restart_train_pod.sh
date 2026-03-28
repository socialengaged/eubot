#!/usr/bin/env bash
# Esegui sul pod: bash restart_train_pod.sh (dopo deploy train.py + training.yaml)
set -euo pipefail
for p in $(pgrep -f "eurobot_baby_venv/bin/python -u scripts/train.py" 2>/dev/null || true); do
  kill "$p" 2>/dev/null || true
done
sleep 2
cd /workspace/eurobot_baby
LATEST="$(ls -d models/checkpoints/step_* 2>/dev/null | sort -V | tail -1)"
echo "Resume: $LATEST"
: >/root/train_loop.log
export PYTHONUNBUFFERED=1
nohup /root/eurobot_baby_venv/bin/python -u scripts/train.py --resume "$LATEST" >>/root/train_loop.log 2>&1 &
echo $! >/root/train_loop.pid
echo "train pid $(cat /root/train_loop.pid)"
sleep 15
nvidia-smi
echo "--- log ---"
tail -n 35 /root/train_loop.log
echo "--- cuda probe ---"
/root/eurobot_baby_venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
