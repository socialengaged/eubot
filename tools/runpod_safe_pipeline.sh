#!/usr/bin/env bash
# Pipeline pod-safe: nessun pkill -f (evita match sulla sessione SSH).
set -euo pipefail

REPO="/workspace/eurobot_baby"
EUBOT="/workspace/eubot"
LOG="/root/train_loop.log"

echo "[STEP] safe stop"

# stop train (pgrep + kill PID, no pkill)
for pid in $(pgrep -f "eurobot_baby_venv/bin/python -u scripts/train.py" 2>/dev/null || true); do
  [ -n "${pid}" ] && kill -9 "${pid}" 2>/dev/null || true
  echo "[OK] train stopped pid=${pid}"
done

# stop serve via porta (NO pkill)
fuser -k 8080/tcp 2>/dev/null || true

# stop orchestrator: path assoluto nel pattern, pgrep + kill (no pkill)
for pid in $(pgrep -f "/workspace/eubot/orchestrator/orchestrator.py" 2>/dev/null || true); do
  [ -n "${pid}" ] && kill -9 "${pid}" 2>/dev/null || true
  echo "[OK] orchestrator stopped pid=${pid}"
done

sleep 2

echo "[STEP] start train"

cd "${REPO}"

: >"${LOG}"
export PYTHONUNBUFFERED=1
nohup /root/eurobot_baby_venv/bin/python -u scripts/train.py \
  --resume models/checkpoints/step_918672 \
  >>"${LOG}" 2>&1 &

echo $! >/root/train_loop.pid
echo "[OK] train started pid=$(cat /root/train_loop.pid)"

sleep 5

echo "[STEP] check GPU"
nvidia-smi

echo "[STEP] tail log"
tail -n 20 "${LOG}"
