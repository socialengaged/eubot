#!/usr/bin/env bash
# RunPod: stop sicuri (no pkill -f "serve.py" da one-liner SSH) + train + worker_gpu_burst + orchestrator.
# Esegui SUL POD da: cd /workspace/eubot && bash tools/runpod_full_utilization.sh
#
# Variabili opzionali:
#   EUBOT_VENV_PYTHON=/root/eurobot_baby_venv/bin/python
#   EUBOT_MAX_STEPS=1400000  EUBOT_BATCH_SIZE=64  EUBOT_SAVE_EVERY=2000
#   EUBOT_TRAIN_LOG=/root/train_loop.log
#
# Nota: eurobot_baby/scripts/train.py legge batch/max_steps/save da configs/training.yaml (non CLI come --dataset).
# worker_gpu_burst.py non ha --mode/--parallel; orchestrator.py non ha --loop (loop da config.yaml).

set -uo pipefail

EUBOT_VENV_PYTHON="${EUBOT_VENV_PYTHON:-/root/eurobot_baby_venv/bin/python}"
BABY="/workspace/eurobot_baby"
REPO="/workspace/eubot"
TRAIN_LOG="${EUBOT_TRAIN_LOG:-/root/train_loop.log}"
MAX_STEPS="${EUBOT_MAX_STEPS:-1400000}"
BATCH_SIZE="${EUBOT_BATCH_SIZE:-64}"
SAVE_EVERY="${EUBOT_SAVE_EVERY:-2000}"
YAML="$BABY/configs/training.yaml"

die() { echo "ERROR: $*" >&2; exit 1; }
test -d "$REPO" || die "missing $REPO"
test -d "$BABY" || die "missing $BABY"
test -x "$EUBOT_VENV_PYTHON" || die "missing venv python $EUBOT_VENV_PYTHON"
test -f "$BABY/scripts/train.py" || die "missing $BABY/scripts/train.py"

_stop_one_pattern() {
  local pat="$1"
  local p
  for p in $(pgrep -f "$pat" 2>/dev/null || true); do
    if kill -0 "$p" 2>/dev/null; then kill "$p" 2>/dev/null || true; fi
  done
}

echo "[1] stop processi (pattern sicuri, no pkill -f serve.py da ssh one-liner)"
fuser -k 8080/tcp 2>/dev/null || true
_stop_one_pattern "eurobot_baby_venv/bin/python -u scripts/train.py"
_stop_one_pattern "parallel/worker_gpu_burst.py"
_stop_one_pattern "orchestrator/watchdog.py"
_stop_one_pattern "orchestrator/orchestrator.py"
sleep 3
LOCK="$REPO/orchestrator/.orchestrator.lock"
if [[ -f "$LOCK" ]]; then rm -f "$LOCK" && echo "[1b] removed stale $LOCK"; fi

echo "[2] nvidia-smi (GPU libera)"
nvidia-smi || true

echo "[3] patch training.yaml (backup + sed)"
if [[ ! -f "$YAML" ]]; then die "missing $YAML"; fi
cp -a "$YAML" "${YAML}.bak.$(date +%Y%m%d%H%M%S)"
sed -i "s/^max_steps:.*/max_steps: ${MAX_STEPS}/" "$YAML"
sed -i "s/^batch_size:.*/batch_size: ${BATCH_SIZE}/" "$YAML"
sed -i "s/^save_every:.*/save_every: ${SAVE_EVERY}/" "$YAML"
grep -E '^max_steps:|^batch_size:|^save_every:' "$YAML"

LATEST="$(ls -d "$BABY"/models/checkpoints/step_* 2>/dev/null | sort -V | tail -1 || true)"
[[ -n "$LATEST" ]] || die "no checkpoint step_* under $BABY/models/checkpoints"
echo "[4] resume da: $LATEST"

export PYTHONUNBUFFERED=1
cd "$BABY"
: >"$TRAIN_LOG"
nohup "$EUBOT_VENV_PYTHON" -u scripts/train.py --resume "$LATEST" >>"$TRAIN_LOG" 2>&1 &
echo $! > /root/train_loop.pid
echo "[OK] TRAIN STARTED pid=$(cat /root/train_loop.pid)"

mkdir -p "$REPO/orchestrator/logs"
# Percorsi assoluti: nohup non deve dipendere dal cwd (evita /root/orchestrator/...).
nohup "$EUBOT_VENV_PYTHON" -u "$REPO/parallel/worker_gpu_burst.py" >>"$REPO/orchestrator/logs/selfplay.log" 2>&1 &
echo $! > /root/gpu_burst.pid
echo "[OK] worker_gpu_burst STARTED pid=$(cat /root/gpu_burst.pid)"

nohup "$EUBOT_VENV_PYTHON" -u "$REPO/orchestrator/orchestrator.py" >>"$REPO/orchestrator/logs/orchestrator.log" 2>&1 &
echo $! > /root/orchestrator.pid
echo "[OK] orchestrator STARTED pid=$(cat /root/orchestrator.pid)"

sleep 5
echo "[5] nvidia-smi"
nvidia-smi || true
echo "[6] ps"
ps aux | grep -E 'train.py|worker_gpu_burst|orchestrator.py' | grep -v grep || true
echo "[7] tail train log"
tail -n 20 "$TRAIN_LOG" || true
