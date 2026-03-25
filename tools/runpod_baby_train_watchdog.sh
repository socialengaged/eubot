#!/usr/bin/env bash
# Run ON RunPod (root), under /workspace/eurobot_baby.
# Runs scripts/train.py in a loop: always --resume from the latest models/.../step_*.
# On crash or non-zero exit, waits (retry with capped exponential backoff) and restarts
# from the newest checkpoint on disk (work done is preserved by train.py checkpoints).
#
# Normal exit: latest checkpoint step >= MAX_STEPS, or train.py exits 0 and step target reached.
#
# Usage (tmux recommended):
#   cd /workspace/eurobot_baby
#   chmod +x /path/to/runpod_baby_train_watchdog.sh
#   MAX_STEPS=600000 bash /path/to/runpod_baby_train_watchdog.sh
#
# Background:
#   nohup env MAX_STEPS=600000 bash /path/to/runpod_baby_train_watchdog.sh >> /root/watchdog_baby.log 2>&1 &
#
# Env:
#   WORKDIR              Default /workspace/eurobot_baby
#   VENV_PY              Default /root/eurobot_baby_venv/bin/python
#   MAX_STEPS            Global target step (required for stop condition)
#   CHECKPOINT_GLOB      Default "models/checkpoints/step_*" (use e.g. models/checkpoints_v3/step_* for v3)
#   TRAIN_LOG            Default $WORKDIR/logs/train_watchdog_inner.log
#   STATE_FILE           Default $WORKDIR/logs/train_watchdog.state.log
#   LOCK_FILE            Default /tmp/eurobot_baby_train_watchdog.lock
#   RETRY_MIN_SEC        First wait after failure (default 60)
#   RETRY_MAX_SEC        Cap backoff (default 900)
#   MAX_RESTARTS         0 = unlimited (default 0)
#   STOP_SERVE           Default 1 — kill scripts/serve.py before each train invocation
#   PATCH_MAX_STEPS_YAML Default 1 — set max_steps: in configs/training.yaml from MAX_STEPS
#
set -uo pipefail

WORKDIR="${WORKDIR:-/workspace/eurobot_baby}"
VENV_PY="${VENV_PY:-/root/eurobot_baby_venv/bin/python}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-models/checkpoints/step_*}"
TRAIN_LOG="${TRAIN_LOG:-$WORKDIR/logs/train_watchdog_inner.log}"
STATE_FILE="${STATE_FILE:-$WORKDIR/logs/train_watchdog.state.log}"
LOCK_FILE="${LOCK_FILE:-/tmp/eurobot_baby_train_watchdog.lock}"
RETRY_MIN_SEC="${RETRY_MIN_SEC:-60}"
RETRY_MAX_SEC="${RETRY_MAX_SEC:-900}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"
STOP_SERVE="${STOP_SERVE:-1}"
PATCH_MAX_STEPS_YAML="${PATCH_MAX_STEPS_YAML:-1}"

if [[ -z "${MAX_STEPS:-}" ]]; then
  echo "ERR: set MAX_STEPS (global target), e.g. MAX_STEPS=600000" >&2
  exit 1
fi

cd "$WORKDIR" || exit 1
mkdir -p "$(dirname "$TRAIN_LOG")" "$(dirname "$STATE_FILE")" 2>/dev/null || true

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERR: Python not found: $VENV_PY" >&2
  exit 1
fi
if [[ ! -f scripts/train.py ]]; then
  echo "ERR: scripts/train.py not found under $WORKDIR" >&2
  exit 1
fi

exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "ERR: another watchdog holds $LOCK_FILE — exit." >&2
  exit 1
fi

if pgrep -f 'scripts/train.py' >/dev/null 2>&1; then
  echo "ERR: train.py is already running. Stop it (pkill -f scripts/train.py) before starting the watchdog." >&2
  exit 1
fi

log_state() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$STATE_FILE"
}

latest_ckpt() {
  shopt -s nullglob
  # shellcheck disable=SC2206
  local arr=( $CHECKPOINT_GLOB )
  [[ ${#arr[@]} -eq 0 ]] && return 1
  printf '%s\n' "${arr[@]}" | sort -V | tail -n1
}

step_from_ckpt() {
  local base
  base="$(basename "$1")"
  echo "${base#step_}"
}

stop_serve_if_needed() {
  if [[ "$STOP_SERVE" != "1" ]]; then
    return 0
  fi
  local p
  p="$(pgrep -f '/workspace/eurobot_baby/scripts/serve.py' || true)"
  if [[ -n "$p" ]]; then
    log_state "Stopping serve.py PID(s): $p"
    pkill -f '/workspace/eurobot_baby/scripts/serve.py' || true
    sleep 2
  fi
}

patch_yaml_max_steps() {
  if [[ "$PATCH_MAX_STEPS_YAML" != "1" ]]; then
    return 0
  fi
  local cfg="$WORKDIR/configs/training.yaml"
  if [[ ! -f "$cfg" ]]; then
    log_state "WARN: no $cfg — skip PATCH_MAX_STEPS_YAML"
    return 0
  fi
  cp -a "$cfg" "${cfg}.bak-watchdog-$(date -u +%Y%m%dT%H%M%SZ)"
  if grep -qE '^max_steps:' "$cfg"; then
    sed -i "s/^max_steps:.*/max_steps: ${MAX_STEPS}/" "$cfg"
  else
    echo "max_steps: ${MAX_STEPS}" >>"$cfg"
  fi
  log_state "Patched configs/training.yaml max_steps=$MAX_STEPS (backup beside file)"
}

patch_yaml_max_steps
stop_serve_if_needed

restarts=0
while true; do
  LATEST="$(latest_ckpt || true)"
  if [[ -z "${LATEST:-}" ]]; then
    log_state "ERR: no checkpoint matching $CHECKPOINT_GLOB — cannot resume."
    exit 1
  fi
  CUR="$(step_from_ckpt "$LATEST")"
  if ! [[ "$CUR" =~ ^[0-9]+$ ]]; then
    log_state "ERR: bad step from $LATEST"
    exit 1
  fi

  if [[ "$CUR" -ge "$MAX_STEPS" ]]; then
    log_state "Done: latest checkpoint step $CUR >= MAX_STEPS $MAX_STEPS ($LATEST)"
    exit 0
  fi

  log_state "START train.py --resume $LATEST (step $CUR / target $MAX_STEPS) restarts_so_far=$restarts"
  echo "----- $(date -u +%Y-%m-%dT%H:%M:%SZ) run restarts=$restarts -----" >>"$TRAIN_LOG"

  set +e
  env PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME:-/root/hf_home}" HF_HUB_CACHE="${HF_HUB_CACHE:-/root/hf_home}" \
    "$VENV_PY" -u scripts/train.py --resume "$LATEST" >>"$TRAIN_LOG" 2>&1
  exit_code=$?
  set -e

  log_state "train.py exited code=$exit_code"

  L2="$(latest_ckpt || true)"
  if [[ -n "$L2" ]]; then
    C2="$(step_from_ckpt "$L2")"
    if [[ "$C2" =~ ^[0-9]+$ ]] && [[ "$C2" -ge "$MAX_STEPS" ]]; then
      log_state "Done: checkpoint now $L2 (step $C2) >= MAX_STEPS"
      exit 0
    fi
  fi

  if [[ "$exit_code" -eq 0 ]]; then
    log_state "train.py exited 0 but step < MAX_STEPS — treating as finished run; exit."
    exit 0
  fi

  restarts=$((restarts + 1))
  if [[ "$MAX_RESTARTS" -gt 0 ]] && [[ "$restarts" -gt "$MAX_RESTARTS" ]]; then
    log_state "Max restarts ($MAX_RESTARTS) reached — exit."
    exit "$exit_code"
  fi

  # backoff: min(RETRY_MAX, RETRY_MIN * 2^min(restarts-1, 5))
  pow=1
  i=1
  while [[ $i -lt "$restarts" ]] && [[ $i -lt 6 ]]; do
    pow=$((pow * 2))
    i=$((i + 1))
  done
  delay=$((RETRY_MIN_SEC * pow))
  if [[ "$delay" -gt "$RETRY_MAX_SEC" ]]; then
    delay=$RETRY_MAX_SEC
  fi
  log_state "Retry in ${delay}s (attempt $restarts) ..."
  sleep "$delay"
  stop_serve_if_needed
done
