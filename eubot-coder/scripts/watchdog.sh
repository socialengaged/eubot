#!/bin/bash
# Watchdog: checks every hour if training is alive, restarts from checkpoint if dead.
# Usage: nohup bash scripts/watchdog.sh > watchdog.log 2>&1 &

WORKDIR="/workspace/eubot/eubot-coder"
LOG="$WORKDIR/training.log"
CHECK_INTERVAL=3600  # seconds (1 hour)

cd "$WORKDIR" || exit 1

is_training_running() {
    pgrep -f "python scripts/finetune.py" > /dev/null 2>&1
}

start_training() {
    echo "[$(date)] Starting training (--resume) ..."
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        nohup python scripts/finetune.py --resume >> "$LOG" 2>&1 &
    echo "[$(date)] Training PID: $!"
}

echo "[$(date)] Watchdog started. Checking every ${CHECK_INTERVAL}s."

while true; do
    if is_training_running; then
        LAST_LINE=$(tail -1 "$LOG" 2>/dev/null)
        echo "[$(date)] Training alive. Last: $LAST_LINE"
    else
        TAIL=$(tail -3 "$LOG" 2>/dev/null)
        if echo "$TAIL" | grep -q "Adapter saved to"; then
            echo "[$(date)] Training COMPLETED. Watchdog exiting."
            exit 0
        fi
        echo "[$(date)] Training NOT running. Restarting..."
        start_training
    fi
    sleep "$CHECK_INTERVAL"
done
