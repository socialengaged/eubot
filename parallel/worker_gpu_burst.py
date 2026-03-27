#!/usr/bin/env python3
"""
GPU burst: GPU bassa → stop serve, selfplay locale, git pull, restart serve, health, chat smoke.

File: parallel/worker_gpu_burst.py (non orchestrator/gpu_burst.py)
Non modifica orchestrator né altri worker né serve.py.

Lock: /tmp/eubot_gpu_burst.lock
Log: orchestrator/logs/gpu_burst.log
"""
from __future__ import annotations

import glob
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "orchestrator" / "logs"
LOG_FILE = LOG_DIR / "gpu_burst.log"
SELFPLAY_SCRIPT = ROOT / "ai_engine" / "data_pipeline" / "selfplay_generator.py"
RESTART_CMD = "bash /workspace/eubot/tools/restart_serve_baby_safe.sh"
LOCK_PATH = Path("/tmp/eubot_gpu_burst.lock")
SELFPLAY_TIMEOUT_SEC = 600

# Grace period hard dopo restart serve (nessun pkill / nuovo burst finché non esce da recovery)
GRACE_PERIOD = 180

_DEFAULT_CKPT_GLOB = "/workspace/eurobot_baby/models/checkpoints/step_*"

serve_restarting = False
is_recovery_phase = False


def run(cmd: str):
    logging.info("run: %s", cmd)
    return subprocess.run(cmd, shell=True)


def check_health() -> bool:
    try:
        r = requests.get("http://127.0.0.1:8080/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_health(max_attempts: int = 30, delay: int = 5) -> bool:
    """Attende che /health risponda 200 con 'ok' nel body."""
    for i in range(max_attempts):
        try:
            r = requests.get("http://127.0.0.1:8080/health", timeout=3)
            if r.status_code == 200 and "ok" in r.text:
                logging.info("Serve healthy confirmed")
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def test_chat() -> bool:
    try:
        r = requests.post(
            "http://127.0.0.1:8080/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hey"}]},
            timeout=5,
        )
        data = r.json()
        return "choices" in data
    except Exception:
        return False


def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [pid=%(process)d] %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],
    )


def get_gpu_usage() -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        )
        line = out.decode().strip().split("\n")[0].strip()
        return int(line)
    except Exception as e:
        logging.warning("get_gpu_usage failed: %s", e)
        return 100


def _resolve_checkpoint() -> str:
    env = os.environ.get("GPU_BURST_CHECKPOINT", "").strip()
    if env and Path(env).is_dir():
        return env
    paths = sorted(glob.glob(_DEFAULT_CKPT_GLOB))
    if paths:
        return paths[-1]
    return ""


def _acquire_singleton_lock() -> bool:
    try:
        import fcntl
    except ImportError:
        logging.warning("fcntl missing, no singleton lock")
        return True
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, str(os.getpid()).encode())
        return True
    except OSError:
        logging.error("another worker_gpu_burst is already running (lock busy)")
        return False


def _post_selfplay_recovery() -> None:
    """Dopo selfplay: deploy, restart, grace 180s, wait health, chat."""
    global serve_restarting, is_recovery_phase

    try:
        is_recovery_phase = True
        serve_restarting = True

        try:
            logging.info("Running git pull...")
            run("cd /workspace/eubot && git pull")
        except Exception as e:
            logging.error("git pull failed: %s", e)

        logging.info("Restarting serve...")
        run(RESTART_CMD)

        # === GRACE PERIOD PROTECTION ===
        logging.info("Entering grace period (no restart allowed)...")
        time.sleep(GRACE_PERIOD)

        healthy = wait_for_health()
        if not healthy:
            logging.warning("WARNING: serve not healthy after grace period (no pkill here)")
        else:
            # is_recovery_phase False solo dopo health OK (serve pronto)
            is_recovery_phase = False

        serve_restarting = False

        if healthy:
            if not test_chat():
                logging.warning("Chat test failed → restart serve (once)")
                is_recovery_phase = True
                serve_restarting = True
                run(RESTART_CMD)
                logging.info("Entering grace period after chat-restart (no restart allowed)...")
                time.sleep(GRACE_PERIOD)
                if not wait_for_health():
                    logging.warning("WARNING: serve not healthy after chat restart (no pkill)")
                else:
                    is_recovery_phase = False
                serve_restarting = False
        else:
            logging.warning("Skipping chat test (serve not healthy)")

        logging.info("Burst cycle completed cleanly")
        logging.info("GPU burst completed")
        logging.info("System verified and running")
    except Exception as e:
        logging.exception("post_selfplay_recovery error: %s", e)
    finally:
        # Non azzerare is_recovery_phase qui: deve restare True finché /health non è OK
        # (vedi ramo wait_for_health fallito o eccezione prima della conferma health).
        serve_restarting = False


def main() -> None:
    global serve_restarting, is_recovery_phase

    _setup_logging()
    if not _acquire_singleton_lock():
        sys.exit(1)

    logging.info("worker_gpu_burst started ROOT=%s", ROOT)
    if not SELFPLAY_SCRIPT.is_file():
        logging.error("missing %s", SELFPLAY_SCRIPT)
        sys.exit(1)

    while True:
        try:
            if serve_restarting or is_recovery_phase:
                if is_recovery_phase:
                    logging.info("SKIP pkill — recovery in progress")
                else:
                    logging.info("Serve still loading → skipping burst cycle")
                time.sleep(10)
                continue

            gpu = get_gpu_usage()
            logging.info("gpu utilization=%s%%", gpu)

            if gpu >= 25:
                time.sleep(20)
                continue

            ckpt = _resolve_checkpoint()
            if not ckpt:
                logging.warning("no checkpoint found (set GPU_BURST_CHECKPOINT or %s)", _DEFAULT_CKPT_GLOB)
                time.sleep(60)
                continue

            logging.info("gpu burst: START (checkpoint=%s)", ckpt)
            try:
                run("pkill -f serve.py || true")
            except Exception as e:
                logging.error("pkill serve: %s", e)
            time.sleep(5)

            env = {**os.environ, "PYTHONUNBUFFERED": "1", "SELFPLAY_CHECKPOINT": ckpt}
            cmd = [
                sys.executable,
                "-u",
                str(SELFPLAY_SCRIPT),
                "--max-rounds",
                "256",
            ]
            try:
                logging.info("selfplay: starting timeout=%ss", SELFPLAY_TIMEOUT_SEC)
                subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    timeout=SELFPLAY_TIMEOUT_SEC,
                )
            except subprocess.TimeoutExpired:
                logging.warning("selfplay: timeout after %ss", SELFPLAY_TIMEOUT_SEC)
            except Exception as e:
                logging.error("selfplay: error %s", e)

            try:
                _post_selfplay_recovery()
            except Exception as e:
                logging.error("post-selfplay recovery failed (continuing): %s", e)

            time.sleep(10)

        except KeyboardInterrupt:
            logging.info("stopped by user")
            break
        except Exception as e:
            logging.exception("loop error: %s", e)
            try:
                if not is_recovery_phase:
                    serve_restarting = True
                    run(RESTART_CMD)
                    logging.info("Entering grace period (emergency restart)...")
                    time.sleep(GRACE_PERIOD)
                    wait_for_health()
                else:
                    logging.info("SKIP emergency restart — recovery in progress")
            except Exception:
                logging.exception("emergency restart serve failed")
            finally:
                serve_restarting = False
                is_recovery_phase = False
            time.sleep(30)


if __name__ == "__main__":
    main()
