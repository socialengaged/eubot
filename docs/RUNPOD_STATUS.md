# RunPod — stato runtime (sintesi)

File aggiornato dagli interventi di deploy/validazione. Path sul pod: `/workspace/eubot/docs/RUNPOD_STATUS.md` (dopo `git pull`).

## Ultimo aggiornamento

- **Data (UTC):** 2026-03-28 ~06:30 — deploy script su pod (SCP + `sed` CRLF→LF); train ripreso da `step_918672` (`batch_size` 64, `max_steps` 1.4M, `save_every` 2000 in `training.yaml`); worker `parallel/worker_gpu_burst.py` attivo; orchestrator **singola istanza** dopo stop watchdog/orchestrator vecchi e rimozione lock; `nohup` usa path assoluti a `worker_gpu_burst.py` / `orchestrator.py` (evita cwd `/root`).
- **Nota:** se `/workspace/eubot` non è clone git sul pod, aggiornare con `scp` o `rsync` da repo locale dopo `git pull`.
- **Precedente:** 2026-03-28 — allineamento script `runpod_full_utilization.sh`, config orchestrator (`sleep_between_tasks: 8`, selfplay/eval disabilitati durante run train dedicato)
- **Host SSH tipico:** `root@194.68.245.207` porta `22125` (verificare in dashboard RunPod se il pod è stato ricreato)
- **Chiave:** `%USERPROFILE%\.ssh\eubot_ed25519`

## Regole operative (anti-footgun)

1. **Non** usare `pkill -f serve.py` o `pkill -f train.py` da una **singola** riga `ssh ... "..."` se la stringa compare nel comando remoto: può terminare la sessione SSH. Preferire: `fuser -k 8080/tcp` per il serve, oppure `pgrep` + `kill` da **script** eseguito sul pod (`tools/runpod_full_utilization.sh`).
2. `eurobot_baby/scripts/train.py` sul pod accetta soprattutto `--resume` e legge **batch / max_steps / save_every** da `configs/training.yaml` — non `--dataset` / `--num_workers` come in altri template.
3. `parallel/worker_gpu_burst.py` **non** espone `--mode selfplay` né `--parallel`; è un loop che interviene quando la GPU è bassa.
4. `orchestrator/orchestrator.py` **non** ha `--loop` / `--interval`; l’intervallo è `loop.sleep_between_tasks` in `orchestrator/config.yaml`.

## Script utilizzo massimo

```bash
cd /workspace/eubot && git pull origin main
bash tools/runpod_full_utilization.sh
```

Variabili opzionali: `EUBOT_MAX_STEPS`, `EUBOT_BATCH_SIZE`, `EUBOT_SAVE_EVERY`, `EUBOT_TRAIN_LOG`.

## Stato atteso dopo avvio (criteri di validazione)

| Criterio | Note |
|----------|------|
| GPU util > 85% | Può richiedere 1–5 min dopo resume (caricamento pesi / primo step) |
| CPU util > 70% | Dipende da vCPU e da orchestrator/cleaner; non garantito al 100% |
| Log train | `/root/train_loop.log` + `PYTHONUNBUFFERED=1` nello script |
| Selfplay log | `orchestrator/logs/selfplay.log` (burst worker) |
| Orchestrator | `orchestrator/logs/orchestrator.log` |

Se **OOM** o instabilità: ridurre `batch_size` in `configs/training.yaml` e rilanciare lo script dopo stop.

## Processi e config

- **Train:** PID in `/root/train_loop.pid` (se creato dallo script)
- **GPU burst:** `/root/gpu_burst.pid`
- **Orchestrator:** `/root/orchestrator.pid`
- **Training YAML:** `/workspace/eurobot_baby/configs/training.yaml` (backup automatico `.bak.*` dallo script)

## Conflitto train vs selfplay GPU

Con `train.py` attivo sulla GPU, i task orchestrator che lanciano **selfplay_generator** / **eval** in locale possono competere per VRAM. In produzione: disabilitare tali task in `orchestrator/config.yaml` (`enabled: false`) oppure non lanciare train in parallelo a selfplay locale.
