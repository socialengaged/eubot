# RunPod — stato runtime (sintesi)

File aggiornato dagli interventi di deploy/validazione. Path sul pod: `/workspace/eubot/docs/RUNPOD_STATUS.md` (dopo `git pull`).

## Ultimo aggiornamento

- **Data (UTC):** 2026-03-28 — **Dual mode (train + inferenza):** checkpoint statico `models/checkpoints/serve_checkpoint` (copia da `step_*` attivo). Script **`tools/start_serve_safe.sh`**: serve su **CPU** (`CUDA_VISIBLE_DEVICES=""`) così il **train resta principale sulla GPU** senza secondo modello in VRAM. API: `GET /health`, **`POST /v1/chat/completions`** (OpenAI `messages`; **non** esiste `/chat`). Timeout chat: `SERVE_CHAT_REQUEST_TIMEOUT_SEC` (default script 300s). Dopo **SCP da Windows**, normalizzare LF: `python3 /workspace/eubot/tools/fix_crlf.py /workspace/eubot/tools/start_serve_safe.sh`. Test rapido chat: `curl -H 'Content-Type: application/json' -d @tools/pod_chat_payload.json …/v1/chat/completions` (sul pod). Log `/root/serve.log`, PID `/root/serve.pid`. GPU sharing **non** usato (serve CPU); train mantiene ~20 GB VRAM.
- **Data (UTC):** 2026-03-28 — **Train stabile post-streaming:** resume senza loop `scheduler.step()` a vuoto: **`last_epoch=resume_step-1`** oppure **`scheduler.pt` / `optimizer.pt`** se presenti nel checkpoint (salvati a ogni `save_every`). AMP: skip micro-batch se loss non finita; `unscale_` → `clip_grad_norm` → `scaler.step`. Metriche **`[TRAIN][PERF]`** ogni `perf_log_every` (EMA `step_time`, `tok/s`, `eff_batch`). Test pod dopo deploy SSH.
- **Data (UTC):** 2026-03-28 — **Streaming train (no preload JSONL):** `train.py` usa `StreamingJsonlIterableDataset` + `collate_fn` (padding, `attention_mask`, `labels` con `-100` sul pad). **Niente** fase “Building train blocks” in RAM; il primo step è subito dopo load pesi. `DataLoader`: `num_workers=0`, `shuffle=False` (IterableDataset). `training.yaml`: `batch_size` 32, `gradient_accumulation_steps` 8 (eff. batch 256; era 128×2 e dava **OOM** su A40 con seq 512). GPU util/memoria salgono subito con i batch (test pod: ~20 GB VRAM, util ~100%).
- **Data (UTC):** 2026-03-28 — **Pipeline pod-safe:** `tools/runpod_safe_pipeline.sh` — stop train con **`pgrep` + `kill` su PID** (no `pkill -f`), serve con **`fuser -k 8080/tcp`**, orchestrator con **`pgrep` + `kill`** su path `/workspace/eubot/orchestrator/orchestrator.py`. Train: **`device=torch.device(cuda|cpu)`**, `model.to(device)`, assert CUDA, batch via `_move_batch_to_device` (dict o tensore). Test pod: `bash tools/runpod_safe_pipeline.sh` poi `nvidia-smi` + `tail /root/train_loop.log`.
- **Data (UTC):** 2026-03-28 — **GPU reale (train):** `torch.amp` (autocast + GradScaler), assert CUDA, batch dict con H2D. Config batch/accum da `training.yaml`. Deploy: `scp` `eurobot_baby/scripts/train.py` sul pod; restart con `tools/runpod_safe_pipeline.sh` o `nohup` da `/workspace/eurobot_baby`.
- **Data (UTC):** 2026-03-28 ~06:30 — deploy script su pod (SCP + `sed` CRLF→LF); train ripreso da `step_918672` (`batch_size` 64, `max_steps` 1.4M, `save_every` 2000 in `training.yaml`); worker `parallel/worker_gpu_burst.py` attivo; orchestrator **singola istanza** dopo stop watchdog/orchestrator vecchi e rimozione lock; `nohup` usa path assoluti a `worker_gpu_burst.py` / `orchestrator.py` (evita cwd `/root`).
- **Nota:** se `/workspace/eubot` non è clone git sul pod, aggiornare con `scp` o `rsync` da repo locale dopo `git pull`.
- **Precedente:** 2026-03-28 — allineamento script `runpod_full_utilization.sh`, config orchestrator (`sleep_between_tasks: 8`, selfplay/eval disabilitati durante run train dedicato)
- **Host SSH tipico:** `root@194.68.245.207` porta `22125` (verificare in dashboard RunPod se il pod è stato ricreato)
- **Chiave:** `%USERPROFILE%\.ssh\eubot_ed25519`

## Regole operative (anti-footgun)

1. **Non** usare `pkill -f serve.py` o `pkill -f train.py` da una **singola** riga `ssh ... "..."` se la stringa compare nel comando remoto: può terminare la sessione SSH. Preferire: `fuser -k 8080/tcp` per il serve, oppure `pgrep` + `kill` da **script** eseguito sul pod (`tools/runpod_full_utilization.sh`).
2. `eurobot_baby/scripts/train.py` sul pod accetta soprattutto `--resume` e legge **batch, accum, max_steps, save_every** da `configs/training.yaml`. Lo stream JSONL usa **`num_workers=0`** nel codice (IterableDataset); non flag `--dataset` da CLI (template vecchi).
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
