# Best practices — continuare il training (Eurobot Baby)

Linee guida operative per il pod GPU RunPod (`/workspace/eurobot_baby/`). Per deploy SSH, venv e log su `/root`, vedi anche [`docs/EUROBOT_BABY_RUNPOD.md`](../../docs/EUROBOT_BABY_RUNPOD.md).

---

## Principi

1. **Un solo carico pesante sulla GPU:** o `train.py` o `serve.py`. Lo script [`tools/runpod_baby_training_resume.sh`](../../tools/runpod_baby_training_resume.sh) termina `serve.py` prima del training; dopo un run lungo, rilancia il serve sull’ultimo `step_*` solo quando serve inference/chat (vedi [`README.md`](../README.md)).
2. **Config riproducibile:** prima di modificare `configs/training.yaml`, copia con timestamp sul pod, es.:  
   `cp -a configs/training.yaml "configs/training.yaml.bak-$(date -u +%Y%m%dT%H%M%SZ)"`.
3. **Log affidabili:** scrivere in **`/root/training_baby_*.log`** con `PYTHONUNBUFFERED=1` e `python -u`; su NFS `/workspace` i log possono restare vuoti fino al flush.

---

## Checklist — prima di ogni run (pod)

Copia e spunta sul terminale SSH:

```
[ ] nvidia-smi          — GPU libera o solo il processo atteso
[ ] df -h               — spazio su / e /workspace sufficiente
[ ] export HF_HOME=/root/hf_home e HF_HUB_CACHE=/root/hf_home (o path coerente)
[ ] ls models/checkpoints/ — annotare ultimo step_* valido per --resume
```

---

## Checklist — configurazione e `max_steps`

```
[ ] Backup di training.yaml con timestamp (vedi Principi §2)
[ ] max_steps GLOBALE > step del checkpoint usato in --resume (altrimenti "Nothing to do")
[ ] Dopo build_dataset: annotare N chunks; steps_per_epoch = ceil(N / batch_eff)
      batch_eff = batch_size × gradient_accumulation_steps (es. 4×4 = 16)
[ ] max_steps = epoche_desiderate × steps_per_epoch (es. 2 epoche: vedi RUNBOOK)
```

Dettaglio rebuild e formula: [`RUNBOOK_PHASE0.md`](RUNBOOK_PHASE0.md).

---

## Checklist — avvio training (pattern produzione)

```
[ ] Se serve solo training: fermare serve.py (pgrep -f serve.py; kill se necessario)
[ ] venv: /root/eurobot_baby_venv/bin/python (o quello in uso sul pod)
[ ] nohup + env PYTHONUNBUFFERED=1 HF_HOME=... HF_HUB_CACHE=... python -u scripts/train.py --resume ...
[ ] Redirect log su /root/, es.: >> /root/training_baby_<nome>.log 2>&1 &
[ ] save_every / eval_every proporzionati (es. 4000 / 2000 come in runpod_baby_training_resume.sh)
```

Esempio (adatta `step_*` e nome log):

```bash
cd /workspace/eurobot_baby
nohup env PYTHONUNBUFFERED=1 HF_HOME=/root/hf_home HF_HUB_CACHE=/root/hf_home \
  /root/eurobot_baby_venv/bin/python -u scripts/train.py \
  --resume models/checkpoints/step_<ULTIMO> \
  >> /root/training_baby_<nome_run>.log 2>&1 &
```

Per sessioni interattive lunghe, `tmux` o `screen` sono utili oltre a `nohup`.

Script one-shot (ferma serve, riscrive yaml, resume): [`tools/runpod_baby_training_resume.sh`](../../tools/runpod_baby_training_resume.sh) (path nel repo root `eubot`).

---

## Checklist — monitoraggio

```
[ ] tail -f /root/training_baby_*.log — tqdm: global_step / max_steps coerente
[ ] pgrep -af train.py — un solo training intenzionale
[ ] Nuovi checkpoint in models/checkpoints/step_* ogni save_every step
```

### CPU/GPU ~0% durante il training

Se `nvidia-smi` mostra GPU inattiva e/o il processo non consuma CPU: verifica `ps`/`pgrep` su `train.py`, leggi il log per crash o DataLoader bloccato, controlla path dataset e config caricata, `CUDA_VISIBLE_DEVICES`, e `num_workers` (→ 0 se hang su I/O). Procedura completa con tabelle fix: [`RUNPOD_TRAIN_V3_24H.md`](RUNPOD_TRAIN_V3_24H.md) (sezione **Diagnostica — training idle**).

---

## Watchdog — retry e resume automatico (checkpoint)

Il trainer salva già `step_*` su disco (`save_every` in `training.yaml`). Se il processo **crash** (OOM, pod instabile, kill accidentale), puoi **riavviare** con `--resume` sull’ultimo `step_*` senza perdere il lavoro già salvato.

Lo script [`tools/runpod_baby_train_watchdog.sh`](../../tools/runpod_baby_train_watchdog.sh) (eseguire sul **pod**):

- Esegue `train.py --resume` sull’**ultimo** checkpoint che matcha `CHECKPOINT_GLOB` (default `models/checkpoints/step_*`).
- Se `train.py` esce con **codice ≠ 0**, attende (backoff con tetto) e **rilancia** dallo stesso checkpoint più recente (dopo un crash, il nuovo checkpoint è l’ultimo step scritto).
- Scrive stato in `logs/train_watchdog.state.log` e log training in `logs/train_watchdog_inner.log` (sotto `WORKDIR`).
- **Lock file** `/tmp/eurobot_baby_train_watchdog.lock` — una sola istanza alla volta.
- Opzionale: `PATCH_MAX_STEPS_YAML=1` imposta `max_steps` in `configs/training.yaml` da `MAX_STEPS` (backup prima della modifica).

Esempio (tmux, target 600k step):

```bash
cd /workspace/eurobot_baby
export MAX_STEPS=600000
# opz. training su cartella v3:
# export CHECKPOINT_GLOB="models/checkpoints_v3/step_*"
bash /path/to/eubot/tools/runpod_baby_train_watchdog.sh
```

**Nota:** se un processo `train.py` è già avviato a mano, ferma prima il duplicato; il watchdog può fermare `serve.py` (`STOP_SERVE=1` default) per liberare la GPU. Dopo il training, riavvia il serve come negli altri runbook.

---

## Checklist — backup prima di spegnere il pod o dopo run lunghi

```
[ ] Tier A/B come in EXPORT_MIGRATE (checkpoint + opz. train.jsonl)
[ ] Bundle opzionale: vedi docs/OFFLINE_EXPORT_BABY_AND_SAGE.md
[ ] Esempio bundle locale: backup_runpod_baby_data/README_ESSENTIALS.txt (checksum)
```

Riferimenti: [`EXPORT_MIGRATE_FROM_RUNPOD.md`](EXPORT_MIGRATE_FROM_RUNPOD.md), [`../../docs/OFFLINE_EXPORT_BABY_AND_SAGE.md`](../../docs/OFFLINE_EXPORT_BABY_AND_SAGE.md).

---

## Dataset: modifiche e staging

Se aggiungi raw o usi `raw_staging`: ferma il training, promuovi staging se applicabile, rebuild, ricalcola `max_steps`. Flusso: [`RUNBOOK_PHASE0.md`](RUNBOOK_PHASE0.md) e [`tools/scraping/SETUP_SCRAPER.md`](../tools/scraping/SETUP_SCRAPER.md).

---

## Allineamento repo ↔ pod

Dopo modifiche a `build_dataset.py` o `train.py` nel monorepo, sincronizza sul pod (scp/rsync) come in RUNBOOK §1.

---

## Ordine operativo sul pod (coerente, senza errori)

Esegui **sempre** da SSH sul pod (path tipico `/workspace/eurobot_baby`). Gli script vivono nella root del repo `eubot` in [`tools/`](../../tools/); copiali sul pod (es. `/root/eubot-tools/`) o monta il repo.

| Step | Azione |
|------|--------|
| 1 | **Sincronizza** `build_dataset.py` / `configs` se hai modificato il monorepo locale (vedi RUNBOOK §1). Gli script in `tools/*.sh` devono avere **fine riga Unix (LF)**; in repo è impostato [`.gitattributes`](../../.gitattributes) (`*.sh text eol=lf`). Su Windows, se `bash` sul pod segnala `$'\r': command not found`, riconverti o ricopia dopo salvataggio LF. |
| 2 | **Precheck:** `bash /path/to/runpod_pod_precheck.sh` — GPU, disco, ultimo `step_*`, processi, log. |
| 3 | **Solo training:** `MODE=train bash /path/to/runpod_pod_orchestrate.sh` — include precheck e avvia [`runpod_baby_training_auto_resume.sh`](../../tools/runpod_baby_training_auto_resume.sh) (ferma `serve`, backup yaml, `--resume`). |
| 4 | **Solo API chat** (training fermo): `MODE=serve bash /path/to/runpod_pod_orchestrate.sh` — usa [`restart_serve_baby_safe.sh`](../../tools/restart_serve_baby_safe.sh) sull’**ultimo** checkpoint. |
| 5 | **Monitor:** `tail -f /root/training_baby_auto.log` (o il `LOG_FILE` scelto). |

Regole: **non** lanciare `MODE=serve` mentre `train.py` è attivo; `orchestrate` in modalità serve esce con errore se il training gira ancora.

Da PC Windows, verifica che la porta SSH del Pod sia aperta prima di `scp`/`ssh` (es. `powershell -File tools/check_runpod_ssh.ps1`). Se vedi **Permission denied (publickey)**, usa la chiave indicata nella dashboard RunPod (es. `~/.ssh/id_ed25519`) e l’utente `root` sulla **TCP diretta** che supporta SCP.

---

## Roadmap dati (dopo training stabile)

Fasi opzionali dataset: [`../../docs/PIANO_DATASET_EXPANSION_v1.md`](../../docs/PIANO_DATASET_EXPANSION_v1.md).
