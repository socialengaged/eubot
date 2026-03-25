# RunPod — training ~24h su `final_dataset_v3.jsonl`

Obiettivo: sincronizzare il dataset, configurare `training_finetune_v3.yaml`, avviare `scripts/train.py` (non `train.py` in root), monitorare loss/checkpoint, testare con `serve.py`.

**Nota:** sul pod il loader YAML usa chiavi tipo `batch_size`, `data_train`, `checkpoint_dir`, `max_steps` (vedi [`configs/training.yaml.example`](../configs/training.yaml.example)). Non sono supportate le chiavi stile Hugging Face (`per_device_train_batch_size`, `dataset_path`, …) salvo diverso `train.py` custom.

## Step 0 — Sync dataset

Da PC (repo), adatta host/porta/chiave SSH al tuo pod:

```bash
scp ai_engine/data/final_dataset_v3.jsonl runpod-baby:/workspace/eurobot_baby/data/
```

Sul pod:

```bash
ssh runpod-baby
mkdir -p /workspace/eurobot_baby/data
ls -lh /workspace/eurobot_baby/data/final_dataset_v3.jsonl
```

### Val obbligatorio (se `train.py` lo richiede)

```bash
cd /workspace/eurobot_baby
head -n 4000 data/final_dataset_v3.jsonl > data/val_v3_sample.jsonl
```

## Step 1 — Ambiente

```bash
cd /workspace/eurobot_baby
source /root/eurobot_baby_venv/bin/activate
pip install -r requirements.txt || true
mkdir -p models/checkpoints_v3 logs
```

## Step 2 — `max_steps` e config

Conta le righe e calcola `max_steps` (3 epoche, batch efficace `2*8=16`):

```bash
N=$(wc -l < data/final_dataset_v3.jsonl)
echo "N=$N"
python3 -c "import math; n=int('$N'); eff=16; e=3; ms=math.ceil(n/eff)*e; w=max(1,int(0.1*ms)); print('max_steps', ms, 'warmup_steps', w)"
```

Inserisci i valori stampati in `configs/training_finetune_v3.yaml` (`max_steps`, `warmup_steps`), poi copia il file come `training.yaml` (passo sotto).

Aggiorna `configs/training_finetune_v3.yaml` con `max_steps` e `warmup_steps` calcolati, poi attiva la config:

```bash
cp -a configs/training.yaml "configs/training.yaml.bak-$(date -u +%Y%m%dT%H%M%SZ)" 2>/dev/null || true
cp configs/training_finetune_v3.yaml configs/training.yaml
# modifica configs/training.yaml con max_steps / warmup_steps reali se non già editato
```

## Step 3 — Avvio training

Ferma eventuali run precedenti e avvia (log sotto `logs/` nel repo):

```bash
pkill -f 'scripts/train.py' || true

cd /workspace/eurobot_baby
source /root/eurobot_baby_venv/bin/activate

nohup env PYTHONUNBUFFERED=1 HF_HOME=/root/hf_home HF_HUB_CACHE=/root/hf_home \
  python -u scripts/train.py \
  >> logs/train_v3.log 2>&1 &

echo $! > logs/train_v3.pid
sleep 10
tail -n 50 logs/train_v3.log
```

Se `train.py` supporta esplicitamente `--config configs/training_finetune_v3.yaml`, puoi omettere la copia su `training.yaml` e passare solo il flag (verifica con `python scripts/train.py --help` sul pod).

**Retry automatico:** se il processo crasha ma i checkpoint sono salvati, puoi usare lo script [`tools/runpod_baby_train_watchdog.sh`](../../tools/runpod_baby_train_watchdog.sh) sul pod (resume dall’ultimo `step_*`, backoff, stato in `logs/train_watchdog.state.log`). Dettagli: [`BEST_PRACTICES_TRAINING.md`](BEST_PRACTICES_TRAINING.md) (sezione **Watchdog**).

## Step 4 — Monitor (ogni 10–15 min)

```bash
tail -n 20 /workspace/eurobot_baby/logs/train_v3.log
nvidia-smi
pgrep -af train.py
```

Controlli: loss in diminuzione, nessun NaN, GPU utilizzata.

## Diagnostica — training idle (CPU/GPU ~0%)

Contesto: pod GPU RunPod, training Baby tipicamente `scripts/train.py` con log in `logs/` o `/root/training_*.log`.

### 1. Verifica processo

```bash
ps aux | grep -E 'train\.py|scripts/train' | grep -v grep
pgrep -af train
```

| Esito | Significato |
|-------|-------------|
| Nessuna riga | Training **non in esecuzione** → vedi **§4 Riavvio** sotto |
| Processo presente | Controlla **CPU%** nella colonna `ps` (non solo esistenza PID) |

### 2. Se il processo non esiste

- Conferma che non sia crashato: `tail -n 80` sull’ultimo log (es. `logs/train_v3.log`).
- Cercare `Error`, `Traceback`, `FileNotFound`, `CUDA`, `DataLoader`.
- **Riavvia** come da runbook training (venv, `nohup`, `configs/training.yaml` o `--config`), salvando nuovo PID in `logs/train_*.pid`.

### 3. Se il processo esiste ma sembra idle

Possibili cause da controllare in ordine:

1. **Deadlock / hang** (raro): processo in `D` state o bloccato su I/O — `ps -o pid,stat,pcpu,pmem,cmd -p <PID>`; se `STAT` anomalo o CPU 0 a lungo senza avanzamento nel log, **kill** pulito e riavvio.
2. **DataLoader bloccato** (`num_workers` > 0 su dataset lento o NFS): sintomo — log fermo all’inizio epoca, GPU 0%. **Fix**: in config ridurre `dataloader_num_workers` a **0** (se il `train.py` del pod lo supporta) o verificare path dataset su disco locale veloce.
3. **In attesa di checkpoint / barra tqdm**: aprire il log; se lo **step globale non aumenta** da molti minuti mentre il PID esiste → trattare come hang e riavvio dopo diagnosi.

Dopo modifiche: **riavvia** il training.

### 4. Verifica GPU

```bash
nvidia-smi
```

**Se GPU util ~0%** con processo `train` presente:

- Training **non sta computando** su GPU (modello su CPU per errore, CUDA non visibile, o loop senza step).
- Se **nessun** processo train: training **non partito** correttamente.

### 5. Fix tipici (mapping ai sintomi)

| Problema | Azione |
|----------|--------|
| **batch_size** troppo alto | OOM o crash precoce nel log; ridurre `batch_size` o `per_device_train_batch_size` e/o aumentare accumulo. |
| **batch_size** troppo basso + GPU piccola | GPU sottoutilizzata ma non zero se gira; se zero → probabile altro. |
| **Dataset path errato** | Log: file not found, 0 batch, `No chunks`; verificare path in YAML (`data_train`, `data_val`) e `ls -lh` sul pod. |
| **Config non caricata** | Training con default sbagliati o exit immediato; verificare `cp ... configs/training.yaml` o `--config`; confrontare hash/timestamp file config. |
| **CUDA_VISIBLE_DEVICES** | Vuoto o errato → training su CPU; `echo $CUDA_VISIBLE_DEVICES`, `python -c "import torch; print(torch.cuda.is_available())"`. |

### 6. Output atteso (checklist operatore)

Dopo la diagnosi, documentare in breve:

- **Stato processo**: PID assente / presente, CPU% approssimativa, ultima riga significativa nel log.
- **GPU**: util %, memoria, processo che occupa la GPU.
- **Fix applicato**: es. path dataset corretto, config ricopiata, `num_workers=0`, riavvio comando usato.

I comandi esatti di riavvio sono negli step sopra e in [`RUNBOOK_PHASE0.md`](RUNBOOK_PHASE0.md); questa sezione è solo **diagnostica idle**, non sostituisce il comando di avvio.

## Step 5 — Checkpoint

```bash
ls -lht /workspace/eurobot_baby/models/checkpoints_v3/
```

Attesi: directory tipo `step_*` o checkpoint ogni `save_every` (500 step nella config di riferimento).

## Step 6 — Stop manuale

```bash
kill "$(cat /workspace/eurobot_baby/logs/train_v3.pid)"
# oppure: pkill -f 'scripts/train.py'
```

## Step 7 — Test inference

```bash
LATEST=$(ls -td /workspace/eurobot_baby/models/checkpoints_v3/* 2>/dev/null | head -1)
pkill -f 'scripts/serve.py' || true
cd /workspace/eurobot_baby
source /root/eurobot_baby_venv/bin/activate
python scripts/serve.py \
  --checkpoint "$LATEST" \
  --host 0.0.0.0 \
  --port 8080 \
  --safe-mode
```

In altro terminale:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"which action"}]}'
```

## Step 8–9 — QA prompt e rotazione

Prompt suggeriti: `which action`, `for example`, `who are you`. Valuta ripetizioni e fallback.

Rotazione: tieni copie nominate `best_quality`, `last_stable`, `fallback_backup`; rimuovi checkpoint intermedi vecchi dopo aver copiato.

## Step 10 — Output fine run

Annota:

- path del miglior checkpoint (o ultimo stabile);
- loss finale dal log;
- 2–3 risposte di esempio dal curl/chat.
