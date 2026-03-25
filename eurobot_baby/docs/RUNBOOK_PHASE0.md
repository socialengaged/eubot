# EUROBOT BABY — Runbook Fase 0 (fix TRAIN_PARTS)

**Obiettivo:** includere in `train.jsonl` i file già presenti in `data/raw/` ma assenti da `TRAIN_PARTS`:
- `15_hf_vivechan.txt`
- `16_hf_sep_philosophy.txt`
- `17_wikipedia_esoteric_filtered.txt`

**Dove:** pod GPU RunPod (path tipico `/workspace/eurobot_baby/`).

## Prerequisiti

1. Training corrente **terminato** (o interrompere consapevolmente se serve solo rebuild).
2. Backup di `data/processed/train.jsonl` e `val.jsonl` (opzionale ma consigliato).

## Staging (`raw_staging`) prima del rebuild

Se durante il training hai accumulato nuovi testi in `data/raw_staging` (merge da scraping senza toccare `data/raw`), prima di `build_dataset.py`:

1. Fermare il training e verificare checkpoint.
2. Promuovere gli staging verso i raw attivi, ad esempio sul pod:

```bash
bash /workspace/eurobot_baby/tools/scraping/promote_raw_staging.sh
# oppure --dry-run per vedere cosa verrebbe copiato
```

3. Proseguire con la sezione **2. Rebuild dataset** qui sotto.

Dettagli e parallelismo sicuro: [`tools/scraping/SETUP_SCRAPER.md`](../tools/scraping/SETUP_SCRAPER.md).

## 1. Sincronizzare `build_dataset.py` dal repo

Dal PC di sviluppo (repo `eubot`):

```bash
scp -P 17100 -i ~/.ssh/eubot_ed25519 \
  eurobot_baby/scripts/build_dataset.py \
  root@213.173.103.83:/workspace/eurobot_baby/scripts/build_dataset.py
```

Oppure applicare manualmente la stessa modifica a `TRAIN_PARTS` sul pod.

## 2. Rebuild dataset

```bash
ssh -p 17100 -i ~/.ssh/eubot_ed25519 root@213.173.103.83
cd /workspace/eurobot_baby
source /root/eurobot_baby_venv/bin/activate
python scripts/build_dataset.py
```

Annotare l’output, es. `train: N chunks`.

## 3. Calcolo `max_steps` (2 epoche)

Con `batch_size=4`, `gradient_accumulation_steps=4` → **16 blocchi per step** (optimizer step).

```text
steps_per_epoch = ceil(N_chunks / 16)
max_steps_2_epochs = steps_per_epoch * 2
```

Esempio: se `N = 330000` → `ceil(330000/16) = 20625` → `max_steps = 41250`.

## 4. Aggiornare `configs/training.yaml`

Impostare almeno:

- `max_steps: <valore calcolato>`
- `save_every`, `eval_every` proporzionati (es. 3000 / 1500)

## 5. Riprendere il training

Dall’ultimo checkpoint disponibile (es. `step_24610` se il run precedente è finito lì):

```bash
cd /workspace/eurobot_baby
nohup env PYTHONUNBUFFERED=1 HF_HOME=/root/hf_home \
  /root/eurobot_baby_venv/bin/python -u scripts/train.py \
  --resume models/checkpoints/step_24610 \
  >> /root/training_baby_phase0.log 2>&1 &
```

**Nota:** se il nuovo `max_steps` è **minore** dello step del checkpoint, `train.py` esce con “Nothing to do”. In quel caso:
- o aumentare `max_steps` oltre lo step corrente,
- o ripartire da checkpoint precedente / da zero (non consigliato senza backup).

## 6. Verifica

```bash
tail -f /root/training_baby_phase0.log
pgrep -af train.py
```

La barra tqdm deve mostrare `global_step/total` coerente con il nuovo `max_steps`.

## File opzionali futuri

Il `build_dataset.py` nel repo include anche nomi placeholder (`08_esoteric_expanded.txt`, `20_physics_corpus.txt`, …). Se **non** esistono in `data/raw/`, vengono saltati senza errore. Quando i file saranno copiati sul pod, un nuovo `build_dataset.py` rigenererà il JSONL includendoli.
