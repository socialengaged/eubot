# Eurobot Baby — dataset & training (repo mirror)

Questa cartella replica la struttura usata sul pod GPU (`/workspace/eurobot_baby/`).

## Contenuti

| Path | Descrizione |
|------|-------------|
| `scripts/build_dataset.py` | Merge di `data/raw/*.txt` in `data/processed/train.jsonl` — **include Fase 0** (file 15/16/17) e placeholder per fasi successive |
| `scripts/rebuild_dataset.py` | Wrapper verso `build_dataset.py` |
| `scripts/serve_v2_extension.py` | Estende `scripts/serve.py` con `POST /v2/chat` (orchestrator `ai_engine`); sul pod: `PYTHONPATH` al monorepo con `ai_engine/`, porta default **8081** (tunnel locale **18082** in `eubot-coder/local_server.py`) |
| `docs/RUNBOOK_PHASE0.md` | Comandi esatti per pod: rebuild + calcolo `max_steps` + resume training |
| `docs/BEST_PRACTICES_TRAINING.md` | Checklist unica: pre-run, config, avvio training, monitoraggio, export, **ordine operativo sul pod** |
| (repo `eubot`) `tools/runpod_pod_precheck.sh` | Sul pod: GPU, disco, checkpoint, log (solo lettura) |
| (repo `eubot`) `tools/runpod_pod_orchestrate.sh` | Sul pod: precheck + `MODE=train` o `MODE=serve` |
| `tools/scraping/` | Script per macchina dedicata (Fasi 1–6, 10) |

## Fase 0 (fix immediato)

1. Copiare `scripts/build_dataset.py` sul pod (vedi runbook).
2. `python scripts/build_dataset.py`
3. Aggiornare `configs/training.yaml` con `max_steps = ceil(N_chunks/16)*2`
4. `python scripts/train.py --resume models/checkpoints/step_<ULTIMO>`

## Stato implementazione (pod RunPod) — agg. 2026-03-24

- **Fase 0 + espansione dataset (piano §3–11, tranne 4–5–9 opzionali)**: `train.jsonl` con **~1 155 955 chunks** (≈1,16M); `TRAIN_PARTS` con wiki, esoterico (fasi 1–3), STEM (fasi 6–8, 10).
- **Checkpoint raggiunto**: `step_286744` (run da `214496` → `286744`, log `training_baby_epoch_next.log`).
- **Training in corso**: resume da **`step_286744`**, **`max_steps: 431240`** (+**2 epoche**, **144 496** step rimanenti sullo stesso `train.jsonl`), log **`/root/training_baby_smarter.log`**. Obiettivo checkpoint finale **`step_431240`**. Script pod: `tools/runpod_baby_training_resume.sh` (upload + `bash` sul RunPod).
- **Produzione** (`eubot.seo.srl`): tunnel OVH → RunPod TCP; Baby API `scripts/serve.py` su `:8080`. **Durante `train.py` il serve è stato fermato** per liberare la GPU: a training finito rilanciare serve con `--checkpoint` sull’ultimo `step_*`.
- **Scraping**: script in `tools/scraping/`; output opzionale in `/workspace/eurobot_scraping_run/`.
- **Ancora da piano (dati nuovi)**: Fasi **4** (Esoteric Archives / Hermetic), **5** (Internet Archive), **9** (peS2o) — vedi `docs/PIANO_DATASET_EXPANSION_v1.md`.

## Dataset aggiuntivi (astrologia, massoneria, informatica)

- **Astrologia** e **massoneria**: sezioni `astrology` e `freemasonry` in `download_sacred_texts_deep.py` → dopo il merge compaiono come **`43_astrology_corpus.txt`** e **`44_masonry_corpus.txt`** (non più solo dentro `08_esoteric_expanded.txt`, per dare peso dedicato).
- **Astronomia**: già `40_*`, `41_*`, `42_*` (HF + arXiv astro-ph).
- **Informatica / sviluppo**: **`70_informatics_corpus.txt`** (arXiv `cs.*`, campione aumentabile con `EUROBOT_ARXIV_CS_MAX`) + **`71_gutenberg_computing.txt`** (Boole, Babbage, Lovelace/Menabrea, Venn — `download_gutenberg_computing.py`).

Dopo merge + copia in `data/raw/`, **`python scripts/build_dataset.py`** e ricalcola **`max_steps`** (vedi `docs/BEST_PRACTICES_TRAINING.md`).

## Scraping (nuova macchina)

```bash
cd eurobot_baby/tools/scraping
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-scraping.txt

# Esempi
python download_gnosis_retry.py --delay 5
python download_gutenberg_esoteric.py
python download_sacred_texts_deep.py   # lungo: usare tmux/nohup
python download_physics_hf.py
python download_stem_gutenberg_science.py

# JSONL -> TXT per concat in un unico blob da copiare in data/raw/
python jsonl_to_txt.py output/physics/flappingairplanes__physics-textbooks-gpt2.jsonl ../output/physics_textbooks.txt
```

Copiare i `.txt` risultanti sul pod in `data/raw/` con i nomi attesi da `TRAIN_PARTS` (es. `08_esoteric_expanded.txt`, `20_physics_corpus.txt`, …).

## Pod spento o SSH non risponde

Se il bot «non parla» e da PC `ssh` fallisce (**connection refused**), la GPU RunPod non è raggiungibile: **accendi il pod** nella dashboard RunPod e attendi SSH (porta tipica in `docs/EUROBOT_BABY_RUNPOD.md`). Su Windows puoi verificare la porta con `powershell -File tools/check_runpod_ssh.ps1` dalla root del repo `eubot`.

Quando il pod è online, per **riprendere il training dall’ultimo checkpoint** fino a `max_steps` (default `431240`, allineato al run documentato):

1. Copia sul pod lo script [`tools/runpod_baby_training_auto_resume.sh`](../tools/runpod_baby_training_auto_resume.sh) (es. in `/root/`) oppure incollane il contenuto.
2. Sul pod: `bash /root/runpod_baby_training_auto_resume.sh`  
   Opzionale: `MAX_STEPS=450000 LOG_FILE=/root/training_baby_run.log bash ...` se serve un target diverso.
3. Monitor: `tail -f /root/training_baby_auto.log` e `pgrep -af train.py`.
4. A training finito, rilancia **`scripts/serve.py`** sull’ultimo `step_*` (la GPU non può servire chat e train insieme). Vedi [`docs/BEST_PRACTICES_TRAINING.md`](docs/BEST_PRACTICES_TRAINING.md).

Lo script ferma `serve.py`, fa backup di `configs/training.yaml` e fa `--resume` sull’**ultima** cartella `models/checkpoints/step_*`.

## Documentazione piano completo

Vedi `docs/PIANO_DATASET_EXPANSION_v1.md` nella root del repo `eubot`.

**Riferimento tecnico unico (dataset, `TRAIN_PARTS`, import su altra macchina):**  
[`../docs/EUROBOT_BABY_DATASET_TECHNICAL_REFERENCE.md`](../docs/EUROBOT_BABY_DATASET_TECHNICAL_REFERENCE.md).
