# RunPod — sintesi per riprendere (snapshot training + RAG)

Ultimo aggiornamento documentazione: **2026-03-25** (valori sul pod vanno verificati con `tail` / `nvidia-smi` al momento della ripresa).

## Quanto manca al training (ordine di grandezza)

- **Target:** `max_steps: 600000` in `/workspace/eurobot_baby/configs/training.yaml`
- **Snapshot tipico (2026-03-25 ~16:40 UTC):** progress bar ~`512200 / 600000` (~85%), **~87800 step** rimanenti, **~6–7 ore** alla barra tqdm (~`3.8 it/s`).
- **Formula rapida:** `(600000 - step_corrente) / it_s` secondi (leggi `step` e `it/s` da `tail` sul log).

Log training: `/workspace/eurobot_baby/logs/train_resume.log`  
Comando: `tail -f /workspace/eurobot_baby/logs/train_resume.log`

## Cosa sta facendo il pod

| Elemento | Dettaglio |
|----------|-----------|
| Training | `python -u scripts/train.py --resume models/checkpoints/step_505000` (PID va verificato con `pgrep -af train.py`) |
| Obiettivo step | 600000 (oltre il checkpoint iniziale 505000) |
| Dataset train | `data/processed/train.jsonl` (come in `training.yaml`) |
| Checkpoint dir | `models/checkpoints/step_*` (salvataggi periodici da `save_every` nel YAML) |
| Serve | Di solito **spento** durante il training (stessa GPU) |

## Dopo la fine del training

1. **Ultimo checkpoint:** `ls -td /workspace/eurobot_baby/models/checkpoints/step_* | head -1`
2. **Riavvio API:** `python scripts/serve.py --checkpoint <ultimo> --host 0.0.0.0 --port 8080` (vedi [`eurobot_baby/docs/BEST_PRACTICES_TRAINING.md`](../eurobot_baby/docs/BEST_PRACTICES_TRAINING.md))
3. **RAG sacro:** `EUROBOT_SACRED_RAG_PATH` + ingest da [`ai_engine/rag/ingest_sacred.py`](../ai_engine/rag/ingest_sacred.py)
4. **RAG espansione (HF):** JSONL già in `/workspace/ai_engine/data/rag_expansion/` — manca **ingest FAISS** + env `EUROBOT_RAG_EXPANSION_PATH` (vedi [`DATASET_EXPANSION_RAG_AND_TRAINING.md`](DATASET_EXPANSION_RAG_AND_TRAINING.md))
5. **Training successivo:** merge `ai_engine/data/expansion_training/lm_train_merged.jsonl` con pipeline `merge_datasets.py` e nuovo `max_steps`

## Script utili (repo)

- Watchdog retry: [`tools/runpod_baby_train_watchdog.sh`](../tools/runpod_baby_train_watchdog.sh)
- Pipeline RAG espansione: [`tools/runpod_rag_expansion_pipeline.sh`](../tools/runpod_rag_expansion_pipeline.sh)
- Run training v3 dataset: [`eurobot_baby/docs/RUNPOD_TRAIN_V3_24H.md`](../eurobot_baby/docs/RUNPOD_TRAIN_V3_24H.md)

## Allineamento codice

Tutto ciò che segue è versionato in Git nel monorepo `eubot` (non sul pod finché non fai `git pull` / deploy).
