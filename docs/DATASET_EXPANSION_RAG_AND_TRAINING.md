# Espansione dataset: RAG conversazionale / medico + training futuro

Documento di sintesi per **riprendere dopo** il training LM attuale sul pod. Per **tempo rimanente al training**, checkpoint e prossimi passi vedi anche [`EUBOT_RUNPOD_RESUME.md`](EUBOT_RUNPOD_RESUME.md). Non interferisce con `scripts/train.py` in esecuzione: download e ingest RAG usano **CPU/RAM/disco** e API OpenAI (embedding); convivono con training GPU se c’è spazio e banda.

## Obiettivi

1. **RAG** — Indice FAISS separato da quello “sacro”, con testi da Hugging Face (conversational, medicina, letteratura biomedica, dialoghi quotidiani come proxy psicologia conversazionale).
2. **Training futuro** — JSONL `User:/Assistant:` in `ai_engine/data/expansion_training/` da mischiare in `merge_datasets.py` quando il run corrente è finito.

## Percorsi nel repo

| Percorso | Contenuto |
|----------|-----------|
| `ai_engine/data/rag_expansion/` | JSONL per sorgente + `combined_rag.jsonl` (merge) |
| `ai_engine/data/expansion_training/` | `*_lm.jsonl` + `lm_train_merged.jsonl` |
| `eurobot_baby/vector_db/rag_expansion/` | `index.faiss` + `metadata.pkl` (dopo ingest) |

Se sul pod hai **solo** `/workspace/eurobot_baby` senza cartella `ai_engine` accanto, imposta prima del download:

`export RAG_EXPANSION_OUT=/workspace/eurobot_baby/data/rag_expansion` e  
`export EXPANSION_TRAIN_OUT=/workspace/eurobot_baby/data/expansion_training`  

e copia lì anche lo script Python, oppure clona il **monorepo completo** sotto `/workspace/eubot`.

## Dataset scaricati (script)

Script: [`eurobot_baby/tools/scraping/download_rag_expansion_hf.py`](../eurobot_baby/tools/scraping/download_rag_expansion_hf.py)

| Fonte Hub | Uso | Topic indicativo |
|-----------|-----|------------------|
| `OpenAssistant/oasst1` | Messaggi open assistant | `conversational` |
| `databricks/databricks-dolly-15k` | Istruzioni/risposte | `conversational_instruction` |
| `medmcqa` (en) | MCQ medicina (anatomia se subject contiene “anatom”) | `anatomy` / `medicine_general` |
| `pubmed_qa` (`pqa_labeled`) | contesto + Q + risposta lunga | `medicine_literature` |
| `daily_dialog` | Dialoghi quotidiani | `psychology_conversational` |

**Filosofia pura:** non inclusa come dataset HF separato qui; resta coperta da testi sacri / Gutenberg + [`ingest_sacred`](../ai_engine/rag/ingest_sacred.py) e da eventuali testi manuali in `ai_engine/data/sacred_texts/`.

**Licenze:** verificare sempre le schede Hugging Face prima dell’uso commerciale (oasst, dolly, medmcqa, pubmed_qa, daily_dialog hanno licenze diverse).

**Dipendenze:** `pip install datasets` (aggiunto in `requirements-ai-engine.txt`).

### Comandi (da root monorepo `eubot`)

```bash
pip install -r requirements-ai-engine.txt
set PYTHONPATH=.
python eurobot_baby/tools/scraping/download_rag_expansion_hf.py --max-per-dataset 5000
```

Opzioni: `--skip-oasst`, `--skip-dolly`, `--merge-only`, ecc. (vedi `--help`).

### Ingest embedding (OpenAI)

Richiede `OPENAI_API_KEY` (stesso modello `text-embedding-3-small` degli altri RAG).

```bash
set PYTHONPATH=.
python -m ai_engine.rag.ingest_multidomain_jsonl ^
  --jsonl ai_engine/data/rag_expansion/combined_rag.jsonl ^
  --out-dir eurobot_baby/vector_db/rag_expansion
```

Sul pod: `/workspace/eurobot_baby/vector_db/rag_expansion` se il repo è montato con `ai_engine` accessibile.

### Serve (Baby)

Variabili:

- `EUROBOT_RAG_EXPANSION_PATH=/workspace/eurobot_baby/vector_db/rag_expansion`
- `OPENAI_API_KEY=...`
- Root monorepo su `PYTHONPATH` per import `ai_engine`

Il trigger dominio è in [`reference_domains_trigger.py`](../ai_engine/orchestrator/reference_domains_trigger.py) (medicina, psichiatria, psicologia, anatomia, ecc.). Sacro resta su `EUROBOT_SACRED_RAG_PATH` + trigger filosofico.

## Training LM successivo

1. Unire (o pesare) `ai_engine/data/expansion_training/lm_train_merged.jsonl` con `clean_dataset_v2` / hard negatives in [`merge_datasets.py`](../ai_engine/training/merge_datasets.py) (nuove percentuali o file dedicato).
2. Rigenerare `final_dataset_v4.jsonl` (o nome concordato) e aggiornare `data_train` / `max_steps` sul pod.
3. Riprendere con `--resume` dall’ultimo `step_*` come da [`BEST_PRACTICES_TRAINING.md`](../eurobot_baby/docs/BEST_PRACTICES_TRAINING.md).

## Pipeline shell (opzionale)

[`tools/runpod_rag_expansion_pipeline.sh`](../tools/runpod_rag_expansion_pipeline.sh) — download + merge + ingest (richiede env e path pod adattati).

## Stato al 2026-03-25

- Training GPU precedente: checkpoint `step_505000`, `max_steps` portato a `600000` sul pod (contesto storico).
- **Download sul pod (run completato):** log `/root/rag_expansion_download.log` — generati `oasst1`, `dolly`, `pubmed_qa`, merge `combined_rag.jsonl` (~7000 righe) e `lm_train_merged.jsonl` sotto `/workspace/ai_engine/data/`. `medmcqa` richiede config `default` (fix nello script repo); `daily_dialog` può essere ignorato se Hub non serve più lo script legacy.
- **Ingest FAISS:** da eseguire quando serve (non durante picco training se vuoi risparmiare RAM): copiare `ai_engine/rag/ingest_multidomain_jsonl.py` + dipendenze, poi `python -m ai_engine.rag.ingest_multidomain_jsonl` con `OPENAI_API_KEY`.
- Questo documento descrive **preparazione dati e RAG**; aggiornare la sezione “Stato” a fine run.
