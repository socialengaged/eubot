# ai_engine (Eurobot Baby)

Modular orchestration layer: routing, HTTP client to **Baby** `serve.py`, optional **FAISS RAG** (CPU), LLM stub/OpenAI.

- Does **not** modify `eurobot_baby` training or the GitHub `serve.py`.
- Configure via environment variables (see `config/settings.py`).

## Install

```bash
cd /path/to/eubot
pip install -r requirements-ai-engine.txt
```

## Test

```bash
# from repo root
python -m pytest ai_engine/tests -v
```

## Manual smoke (Baby running on :8080)

```bash
set PYTHONPATH=.
python -c "from ai_engine import call_baby; print(call_baby([{'role':'user','content':'ciao'}]))"
```

## Env flags

| Variable | Default | Meaning |
|----------|---------|---------|
| `BABY_BASE_URL` | `http://127.0.0.1:8080` | Baby OpenAI-compatible API |
| `AI_ENGINE_USE_ORCHESTRATOR` | `0` | `1` enables heuristic routing |
| `AI_ENGINE_DEFAULT_ROUTE` | `baby` | When orchestrator off |
| `AI_ENGINE_LLM_STUB_MODE` | `1` | `0` + `OPENAI_API_KEY` → real OpenAI call |
| `AI_ENGINE_RAG_INDEX_PATH` | (empty) | Directory with `index.faiss` + `metadata.pkl` from ingest |
| `AI_ENGINE_RAG_USE_REAL` | `1` | `0` forces stub RAG (no FAISS) |
| `AI_ENGINE_RAG_EMBED_BACKEND` | `local` | `local` = sentence-transformers; `openai` = OpenAI embeddings |
| `AI_ENGINE_RAG_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model id |
| `AI_ENGINE_RAG_ANSWER_TARGET` | `baby` | After retrieval: `baby` or `llm` |

## RAG ingest (CPU)

From repo root, with `eurobot_baby/data/processed/train.jsonl` present:

```bash
set PYTHONPATH=.
python -m ai_engine.rag.ingest --jsonl eurobot_baby/data/processed/train.jsonl --out-dir eurobot_baby/data/processed/rag_index
```

Then set `AI_ENGINE_RAG_INDEX_PATH` to that directory and enable orchestrator routing to RAG.

FastAPI `/v2/chat` — see `eurobot_baby/scripts/serve_v2_extension.py` on the pod.

## RAG espansione (HF conversazionale / medico)

Download JSONL, merge, ingest: vedi [`docs/DATASET_EXPANSION_RAG_AND_TRAINING.md`](../docs/DATASET_EXPANSION_RAG_AND_TRAINING.md). Modulo ingest: `python -m ai_engine.rag.ingest_multidomain_jsonl`. Su Baby `serve.py`: `EUROBOT_RAG_EXPANSION_PATH`.
