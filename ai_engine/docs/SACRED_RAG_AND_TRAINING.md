# Sacred texts: RAG vs fine-tuning

Deep knowledge of sacred / philosophical corpora should come primarily from **retrieval** (FAISS + `text-embedding-3-small`), not from memorizing them in the causal LM. Heavy inclusion in `final_dataset_v3.jsonl` risks **overfitting** (verbatim tone, repetition).

## RAG (runtime)

1. Build a manifest JSONL: see [`data/sacred_sources.example.jsonl`](../data/sacred_sources.example.jsonl) (`file`, `source`, `topic` per line).
2. Run ingest (from repo root, `OPENAI_API_KEY` set):

   ```bash
   set PYTHONPATH=.
   python -m ai_engine.rag.ingest_sacred --manifest ai_engine/data/sacred_sources.jsonl --out-dir eurobot_baby/vector_db/sacred
   ```

3. On RunPod, target directory: `/workspace/eurobot_baby/vector_db/sacred/`.
4. **Baby `serve.py`**: set `EUROBOT_SACRED_RAG_PATH` to that directory; ensure monorepo root is on `PYTHONPATH` so `ai_engine` imports resolve.
5. **Gateway `ai_engine`**: set `AI_ENGINE_RAG_INDEX_PATH` to the same directory, `AI_ENGINE_RAG_EMBED_BACKEND=openai`, orchestrator enabled; philosophical queries route to RAG (see `philosophy_trigger.py`).

## Training (anti-overfit)

The merge pipeline [`training/merge_datasets.py`](../training/merge_datasets.py) mixes clean / hard-negative / sacred QA lines. To rely on RAG instead of LM memorization, **lower or zero the sacred weight**:

```bash
python ai_engine/training/merge_datasets.py --w-sacred 0 --w-clean 0.65 --w-hard 0.35 ...
```

Weights must sum to `1.0`. Rebuild `final_dataset_v3.jsonl` after changing weights.
