# Eurobot Baby — training / retraining pipeline

All scripts assume **repo root** on `PYTHONPATH` or run from monorepo root.

## Schema JSONL (causal LM, aligned with `eurobot_baby/scripts/build_dataset.py`)

Each line:

```json
{"text": "User: ...\nAssistant: ..."}
```

Sacred Q/A lines may use:

```json
{"text": "Passage: ...\nQuestion: ...\nAnswer: ..."}
```

## Pipeline

1. **`build_clean_dataset.py`** — filter raw logs + original dataset → `ai_engine/data/clean_dataset_v2.jsonl`
2. **`hard_negatives.jsonl`** — curated (edit manually or extend)
3. **`build_sacred_dataset.py`** — chunk texts in `ai_engine/data/sacred_texts/*.txt` → `sacred_qa.jsonl`
4. **`merge_datasets.py`** — weighted merge (50% clean / 30% hard / 20% sacred) → `final_dataset_v3.jsonl`
5. Copy merged file to RunPod:

   ```bash
   cp ai_engine/data/final_dataset_v3.jsonl eurobot_baby/data/processed/train.jsonl
   # optional: hold-out val split
   ```

6. On RunPod: use [`eurobot_baby/configs/training_finetune_v3.yaml`](../eurobot_baby/configs/training_finetune_v3.yaml) with `scripts/train.py` (not in local mirror — sync from pod/Git).

### Copy merged data into Baby tree

From repo root:

```bash
# Optional: hold out ~1% for val (e.g. split with head/tail or Python)
cp ai_engine/data/final_dataset_v3.jsonl eurobot_baby/data/processed/train.jsonl
cp ai_engine/data/final_dataset_v3.jsonl eurobot_baby/data/processed/val.jsonl
# (better: real split — keep last N lines for val)
```

Then on RunPod: `python -u scripts/train.py` with config pointing at `data/processed/train.jsonl` (see `training_finetune_v3.yaml`).

### QA metrics

```bash
python -m ai_engine.tests.eval_quality
python -m ai_engine.tests.eval_quality --url https://eubot.seo.srl/api/baby/v1/chat/completions --samples 5
```

### Checkpoint naming (RunPod)

Use your real `train.py` convention (`step_<global_step>`). For experiments, tag runs in logs:

- after clean merge only → note as `clean_v1` / `clean_v2` in training log;
- after adding sacred mix → `sacred_mix`.

Compare checkpoints with `eval_quality` + `run_chat_preview` (fallback rate, diversity).

## Anti-template

Rows matching server fallback templates (e.g. `Let me give you a simple and clear answer`) are **dropped** in `build_clean_dataset.py` and optionally in `merge_datasets.py`.
