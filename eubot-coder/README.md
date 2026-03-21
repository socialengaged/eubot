# Eubot Coding Assistant

Fine-tune **Qwen2.5-Coder-7B-Instruct** with **QLoRA** (4-bit + LoRA) on open instruction datasets + optional synthetic personality data.

## Quick start (RunPod / Linux GPU)

```bash
cd eubot-coder
pip install -r requirements.txt

# 1) Build train/val JSONL from HuggingFace datasets
python scripts/prepare_data.py

# Optional: merge GPT-generated JSONL (same schema as train.jsonl)
# python scripts/prepare_data.py --merge_style data/processed/style_gpt.jsonl

# 2) Train adapter (~1–3 h depending on GPU)
python scripts/finetune.py

# 3) Merge LoRA into full weights (optional, for simpler deployment)
python scripts/merge_adapter.py

# 4) Chat
python scripts/chat.py

# Or HTTP API (OpenAI-style)
python scripts/serve.py --host 0.0.0.0 --port 8080
```

## Personality / synthetic data

```bash
python scripts/generate_style.py --print_jsonl_example
```

Use the printed master prompt in ChatGPT/Claude to produce JSONL lines; save as `data/processed/style_gpt.jsonl` and re-run `prepare_data.py --merge_style ...`.

## Lingua (italiano + inglese)

- **`prepare_data.py`** di default aggiunge dati per l’italiano:
  - **OPUS-100 `en-it`**: coppie di traduzione come istruzioni (streaming, leggero in RAM).
  - **CodeFeedback `lang=it`** (se la colonna `lang` esiste nel dataset).
- Per **solo inglese** (come prima):  
  `python scripts/prepare_data.py --no-include_italian`
- Regola quantità IT:  
  `--max_opus_pairs 8000` , `--max_codefeedback_it 3000`

Il **system prompt** in `finetune.yaml` chiede di rispondere nella lingua dell’utente (IT o EN).

Nessun pacchetto extra: tutto via `datasets` (HuggingFace).

## Config

- [`configs/finetune.yaml`](configs/finetune.yaml) — model id, LoRA targets, LR, batch, paths.

## Integration with existing Eubot (Node)

Point `AI_BASE_URL` to `http://<pod-ip>:8080` and map the OpenAI-compatible `POST /v1/chat/completions` (same shape as OpenAI).

## Notes

- Requires CUDA GPU with enough VRAM for 7B QLoRA (typically 16 GB+ comfortable).
- Set `HF_TOKEN` if HuggingFace Hub rate-limits downloads.
