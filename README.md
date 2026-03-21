# eubot

Repository per il progetto **Eubot**. Contiene:

- **`brain-zero/`** — pipeline di training da zero per un mini-LLM (GPT-style): dati, tokenizer BPE, `train.py`, inferenza. Vedi [brain-zero/README.md](brain-zero/README.md).
- **`eubot-coder/`** — fine-tuning **QLoRA** su **Qwen2.5-Coder-7B-Instruct** (assistente coding + personalità). Vedi [eubot-coder/README.md](eubot-coder/README.md) e [docs/EUBOT_CODER_RUNPOD.md](docs/EUBOT_CODER_RUNPOD.md).

## Quick link

```bash
cd brain-zero
pip install -r requirements.txt
python scripts/test_baby.py
```

### Coding assistant (consigliato per uso reale)

```bash
cd eubot-coder
pip install -r requirements.txt
python scripts/prepare_data.py   # include italiano (OPUS en-it) di default
python scripts/finetune.py
```

Solo inglese: `python scripts/prepare_data.py --no-include_italian`

### brain-zero + italiano (LM da zero)

```bash
cd brain-zero
python scripts/download_data.py --mode large --wikipedia_it 50000
```
