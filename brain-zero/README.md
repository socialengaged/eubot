# brain-zero

Pipeline minimale ma reale per addestrare un **mini-LLM** (GPT-style) da zero: dati, tokenizer BPE, training PyTorch, checkpoint, inferenza.

Non usa pesi pretrained: solo architettura HuggingFace `GPT2LMHeadModel` inizializzata a caso.

## Requisiti

- Python 3.10+
- GPU CUDA consigliata (RunPod / locale)
- ~2–5 GB disco per dataset di prova

## Setup

```bash
cd brain-zero
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ordine consigliato

1. **`python scripts/test_baby.py`** — verifica end-to-end in pochi minuti (CPU ok, GPU più veloce).
2. **`python scripts/download_data.py`** — scarica WikiText-103 (subset configurabile).
3. **`python scripts/build_dataset.py`** — pulisce e produce `data/processed/*.jsonl`.
4. **`python scripts/train_tokenizer.py`** — addestra BPE → `models/tokenizer/`.
5. **`python scripts/train.py`** — training vero → `models/checkpoints/`.
6. **`python scripts/inference.py --prompt "Hello"`** — generazione.

Dettagli: **[PROJECT_SPEC.md](PROJECT_SPEC.md)**.
