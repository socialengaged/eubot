# brain-zero — specifica tecnica (documento vivente)

**Stato:** pipeline implementata; training su GPU da lanciare sul Pod.

## Obiettivo

- Modello **causale** tipo GPT-2, pesi **non** pretrained.
- Dati: WikiText (base) + estendibile a corpus custom (fase personalità Eubot).
- Tokenizer **BPE** addestrato sui dati di progetto.

## Struttura

| Path | Ruolo |
|------|--------|
| `configs/model.yaml` | Profili `baby` / `small` / `medium` |
| `configs/training.yaml` | LR, batch, checkpoint, path |
| `data/raw/` | Download HF (gitignored) |
| `data/processed/` | JSONL train/val (righe con campo `text`) |
| `models/tokenizer/` | Tokenizer salvato |
| `models/checkpoints/` | Checkpoint `step_*` |

## Flusso dati

1. `download_data.py` → raw testuale
2. `build_dataset.py` → chunk di testo → JSONL
3. `train_tokenizer.py` → BPE
4. `train.py` → loss, checkpoint
5. `inference.py` → generazione

## Profili modello

| Profilo | Uso |
|---------|-----|
| baby | Test rapido pipeline |
| small | ~125M params (semi-serio) |
| medium | ~350M params |

## Prossimi passi

- [ ] Fine-tune su dataset personalità (testi proprietari)
- [ ] Log estesi (WandB opzionale)
- [ ] API inference (FastAPI) fuori da questo repo

## RunPod

```bash
git clone https://github.com/socialengaged/eubot.git && cd eubot/brain-zero
pip install -r requirements.txt
python scripts/test_baby.py
python scripts/download_data.py
python scripts/build_dataset.py
python scripts/train_tokenizer.py
python scripts/train.py --max_train_blocks 2000   # prova breve
python scripts/inference.py --checkpoint models/checkpoints/step_1000 --prompt "The"
```

## Changelog

- **2026-03-21:** Prima versione pipeline + `test_baby.py` (test locale CPU OK).
