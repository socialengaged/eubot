# brain-zero — specifica tecnica (documento vivente)

**Stato:** codice completo, testato CPU locale. Prossimo: training GPU su RunPod.

---

## Obiettivo

Modello **causale** tipo GPT-2, pesi **non** pretrained, addestrato da zero su dati open.
Estendibile a corpus custom (personalita Eubot) e poi italiano.

---

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

| Profilo | Params | Uso |
|---------|--------|-----|
| baby | ~10M | Test rapido pipeline (minuti) |
| small | ~125M | Primo modello semi-serio (ore) |
| medium | ~350M | Modello piu capace (ore-giorni) |

---

## Roadmap operativa

### Fase 1 — Baby run (EN, test pipeline) ← PROSSIMO STEP
- [x] Codice pipeline completo
- [x] `test_baby.py` OK su CPU locale
- [ ] Eseguire sul Pod RunPod con GPU (vedi sezione RunPod sotto)
- [ ] Verificare che `inference.py` genera testo inglese coerente

### Fase 2 — Small run (EN, primo modello semi-serio)
- [ ] Cambiare profilo in `model.yaml` a `small`
- [ ] Training completo su WikiText-103 (~125M params)
- [ ] Valutare qualita output (loss, campioni generati)

### Fase 3 — Aggiunta italiano
- [ ] Aggiornare `download_data.py` per scaricare Wikipedia IT (HF `wikipedia`, lang `it`)
- [ ] Opzione: aggiungere OSCAR o MC4 italiano come fonte aggiuntiva
- [ ] Riaddrestrare tokenizer BPE su dati EN+IT combinati
- [ ] Rilanciare training su dataset bilingue

### Fase 4 — Personalita Eubot
- [ ] Creare dataset custom (testi di Eugenio, stile, FAQ, istruzioni)
- [ ] Fine-tune su checkpoint esistente (`train.py --finetune --checkpoint ...`)
- [ ] Valutare coerenza personalita

### Fase 5 — Scaling e produzione
- [ ] Profilo `medium` (350M) o piu grande se GPU lo consente
- [ ] Dataset piu ampi (OpenWebText, The Pile subset)
- [ ] Log estesi (WandB opzionale)
- [ ] API inference (FastAPI)
- [ ] Integrazione con frontend Eubot (chatbot web)

---

## RunPod — comandi per Fase 1

Apri il terminale web RunPod o `ssh eubot`, poi:

```bash
# 1. Clone e setup
git clone https://github.com/socialengaged/eubot.git
cd eubot/brain-zero
pip install -r requirements.txt

# 2. Test pipeline (1-2 min, conferma che PyTorch+CUDA funzionano)
python scripts/test_baby.py

# 3. Scarica dati WikiText-103 inglese
python scripts/download_data.py

# 4. Pulisci e prepara JSONL
python scripts/build_dataset.py

# 5. Addestra tokenizer BPE
python scripts/train_tokenizer.py

# 6. TRAINING baby (pochi minuti su GPU)
python scripts/train.py

# 7. Genera testo dal modello addestrato
python scripts/inference.py --checkpoint models/checkpoints/step_5000 --prompt "The world"
```

Dopo step 7, il modello "parla" (in inglese, qualita baby).

---

## Server RunPod

- **Pod ID:** rwj7ag6zz2entl
- **Terminale web:** https://rwj7ag6zz2entl-19123.proxy.runpod.net/603oeo45uinrzrvky38di2xtdqj575a7/
- **SSH:** `ssh eubot` (alias in `~/.ssh/config`, chiave `eubot_ed25519`)

---

## Changelog

- **2026-03-21:** Pipeline completa: test_baby.py, download, build, tokenizer, train, inference. Test locale CPU OK. Roadmap aggiornata con fasi IT e personalita.
