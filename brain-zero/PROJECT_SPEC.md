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

### Fase 1 — Baby run (EN, test pipeline) ✓
- [x] Codice pipeline completo
- [x] `test_baby.py` OK su CPU locale
- [x] Training baby su RunPod GPU (5000 step, ~2 min)
- [x] Fix tokenizer ByteLevel decoder (bug Ġ)

### Fase 2 — Small run (EN, training massivo) ← PROSSIMO STEP
- [x] `--resume` per continuare training da checkpoint
- [x] `download_data.py --mode large` (WikiText + OpenWebText 200k docs)
- [x] `training_phase2.yaml` (125M params, 50k step, seq 1024)
- [x] `run_phase2.sh` script unico
- [ ] Eseguire `run_phase2.sh` su RunPod
- [ ] Valutare qualita output (loss < 3.5 target, campioni generati)

### Fase 3 — Aggiunta italiano
- [x] `download_data.py --wikipedia_it N` — Wikipedia IT streaming (`20220301.it`)
- [ ] Opzione futura: OSCAR / MC4 italiano (corpus piu grande)
- [ ] Dopo aver aggiunto testo IT: riaddrestrare tokenizer BPE su `train.jsonl` misto
- [ ] Rilanciare `train.py` su dataset bilingue (EN+IT)

**Eubot Coding (QLoRA, consigliato per chat IT/EN):** in `eubot-coder/`, `prepare_data.py` include per default:
- OPUS-100 `en-it` (traduzione bidirezionale come istruzioni)
- sottoinsieme CodeFeedback con `lang=it` se la colonna esiste  
Vedi [eubot-coder/README.md](../eubot-coder/README.md) sezione Lingua.

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

## RunPod — comandi

### Fase 2 (training massivo)

```bash
cd /eubot/eubot && git pull && cd brain-zero
bash scripts/run_phase2.sh
```

Tempo: ~1-3 ore. Il modello cresce a 125M params su ~500 MB di testo.
Checkpoint salvati ogni 5000 step. Se interrotto, riprendere con:

```bash
python scripts/train.py --profile small --training_config training_phase2.yaml \
  --resume models/checkpoints/step_XXXXX
```

Dopo il training: `python scripts/chat.py`

---

## Server RunPod

- **Pod ID:** rwj7ag6zz2entl
- **Terminale web:** https://rwj7ag6zz2entl-19123.proxy.runpod.net/603oeo45uinrzrvky38di2xtdqj575a7/
- **SSH:** `ssh eubot` (alias in `~/.ssh/config`, chiave `eubot_ed25519`)

---

## Changelog

- **2026-03-21:** Pipeline completa. Baby run OK su RunPod. Fix tokenizer. Phase 2 pronta (125M, 50k step, OpenWebText).
