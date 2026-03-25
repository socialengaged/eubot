# Export e migrazione da RunPod (Eurobot Baby)

**Stato verificabile (checkpoint corrente sul pod 2026-03-24):** training completato fino a **`step_214496`**, checkpoint in  
`models/checkpoints/step_214496/`. (Run storico precedente: `step_144496` nei bundle Tier A scaricati prima del run notturno.)

Se non vedi processi `train.py` attivi e la GPU è idle: **è normale** — il processo è uscito dopo `Training finished.`

Per **bundle automatici sul pod** (Baby + Sage + cache HF), test **offline** e comandi `scp`/`rsync`, vedi anche:  
[`docs/OFFLINE_EXPORT_BABY_AND_SAGE.md`](../../docs/OFFLINE_EXPORT_BABY_AND_SAGE.md) e lo script `tools/export_runpod_bundles.sh`.

---

## Cosa esportare (in ordine di priorità)

### Tier A — Minimo per ripartire il modello (senza rifare training)

| Percorso sul pod | ~Dimensione | Contenuto |
|------------------|-------------|-----------|
| `models/checkpoints/step_214496/` (o ultimo `step_*`) | ~380 MB | Pesi + tokenizer nella cartella checkpoint |
| `models/tokenizer/` | ~3 MB | Tokenizer “ufficiale” del repo (se preferisci copiarlo separatamente) |
| `configs/training.yaml` | pochi KB | Iperparametri e path |

**Totale ~380 MB** — sufficiente per inference / fine-tuning successivo con `--resume`.

### Tier B — Dataset riproducibile (consigliato)

| Percorso | ~Dimensione | Nota |
|----------|-------------|------|
| `data/processed/train.jsonl` | ~4.3 GB | Corpus tokenizzabile usato dall’ultimo training |
| `data/processed/val.jsonl` | piccolo | Validazione |

Senza questo, su un nuovo server puoi comunque **rigenerare** `train.jsonl` da `data/raw/` con `scripts/build_dataset.py`, se copi il Tier C.

### Tier C — Raw (opzionale se hai già `train.jsonl`)

| Percorso | ~Dimensione |
|----------|-------------|
| `data/raw/*.txt` | ~4 GB |

Utile se vuoi **cambiare chunking** o **aggiungere file** e rifare `build_dataset.py`.

### Tier D — Scraping intermedio (opzionale)

| Percorso | ~Dimensione |
|----------|-------------|
| `/workspace/eurobot_scraping_run/` | ~2 GB |

Output degli script di download (gutenberg, arxiv, ecc.). **Non** necessario se `data/raw` e `train.jsonl` sono già copiati.

### Tier E — Intero repo `eurobot_baby`

`du` sul pod ha mostrato **~22 GB** per `/workspace/eurobot_baby` (include **tutti** i checkpoint intermedi `step_*`, log locali, ecc.).

Per risparmiare spazio, su disco locale puoi tenere **solo**:
- `step_214496` (o l’ultimo disponibile)
- eventualmente **un** checkpoint intermedio (es. `step_80000`) se vuoi backup

Gli altri `step_*` si possono **eliminare sul pod prima dell’export** se serve ridurre.

---

## Comandi di export (da PC Windows / WSL)

Sostituisci chiave e host se diversi.

```powershell
# Variabili
$KEY = "$env:USERPROFILE\.ssh\eubot_ed25519"
$HOST = "root@194.68.245.207"
$PORT = 22125
$DEST = "C:\Users\info\progetti\eubot\backup_runpod_eurobot_baby"

# Tier A minimo
scp -P $PORT -i $KEY -r "${HOST}:/workspace/eurobot_baby/models/checkpoints/step_214496" "$DEST\models\checkpoints\"
scp -P $PORT -i $KEY -r "${HOST}:/workspace/eurobot_baby/models/tokenizer" "$DEST\models\"
scp -P $PORT -i $KEY "${HOST}:/workspace/eurobot_baby/configs/training.yaml" "$DEST\configs\"

# Tier B (lungo: ~4.3 GB)
scp -P $PORT -i $KEY "${HOST}:/workspace/eurobot_baby/data/processed/train.jsonl" "$DEST\data\processed\"
```

Per cartelle grandi è meglio **rsync** (WSL o Git Bash):

```bash
rsync -avz -e "ssh -p 22125 -i ~/.ssh/eubot_ed25519" \
  root@194.68.245.207:/workspace/eurobot_baby/data/processed/train.jsonl \
  ./backup_runpod_eurobot_baby/data/processed/
```

---

## Ripristino su nuovo server

1. Clonare / copiare il repo `eurobot_baby` (o sync cartella).
2. Copiare `models/checkpoints/step_214496`, `models/tokenizer`, `configs/training.yaml`.
3. Copiare `data/processed/train.jsonl` e `val.jsonl` (o rigenerare da `data/raw` + `build_dataset.py`).
4. Creare venv, installare dipendenze come sul pod (`torch`, `transformers`, …).
5. **Per continuare training** (nuovo `max_steps` o nuovo dataset):  
   `python scripts/train.py --resume models/checkpoints/step_214496`
6. **Per inference/chat**: caricare da `step_214496` come in `scripts/chat.py` / `serve.py` (se presente).

---

## “Quanto mancava?” (chiusura run)

I run con **`max_steps: 144496`** e poi **`max_steps: 214496`** sullo stesso corpus sono andati a **`Training finished.`** Per ulteriori epoche alzare `max_steps` oltre l’ultimo `step_*` e ripartire con `--resume`.

---

## Anti-crash / riavvio

Il codice **non** rilancia automaticamente i processi. In caso di crash intermedio si riparte dall’**ultima cartella** `models/checkpoints/step_<N>` con `--resume`.  
Checkpoint intermedi ogni `save_every` (es. 4000–8000 a seconda del run); ultimo obiettivo: `step_214496`.
