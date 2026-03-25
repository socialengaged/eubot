# Setup macchina scraping (Blocco B/C del piano)

## Opzioni

- **Hetzner CX32** (4 vCPU, 8 GB RAM, 80 GB): ~7 EUR/mese
- **RunPod CPU** on-demand (~$0.10/h)
- **OVH VPS** se già disponibile (verificare Python 3.11+ e disco)
- **Stesso pod GPU del training** — possibile: vedi [Scraping in parallelo al training](#scraping-in-parallelo-al-training-sulla-stessa-macchina)

## Passi

1. Creare la VM, aprire SSH (porta 22).
2. Copiare la cartella `eurobot_baby/tools/scraping/` sulla macchina (scp/rsync).
3. Eseguire:

```bash
chmod +x setup_scraper.sh
sudo ./setup_scraper.sh
source /root/scraper_venv/bin/activate
```

4. Variabili ambiente utili:

| Variabile | Uso |
|-----------|-----|
| `EUROBOT_SCRAPING_RUN` | Base output (default: `/workspace/eurobot_scraping_run`) — sotto `output/` stanno tutti i sorgenti per il merge |
| `EUROBOT_MERGE_OUT` | Destinazione merge (`merge_outputs_to_raw_names.py`): default in `run_stem_pipeline.sh` è `data/raw`; con `run_scraping_safe.sh` è **`data/raw_staging`** |
| `EUROBOT_GUTENBERG_OUT` | Output Gutenberg esoterico / scienza |
| `EUROBOT_SACRED_OUT` | Output sacred-texts deep |
| `EUROBOT_RAW_DIR` | Output gnosis retry |
| `HF_HOME` | Cache HuggingFace (opzionale, disco grande) |
| `EUROBOT_GUTENBERG_THEOLOGY_OUT` | Output Gutenberg teologia (`download_theology_gutenberg.py`) |
| `EUROBOT_ARXIV_MODE=cs` | Subset informatica da `download_arxiv_stem.py` → `output/arxiv/arxiv_cs.jsonl` |
| `EUROBOT_ARXIV_CS_MAX` | (solo pipeline STEM) più righe per `arxiv_cs.jsonl` — default `12000` in `run_stem_pipeline.sh` |
| `EUROBOT_GUTENBERG_COMPUTING_OUT` | Testi storici logica/calcolo (`download_gutenberg_computing.py`) → merge in `71_gutenberg_computing.txt` |
| `PY` | Interprete Python (default `/root/eurobot_baby_venv/bin/python`) |

**Teologia + informatica:** dopo i download, dalla root scraping (o con le pipeline sotto) eseguire merge verso `raw` o `raw_staging`:

```bash
python merge_outputs_to_raw_names.py --base "$EUROBOT_SCRAPING_RUN" --out /path/to/eurobot_baby/data/raw_staging
```

5. Trasferire gli output verso il pod GPU (se scraping è su altra macchina):

```bash
scp -P 17100 -i ~/.ssh/eubot_ed25519 output/*.txt root@213.173.103.83:/workspace/eurobot_baby/data/raw/
```

---

## Scraping in parallelo al training (sulla stessa macchina)

### Cosa è sicuro

- **Training sulla GPU** + job **CPU/rete** (Gutenberg, sacred-texts, gnosis, merge su disco) in genere **non** competono per la VRAM.
- **Una** pipeline HF streaming pesante alla volta (`run_stem_pipeline.sh` è già sequenziale: fisica HF → arXiv ×4 → math → astro → …). Non lanciare **due** `run_stem_pipeline.sh` contemporaneamente.
- In un secondo terminale si può lanciare **`run_esoteric_pipeline.sh`** o **`run_gutenberg_esoteric.sh`** solo se rete e disco reggono (monitorare `iowait`, `df -h`).

### Cosa evitare

- **`python scripts/build_dataset.py`** che sovrascrive `data/processed/train.jsonl` **mentre** `train.py` è in esecuzione sullo stesso file: rischio output corrotto o run instabile. Fermare il training (o attendere checkpoint) prima del rebuild.
- Due merge simultanei verso la **stessa** directory `--out`.

### `data/raw` vs `data/raw_staging`

| Directory | Uso |
|-----------|-----|
| `data/raw` | Corpus “attivo” usato dall’ultimo `build_dataset.py` generato |
| `data/raw_staging` | Destinazione consigliata per il **merge** mentre il training è ancora in corso su un `train.jsonl` già costruito |

Flusso:

1. Durante il training: `run_scraping_safe.sh` (o `EUROBOT_MERGE_OUT=.../data/raw_staging bash run_stem_pipeline.sh`) popola **`raw_staging`**.
2. Finestra di manutenzione: training fermo → `bash promote_raw_staging.sh` (o `rsync` equivalente) da `raw_staging` → `raw`.
3. `cd /workspace/eurobot_baby && python scripts/build_dataset.py` → aggiornare `max_steps` → resume — vedi [RUNBOOK_PHASE0.md](../../docs/RUNBOOK_PHASE0.md).

### Script orchestrazione

| Script | Ruolo |
|--------|--------|
| `run_stem_pipeline.sh` | Pipeline STEM completa + merge (`EUROBOT_MERGE_OUT`, default `data/raw`) |
| `run_scraping_safe.sh` | Wrapper: imposta `HF_HOME`, merge default su **`data/raw_staging`** |
| `promote_raw_staging.sh` | `rsync` staging → `data/raw` (senza `--delete`; i file solo in `raw` restano) |
| `run_esoteric_pipeline.sh` | Sacred-texts deep → gnosis retry |
| `run_gutenberg_esoteric.sh` | Solo Gutenberg esoterico (fase 1) |

### Job HF / arXiv

Eseguire in **serie** gli step dentro `run_stem_pipeline.sh` (già così). Non avviare altri `load_dataset(streaming=True)` massicci in parallelo sulla stessa cache/disco.

### Monitoraggio rapido

```bash
nvidia-smi
df -h /workspace
free -h
```

Lasciare margine disco (~10–15% libero) prima di download grandi.
