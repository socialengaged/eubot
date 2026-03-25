# Solo Eurobot Baby (niente Sage) — cosa serve in locale, spegnimento server, chat di test

## 1) È già tutto in locale?

**Controlla la dimensione del file** `backup_runpod_bundles/eurobot_baby_FULL_repo_*.tar`:

| Stato | Cosa fare |
|--------|-----------|
| Dimensione **≈ uguale** a `BYTES=` nel file `.eurobot_full_export_ok` (ordine **~22 000 000 000** byte) | OK per considerare il download **completo** (verifica anche SHA256, vedi sotto). |
| Dimensione **molto più piccola** (es. centinaia di MB) | Download **incompleto** / interrotto → **non** spegnere il server finché non finisci `rsync`. |

**Verifica rapida (PowerShell):**

```powershell
cd C:\Users\info\progetti\eubot\backup_runpod_bundles
Get-Content .eurobot_full_export_ok
(Get-Item .\eurobot_baby_FULL_repo_*.tar).Length
```

I due numeri (`BYTES=` e `Length`) devono **coincidere**.

**Checksum (WSL o Git Bash):**

```bash
cd /mnt/c/Users/info/progetti/eubot/backup_runpod_bundles
sha256sum -c SHA256SUMS_ONLY_FULL_*.txt
```

Oppure esegui `tools/verify_baby_bundle.ps1` dalla root del repo.

---

## 2) Si può spegnere il server RunPod?

**Sì**, quando:

1. Il `.tar` FULL ha **la dimensione giusta** e `sha256sum -c` **passa**.
2. (Opzionale ma consigliato) Hai anche `eurobot_baby_TIER_A_minimal_*.tar.gz` se vuoi un pacchetto piccolo di **backup** oltre al FULL.

**Senza** download completo del FULL, spegnere il pod = **rischio di perdere** l’unica copia “completa” del repo + checkpoint.

---

## 3) Cosa scaricare se vuoi **solo Baby** (ignora Sage)

Sul pod, in `eubot_offline_bundles/`, ti bastano ad esempio:

- `eurobot_baby_FULL_repo_*.tar` (o `.tar.gz` se generato così)
- `SHA256SUMS_ONLY_FULL_*.txt`
- `.eurobot_full_export_ok`
- Opzionale: `eurobot_baby_TIER_A_minimal_*.tar.gz`, `eurobot_baby_data_processed_*.tar.gz`, ecc.

**Non** servono i file `*sage*`.

Comando (WSL) già documentato in `tools/download_runpod_bundles.sh` — scarica tutta la cartella; puoi **interrompere** e **escludere** file Sage a mano se vuoi risparmiare spazio, oppure scaricare tutto e cancellare in locale i `*sage*`.

---

## 4) Chat / test in locale

Questo repo (`eurobot_baby/` su Git) è un **mirror** con dataset tooling: **non** contiene di solito `train.py` / `chat.py` — quelli stanno sul **pod** e sono dentro il **`FULL` tar**.

Dopo aver **estratto** il tar:

```bash
# Esempio WSL
cd /mnt/c/Users/info/progetti/eubot
mkdir -p restore && cd restore
tar -xf ../backup_runpod_bundles/eurobot_baby_FULL_repo_*.tar
# di solito: ./eurobot_baby/...
```

Poi cerca gli script di inference:

```bash
cd eurobot_baby
tar -tf ../backup_runpod_bundles/eurobot_baby_FULL_repo_*.tar | grep -iE 'chat|infer|serve|generate' | head -20
# oppure
ls scripts/
```

**Requisiti tipici:**

- Python 3.10+, **GPU CUDA** adeguata (modello e checkpoint definiti in `configs/training.yaml` **dentro** il tar).
- `pip install` di `torch`, `transformers`, ecc. come sul pod (vedi eventuale `requirements.txt` nel tar).

**Test minimo:**  
Se esiste `scripts/chat.py` (o simile):

```bash
python scripts/chat.py --help
# oppure con checkpoint step_144496, come indicato nel README sul pod
```

Se **non** c’è uno script dedicato, il test è avviare un **forward** da notebook o uno script minimo che carica `AutoModelForCausalLM.from_pretrained(...)` dal checkpoint — dipende da come è stato salvato il training sul pod.

---

## 5) Riferimenti

- Export / download: `docs/SHUTDOWN_CHECKLIST_AND_FRESH_INSTALL.md` (root `eubot`)
- Migrazione checkpoint: `docs/EXPORT_MIGRATE_FROM_RUNPOD.md`
