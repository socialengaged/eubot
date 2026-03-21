# Eubot Coder su RunPod

## Prerequisiti

- Pod con GPU (consigliato 16 GB+ VRAM)
- Repo clonato: `git clone https://github.com/socialengaged/eubot.git`

## Comandi

```bash
cd eubot/eubot-coder   # oppure path dove hai clonato
pip install -r requirements.txt
export HF_TOKEN=hf_xxx   # opzionale, evita rate limit HF

python scripts/prepare_data.py
# IT+EN (default): OPUS en-it + CodeFeedback IT se presente
# Solo EN: python scripts/prepare_data.py --no-include_italian
python scripts/finetune.py
python scripts/merge_adapter.py
python scripts/chat.py
```

## SSH (TCP esposto)

Esempio dalla dashboard RunPod:

```bash
ssh root@213.173.103.178 -p 46829 -i ~/.ssh/id_ed25519
```

Usa la **stessa chiave** registrata sul Pod (es. `eubot_ed25519` se quella e quella caricata su RunPod).

Questa modalità supporta anche SCP/SFTP per copiare file sul Pod.

**Nota (Cursor / agent):** la connessione SSH dal terminale integrato puo richiedere chiave corretta, rete, o andare in timeout. In caso di `Permission denied`, verifica path della chiave e che la porta TCP sia ancora esposta sul Pod. Per training lunghi, preferisci **tmux** o **screen** sulla sessione SSH.

## Esporre l'API

```bash
python scripts/serve.py --host 0.0.0.0 --port 8080
```

Apri la porta HTTP nel pannello RunPod e usa `http://<ip>:8080/v1/chat/completions`.
