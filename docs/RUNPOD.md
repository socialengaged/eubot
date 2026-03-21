# Deploy su RunPod (SSH)

## Connessione SSH

Nel tuo PC (PowerShell o terminale), con chiave già configurata:

```bash
ssh eubot
```

Se non usi l’alias, l’equivalente è:

```bash
ssh -i ~/.ssh/eubot_ed25519 rwj7ag6zz2entl-644120d4@ssh.runpod.io
```

> **Nota:** se avevi `~/.ssh/id_ed25519`, copia o rinominando la chiave corretta oppure imposta `IdentityFile` in `~/.ssh/config` (nel repo è documentato `eubot_ed25519`).

Alcuni client (incluso l’ambiente IDE) **non supportano bene** `ssh host comando` o `scp` verso RunPod: usa un **terminale locale interattivo** per SSH, `git clone`/`rsync` sul pod, o l’upload file del dashboard RunPod.

## Sul Pod (dopo aver copiato il progetto)

```bash
cd ~/eubot
# oppure: cd /workspace/eubot
chmod +x scripts/setup-runpod.sh
./scripts/setup-runpod.sh
```

Oppure a mano:

```bash
cp -n .env.example .env
nano .env   # AI_PROVIDER=openai + Groq consigliato sul cloud (niente Ollama se non installato)
npm install
HOST=0.0.0.0 PORT=3000 npm start
```

## Porta HTTP

1. Nel pannello RunPod, apri **TCP** sulla porta **3000** (o quella in `PORT`) verso il Pod.
2. In alternativa testa in locale sul Pod: `curl` verso `127.0.0.1:3000`.

## Test chat da terminale (sul Pod o da PC con tunnel)

Con il server avviato:

```bash
curl -s -X POST http://127.0.0.1:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"ciao","userId":"test-runpod"}'
```

Risposta attesa: JSON `{"reply":"..."}`.

## Backend AI sul Pod

- **Consigliato:** `AI_PROVIDER=openai` + **Groq** (veloce, tier free) in `.env` — non serve GPU sul Pod per inferenza.
- **Ollama sul Pod:** solo se installi Ollama e scarichi il modello (richiede RAM/VRAM adeguate).
