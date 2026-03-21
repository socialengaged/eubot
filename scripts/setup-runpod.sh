#!/usr/bin/env bash
# Setup Eubot su Linux (RunPod / VPS). Esegui dalla root del repo: ./scripts/setup-runpod.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f package.json ]]; then
  echo "Errore: esegui questo script dalla cartella eubot (package.json mancante)."
  exit 1
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Creato .env da .env.example — modificalo (API keys, AI_PROVIDER)."
fi

if ! command -v node &>/dev/null; then
  echo "Node.js non trovato. Su Debian/Ubuntu: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt-get install -y nodejs"
  exit 1
fi

npm install
echo ""
echo "OK. Avvio suggerito:"
echo "  HOST=0.0.0.0 PORT=3000 npm start"
echo "Poi test:"
echo '  curl -s -X POST http://127.0.0.1:3000/chat -H "Content-Type: application/json" -d '"'"'{"message":"ciao","userId":"test"}'"'"''
