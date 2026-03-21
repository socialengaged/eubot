# Eubot

Chatbot locale (Node.js + Express) con personalità definita, memoria per utente e supporto **Ollama** o API **OpenAI-compat** (Groq, OpenRouter, Mistral, ecc.).

## Quick start

```bash
git clone https://github.com/socialengaged/eubot.git
cd eubot
cp .env.example .env   # Windows: copy .env.example .env
npm install
npm start
```

Apri **http://localhost:3000**.

### Backend AI

- **Ollama (locale):** installa [Ollama](https://ollama.com), `ollama pull mistral`, lascia `AI_PROVIDER=ollama` in `.env`.
- **Groq / altri:** imposta `AI_PROVIDER=openai`, `AI_BASE_URL`, `AI_API_KEY`, `AI_MODEL` (vedi `.env.example`).

## Documentazione

Architettura, variabili d’ambiente e API: **[PROJECT_SPEC.md](PROJECT_SPEC.md)**.

## Licenza

MIT
