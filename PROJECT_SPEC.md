# Eubot — Project specification (living document)

**Last updated:** 2026-03-21  
**Status:** Implemented — multi-provider AI (Ollama + OpenAI-compatible APIs).

---

## 1. Purpose

**Eubot** is a local-first AI chatbot: Node.js + Express backend, vanilla web UI. **Default:** Mistral (or any model) via **Ollama** locally. **Optional:** OpenAI-compatible APIs (**Groq**, **OpenRouter**, Mistral La Plateforme, Together, etc.) via `AI_PROVIDER=openai` — no code changes to switch. Designed to be extended toward DB-backed memory, auth, multi-tenant SaaS, and embeddable widgets.

---

## 2. Quick start

1. **Prerequisites:** Node.js 18+.
2. **AI backend (pick one):**
   - **Ollama (local):** [Ollama](https://ollama.com) running; `ollama pull mistral` (or set `OLLAMA_MODEL`).
   - **Groq / OpenRouter / etc.:** Set `AI_PROVIDER=openai`, `AI_BASE_URL`, `AI_API_KEY`, `AI_MODEL` in `.env` (see [§3](#3-environment-variables)).
3. **Install & run:**
   ```bash
   npm install
   npm start
   ```
4. Open **http://localhost:3000** (UI) or `POST http://localhost:3000/chat` (API).

---

## 3. Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | HTTP port for Express |
| `AI_PROVIDER` | `ollama` | `ollama` (local) or `openai` (OpenAI-compatible HTTP API) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server (no trailing slash); used when `AI_PROVIDER=ollama` |
| `OLLAMA_MODEL` | `mistral` | Model name (`ollama list`) |
| `AI_BASE_URL` | `""` | OpenAI-compatible base URL, e.g. `https://api.groq.com/openai` or `https://openrouter.ai/api` |
| `AI_API_KEY` | `""` | Bearer token for `openai` provider |
| `AI_MODEL` | `mistral` | Model id for `openai` provider (e.g. Groq `mixtral-8x7b-32768`, OpenRouter `mistralai/mistral-7b-instruct`) |
| `MAX_EXCHANGES` | `10` | Max user/assistant **pairs** kept per `userId` |

**Cost notes (rough):** Groq offers a generous free tier and very fast inference; OpenRouter aggregates many models pay-as-you-go. Remote GPU VPS (Vast.ai, RunPod) is ~$70–150+/month if you self-host Ollama on a GPU — use only if you need full data control or high volume.

Copy `.env` and adjust as needed. `.env` is gitignored from sharing secrets in real deployments; for local dev it’s fine.

---

## 4. API contract

### `POST /chat`

**Request JSON:**

```json
{
  "message": "string",
  "userId": "string"
}
```

**Success (200):**

```json
{
  "reply": "string"
}
```

**Errors:** `400` validation; `502` backend unreachable or model error (body: `{ "error": "..." }`).

---

## 5. Architecture

| Layer | Responsibility |
|--------|----------------|
| `src/server.js` | Express: JSON body, `/chat` route, static `public/`, SPA fallback `GET *` → `index.html` |
| `src/routes/chat.js` | Validate input, memory + prefs, build prompt or messages, call AI backend, persist exchange |
| `src/services/ai.js` | Ollama `POST /api/generate` **or** OpenAI-compatible `POST /v1/chat/completions` |
| `src/services/memory.js` | In-memory `Map<userId, { history, preferences }>` |
| `src/services/preferences.js` | Topic keywords + repeated-term hints |
| `src/services/prompt.js` | `buildPrompt()` (flat string for Ollama) and `buildMessages()` (chat array for OpenAI-compat) |
| `src/prompts/system.txt` | Assistant identity and rules (editable without code changes) |
| `public/` | Mobile-first chat UI |

**Conversation window:** Last `MAX_EXCHANGES` **exchanges** (each = user + assistant), stored as alternating messages.

**Preferences:** On each request, interests are detected from the **current** message, merged into the user’s list (deduped, capped), and **included in the same** prompt so personalization applies immediately.

---

## 6. AI backends

### Ollama (`AI_PROVIDER=ollama`)

- Endpoint: `{OLLAMA_BASE_URL}/api/generate`
- Body: `{ "model", "prompt", "stream": false }`
- Reply: JSON field `response`

### OpenAI-compatible (`AI_PROVIDER=openai`)

- Endpoint: `{AI_BASE_URL}/v1/chat/completions`
- Headers: `Authorization: Bearer {AI_API_KEY}`
- Body: `{ "model", "messages", "stream": false }` — messages from `buildMessages()` (system + history + user)
- Reply: `choices[0].message.content`

Works with **Groq**, **OpenRouter**, Mistral API, Together, and any OpenAI-compatible server.

If the backend is unreachable or returns an error, `/chat` returns **502** with a clear message.

---

## 7. Frontend behavior

- `userId`: generated once as `u_<uuid>`, stored in `localStorage` under `eubot_user_id`.
- `POST /chat` with `{ message, userId }`; shows user bubbles (right), assistant (left), loading dots, sticky bottom input.
- Uses relative `/chat` (same origin as the app).

---

## 8. File map

```
eubot/
  package.json
  .env                 # local config (not committed with secrets in prod)
  .gitignore
  PROJECT_SPEC.md      # this file
  src/
    server.js
    config.js
    routes/chat.js
    services/
      ai.js
      memory.js
      preferences.js
      prompt.js
    prompts/system.txt
  public/
    index.html
    css/style.css
    js/app.js
```

---

## 9. Future evolution (hooks)

| Goal | Suggested approach |
|------|-------------------|
| **Database** | Replace `memory.js` with a repository interface; keep route + prompt shape |
| **Multiple personalities** | Load different `prompts/*.txt` or DB rows; pass `personalityId` in API later |
| **Authentication** | Map sessions/users to `userId` or replace `userId` with authenticated subject |
| **Deploy** | Reverse proxy, HTTPS, env-based `OLLAMA_*`, health check on `/` or `/health` |
| **Embed on sites** | iframe or small JS bundle pointing to your deployed `/chat` + CORS allowlist |

---

## 10. Known limitations (MVP)

- Memory is **process RAM** only — restart clears all users.
- Preference detection is **heuristic** (keywords + repetition), not an ML classifier.
- No rate limiting, auth, or persistence — add before public internet exposure.

---

## 11. Resume checklist

When continuing work:

1. Read this file + `.env`.
2. Run `npm start` and confirm the chosen backend (Ollama running + model pulled, or `openai` vars set).
3. Extend **one module at a time** (e.g. DB → only `memory.js` + config).

---

## 12. Changelog (high level)

- **2026-03-21:** Initial implementation per plan: Express `/chat`, Ollama generate, in-memory memory + prefs, prompt builder, static chat UI, `PROJECT_SPEC.md`.
- **2026-03-21:** Personality update — Eubot now reflects Eugenio's operator mindset: direct, practical, strategic, no fluff. System prompt rewritten, welcome message updated.
- **2026-03-21:** Multi-provider AI — `src/services/ai.js` supports `ollama` or `openai` (Groq, OpenRouter, etc.); `buildMessages()` for chat completions; env `AI_PROVIDER`, `AI_BASE_URL`, `AI_API_KEY`, `AI_MODEL`.
