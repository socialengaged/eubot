import "dotenv/config";

const port = Number.parseInt(process.env.PORT ?? "3000", 10);
const ollamaBaseUrl = (process.env.OLLAMA_BASE_URL ?? "http://localhost:11434").replace(/\/$/, "");
const ollamaModel = process.env.OLLAMA_MODEL ?? "mistral";

/** @type {"ollama" | "openai"} */
const rawProvider = (process.env.AI_PROVIDER ?? "ollama").toLowerCase().trim();
const aiProvider = rawProvider === "openai" ? "openai" : "ollama";

const aiBaseUrl = (process.env.AI_BASE_URL ?? "").replace(/\/$/, "");
const aiApiKey = process.env.AI_API_KEY ?? "";
const aiModel = process.env.AI_MODEL ?? "mistral";

const host = (process.env.HOST ?? "0.0.0.0").trim() || "0.0.0.0";

export const config = {
  /** Bind address (0.0.0.0 = tutte le interfacce, utile su RunPod/VPS) */
  host,
  port: Number.isFinite(port) ? port : 3000,
  ollamaBaseUrl,
  ollamaModel,
  aiProvider,
  /** OpenAI-compatible API base (e.g. https://api.groq.com/openai) */
  aiBaseUrl,
  aiApiKey,
  /** Model id for OpenAI-compatible providers */
  aiModel,
  /** Max user/assistant pairs kept per user */
  maxExchanges: Number.parseInt(process.env.MAX_EXCHANGES ?? "10", 10) || 10,
};
