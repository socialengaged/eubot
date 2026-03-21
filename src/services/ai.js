import { config } from "../config.js";

/**
 * @param {string} prompt
 * @returns {Promise<string>}
 */
async function generateOllama(prompt) {
  const url = `${config.ollamaBaseUrl}/api/generate`;
  const body = {
    model: config.ollamaModel,
    prompt,
    stream: false,
  };

  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(
      `Cannot reach Ollama at ${config.ollamaBaseUrl}. Is Ollama running? (${msg})`
    );
  }

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Ollama error ${res.status}: ${text.slice(0, 500)}`);
  }

  const data = await res.json();
  const reply = typeof data.response === "string" ? data.response.trim() : "";
  if (!reply) {
    throw new Error("Ollama returned an empty response.");
  }
  return reply;
}

/**
 * @param {Array<{ role: string, content: string }>} messages
 * @returns {Promise<string>}
 */
async function generateOpenAICompat(messages) {
  if (!config.aiBaseUrl) {
    throw new Error(
      "AI_BASE_URL is required when AI_PROVIDER=openai (e.g. https://api.groq.com/openai)"
    );
  }
  if (!config.aiApiKey.trim()) {
    throw new Error("AI_API_KEY is required when AI_PROVIDER=openai");
  }

  const url = `${config.aiBaseUrl}/v1/chat/completions`;
  const body = {
    model: config.aiModel,
    messages,
    stream: false,
  };

  const headers = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${config.aiApiKey}`,
  };

  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Cannot reach AI API at ${config.aiBaseUrl}. (${msg})`);
  }

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`AI API error ${res.status}: ${text.slice(0, 800)}`);
  }

  const data = await res.json();
  const content = data?.choices?.[0]?.message?.content;
  const reply = typeof content === "string" ? content.trim() : "";
  if (!reply) {
    throw new Error("AI API returned an empty response.");
  }
  return reply;
}

/**
 * Generate a reply using the configured provider.
 * @param {object} opts
 * @param {string} [opts.prompt] Flat prompt for Ollama /api/generate
 * @param {Array<{ role: string, content: string }>} [opts.messages] Chat messages for OpenAI-compatible APIs
 * @returns {Promise<string>}
 */
export async function generateCompletion({ prompt, messages }) {
  if (config.aiProvider === "ollama") {
    if (typeof prompt !== "string" || !prompt.trim()) {
      throw new Error("generateCompletion: prompt is required for Ollama provider.");
    }
    return generateOllama(prompt);
  }

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    throw new Error("generateCompletion: messages array is required for openai provider.");
  }
  return generateOpenAICompat(messages);
}
