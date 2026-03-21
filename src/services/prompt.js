import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SYSTEM_PROMPT_PATH = path.join(__dirname, "..", "prompts", "system.txt");

let cached = "";

export function loadSystemPrompt() {
  if (cached) return cached;
  cached = fs.readFileSync(SYSTEM_PROMPT_PATH, "utf8").trim();
  return cached;
}

/**
 * @param {object} opts
 * @param {string} opts.systemPrompt
 * @param {string[]} opts.preferences
 * @param {Array<{ role: string, content: string }>} opts.history
 * @param {string} opts.userMessage
 */
export function buildPrompt({ systemPrompt, preferences, history, userMessage }) {
  const parts = [];

  parts.push(systemPrompt.trim());
  parts.push("");

  if (preferences.length > 0) {
    parts.push("Known user interests/preferences (use naturally, do not list back unless relevant):");
    parts.push(preferences.join(", "));
    parts.push("");
  }

  if (history.length > 0) {
    parts.push("Recent conversation:");
    for (const m of history) {
      const label = m.role === "user" ? "User" : "Assistant";
      parts.push(`${label}: ${m.content}`);
    }
    parts.push("");
  }

  parts.push("User:");
  parts.push(userMessage);
  parts.push("");
  parts.push("Assistant:");

  return parts.join("\n");
}

/**
 * OpenAI-compatible chat messages (system + history + current user message).
 * @param {object} opts
 * @param {string} opts.systemPrompt
 * @param {string[]} opts.preferences
 * @param {Array<{ role: string, content: string }>} opts.history
 * @param {string} opts.userMessage
 * @returns {Array<{ role: "system" | "user" | "assistant", content: string }>}
 */
export function buildMessages({ systemPrompt, preferences, history, userMessage }) {
  let systemContent = systemPrompt.trim();
  if (preferences.length > 0) {
    systemContent +=
      "\n\nKnown user interests/preferences (use naturally, do not list back unless relevant):\n" +
      preferences.join(", ");
  }

  /** @type {Array<{ role: "system" | "user" | "assistant", content: string }>} */
  const messages = [{ role: "system", content: systemContent }];

  for (const m of history) {
    if (m.role === "user" || m.role === "assistant") {
      messages.push({ role: m.role, content: m.content });
    }
  }

  messages.push({ role: "user", content: userMessage });

  return messages;
}
