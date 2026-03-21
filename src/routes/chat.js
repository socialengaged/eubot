import { Router } from "express";
import { config } from "../config.js";
import { generateCompletion } from "../services/ai.js";
import {
  getHistory,
  addExchange,
  getPreferences,
  mergePreferences,
} from "../services/memory.js";
import { buildPrompt, buildMessages, loadSystemPrompt } from "../services/prompt.js";
import { detectPreferences } from "../services/preferences.js";

const router = Router();

let systemPromptCache = "";

function getSystemPrompt() {
  if (!systemPromptCache) {
    systemPromptCache = loadSystemPrompt();
  }
  return systemPromptCache;
}

router.post("/chat", async (req, res) => {
  try {
    const { message, userId } = req.body ?? {};

    if (typeof message !== "string" || !message.trim()) {
      return res.status(400).json({ error: "message is required (non-empty string)" });
    }
    if (typeof userId !== "string" || !userId.trim()) {
      return res.status(400).json({ error: "userId is required (non-empty string)" });
    }

    const trimmedMessage = message.trim();
    const uid = userId.trim();

    const history = getHistory(uid);
    const detected = detectPreferences(trimmedMessage);
    mergePreferences(uid, detected);
    const preferences = getPreferences(uid);
    const reply =
      config.aiProvider === "openai"
        ? await generateCompletion({
            messages: buildMessages({
              systemPrompt: getSystemPrompt(),
              preferences,
              history,
              userMessage: trimmedMessage,
            }),
          })
        : await generateCompletion({
            prompt: buildPrompt({
              systemPrompt: getSystemPrompt(),
              preferences,
              history,
              userMessage: trimmedMessage,
            }),
          });

    addExchange(uid, trimmedMessage, reply);

    return res.json({ reply });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    console.error("[/chat]", err);
    return res.status(502).json({ error: msg });
  }
});

export default router;
