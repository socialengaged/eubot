import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { config } from "./config.js";
import chatRouter from "./routes/chat.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(__dirname, "..", "public");

const app = express();

app.use(express.json({ limit: "256kb" }));

app.use(chatRouter);

app.use(express.static(publicDir));

app.get("*", (_req, res) => {
  res.sendFile(path.join(publicDir, "index.html"));
});

app.listen(config.port, () => {
  console.log(`Eubot listening on http://localhost:${config.port}`);
  if (config.aiProvider === "openai") {
    console.log(`AI: OpenAI-compatible (${config.aiBaseUrl}, model: ${config.aiModel})`);
  } else {
    console.log(`AI: Ollama ${config.ollamaBaseUrl} (model: ${config.ollamaModel})`);
  }
});
