import { config } from "../config.js";

/** @typedef {{ role: 'user' | 'assistant', content: string }} ChatMessage */

/** @type {Map<string, { history: ChatMessage[], preferences: string[] }>} */
const store = new Map();

const maxMessages = Math.max(2, (config.maxExchanges || 10) * 2);

function ensureUser(userId) {
  if (!store.has(userId)) {
    store.set(userId, { history: [], preferences: [] });
  }
  return store.get(userId);
}

/**
 * @param {string} userId
 * @returns {ChatMessage[]}
 */
export function getHistory(userId) {
  return [...ensureUser(userId).history];
}

/**
 * @param {string} userId
 * @returns {string[]}
 */
export function getPreferences(userId) {
  return [...ensureUser(userId).preferences];
}

/**
 * Append one exchange and trim to max length.
 * @param {string} userId
 * @param {string} userMessage
 * @param {string} assistantReply
 */
export function addExchange(userId, userMessage, assistantReply) {
  const u = ensureUser(userId);
  u.history.push({ role: "user", content: userMessage });
  u.history.push({ role: "assistant", content: assistantReply });
  while (u.history.length > maxMessages) {
    u.history.shift();
  }
}

/**
 * Merge new preference tokens; cap list size, dedupe case-insensitively.
 * @param {string} userId
 * @param {string[]} newPrefs
 */
export function mergePreferences(userId, newPrefs) {
  const u = ensureUser(userId);
  const seen = new Set(u.preferences.map((p) => p.toLowerCase()));
  for (const p of newPrefs) {
    const t = p.trim();
    if (!t) continue;
    const key = t.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      u.preferences.push(t);
    }
  }
  u.preferences = u.preferences.slice(-30);
}

/** For tests / future admin: clear all in-memory data */
export function _clearAllForTests() {
  store.clear();
}
