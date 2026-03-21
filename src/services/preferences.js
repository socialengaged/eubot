/**
 * Topic/category keywords (English + common Italian variants for EU users).
 * Maps a canonical label to trigger words.
 */
const TOPIC_MAP = [
  { label: "technology", words: ["tech", "software", "code", "programming", "ai", "llm", "developer", "app", "tecnologia", "programmazione"] },
  { label: "science", words: ["science", "physics", "chemistry", "biology", "research", "scienza", "fisica"] },
  { label: "sports", words: ["sport", "football", "soccer", "basketball", "tennis", "calcio", "pallacanestro"] },
  { label: "music", words: ["music", "song", "band", "concert", "musica", "canzone"] },
  { label: "travel", words: ["travel", "trip", "flight", "vacation", "viaggio", "vacanza"] },
  { label: "food", words: ["food", "cooking", "recipe", "restaurant", "cucina", "ricetta", "cibo"] },
  { label: "books", words: ["book", "read", "novel", "libro", "lettura"] },
  { label: "health", words: ["health", "fitness", "doctor", "salute", "benessere"] },
  { label: "business", words: ["business", "startup", "marketing", "finance", "azienda", "lavoro"] },
  { label: "art", words: ["art", "design", "painting", "museum", "arte", "design"] },
];

const STOPWORDS = new Set([
  "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "do", "does", "did", "will", "would", "could",
  "should", "may", "might", "must", "shall", "can", "need", "dare",
  "i", "you", "he", "she", "it", "we", "they", "what", "which", "who",
  "this", "that", "these", "those", "am", "is", "are", "my", "your",
  "his", "her", "its", "our", "their", "me", "him", "us", "them",
  "and", "or", "but", "if", "then", "so", "because", "as", "of", "at",
  "to", "for", "in", "on", "with", "by", "from", "about", "into", "through",
  "just", "very", "really", "also", "not", "no", "yes", "how", "when",
  "where", "why", "there", "here", "some", "any", "all", "each", "every",
  "il", "lo", "la", "i", "gli", "le", "un", "una", "uno", "di", "da", "in",
  "su", "per", "con", "che", "non", "mi", "ti", "si", "ci", "vi", "è", "sono",
]);

/**
 * @param {string} message
 * @returns {string[]} unique preference labels / keywords
 */
export function detectPreferences(message) {
  const lower = message.toLowerCase();
  const found = new Set();

  for (const { label, words } of TOPIC_MAP) {
    for (const w of words) {
      if (lower.includes(w.toLowerCase())) {
        found.add(label);
        break;
      }
    }
  }

  // Simple "interesting" tokens: words 4+ chars, not stopwords
  const tokens = lower.match(/[a-zàèéìòù]{4,}/gi) ?? [];
  const freq = new Map();
  for (const t of tokens) {
    const w = t.toLowerCase();
    if (STOPWORDS.has(w)) continue;
    freq.set(w, (freq.get(w) ?? 0) + 1);
  }
  for (const [w, c] of freq) {
    if (c >= 2) found.add(w);
  }

  return [...found];
}
