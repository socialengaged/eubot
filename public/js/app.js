const STORAGE_KEY = "eubot_user_id";

function getOrCreateUserId() {
  let id = localStorage.getItem(STORAGE_KEY);
  if (!id) {
    id = `u_${crypto.randomUUID()}`;
    localStorage.setItem(STORAGE_KEY, id);
  }
  return id;
}

const chatEl = document.getElementById("chat");
const formEl = document.getElementById("form");
const inputEl = document.getElementById("input");
const sendEl = document.getElementById("send");
const loadingEl = document.getElementById("loading");

const userId = getOrCreateUserId();

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatEl.scrollTop = chatEl.scrollHeight;
  });
}

function addBubble(text, role) {
  const div = document.createElement("div");
  div.className = `msg msg--${role === "user" ? "user" : "ai"}`;
  div.textContent = text;
  chatEl.appendChild(div);
  scrollToBottom();
}

function addError(text) {
  const div = document.createElement("div");
  div.className = "msg msg--error";
  div.textContent = text;
  chatEl.appendChild(div);
  scrollToBottom();
}

function setLoading(on) {
  loadingEl.classList.toggle("hidden", !on);
  sendEl.disabled = on;
  inputEl.disabled = on;
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;

  addBubble(message, "user");
  inputEl.value = "";
  setLoading(true);

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, userId }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      const err = typeof data.error === "string" ? data.error : res.statusText;
      addError(err || "Something went wrong.");
      return;
    }

    if (typeof data.reply === "string") {
      addBubble(data.reply, "ai");
    } else {
      addError("Invalid response from server.");
    }
  } catch {
    addError("Network error. Is the server running?");
  } finally {
    setLoading(false);
    inputEl.focus();
  }
});

addBubble(
  "Eubot here. What are you working on?",
  "ai"
);
inputEl.focus();
