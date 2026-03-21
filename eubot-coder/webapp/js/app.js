(() => {
  const API_URL = window.API_URL || `${location.origin}/v1/chat/completions`;

  const chat = document.getElementById("chat");
  const input = document.getElementById("input");
  const btnSend = document.getElementById("btn-send");
  const btnClear = document.getElementById("btn-clear");
  const statusEl = document.getElementById("status");

  let messages = [];
  let busy = false;

  function setStatus(online) {
    statusEl.textContent = online ? "Online" : "Offline";
    statusEl.className = "status " + (online ? "online" : "offline");
  }

  function addBubble(role, text) {
    const div = document.createElement("div");
    div.className = `message ${role === "user" ? "user" : "bot"}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    div.appendChild(bubble);
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return bubble;
  }

  function addTyping() {
    const div = document.createElement("div");
    div.className = "message bot typing";
    div.id = "typing";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    div.appendChild(bubble);
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return bubble;
  }

  function removeTyping() {
    const el = document.getElementById("typing");
    if (el) el.remove();
  }

  function formatCode(text) {
    return text
      .replace(/```(\w*)\n([\s\S]*?)```/g, "<pre><code>$2</code></pre>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text || busy) return;

    busy = true;
    btnSend.disabled = true;
    input.value = "";
    autoResize();

    addBubble("user", text);
    messages.push({ role: "user", content: text });

    const typingBubble = addTyping();

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "eubot-coder",
          messages: messages,
          max_tokens: 1024,
          temperature: 0.7,
        }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const reply = data.choices?.[0]?.message?.content || "(nessuna risposta)";

      removeTyping();
      const bubble = addBubble("bot", "");
      bubble.innerHTML = formatCode(reply);
      messages.push({ role: "assistant", content: reply });
      setStatus(true);
    } catch (err) {
      removeTyping();
      addBubble("bot", `Errore: ${err.message}`);
      setStatus(false);
    }

    busy = false;
    btnSend.disabled = false;
    input.focus();
  }

  function clearChat() {
    messages = [];
    chat.innerHTML = "";
    addBubble("bot", "Eubot qui. Cosa stai costruendo?");
    input.focus();
  }

  function autoResize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 150) + "px";
  }

  // Health check
  async function checkHealth() {
    try {
      const base = API_URL.replace("/v1/chat/completions", "");
      const r = await fetch(`${base}/health`, { method: "GET" });
      setStatus(r.ok);
    } catch {
      setStatus(false);
    }
  }

  btnSend.addEventListener("click", sendMessage);
  btnClear.addEventListener("click", clearChat);

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.addEventListener("input", autoResize);

  checkHealth();
  setInterval(checkHealth, 15000);
  input.focus();
})();
