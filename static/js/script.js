/* =========================================================
   CodePilot AI — script.js
   One file. Sections, in order:

     1. API      — every network call
     2. MD       — small escaping Markdown renderer
     3. UI       — DOM helpers, app state, toasts
     4. Chat     — send a question, render the answer
     5. Upload   — file picker and drag-and-drop
     6. Compare  — both models, side by side
     7. History  — sidebar list of earlier questions
     8. Settings — API keys saved from the browser
     9. Drawer   — the slide-in panel on phones
    10. Startup  — wiring, runs on DOMContentLoaded
   ========================================================= */


/* =========================================================
   1. API — every network call lives here and nowhere else
   ========================================================= */
const API = (() => {

  async function request(url, options = {}) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (err) {
      throw new Error("Could not reach the server. Is Flask still running?");
    }

    let data = {};
    try {
      data = await response.json();
    } catch (err) {
      throw new Error("The server sent back something unreadable.");
    }

    if (!response.ok) {
      throw new Error(data.error || `Request failed (${response.status}).`);
    }
    return data;
  }

  /*
    Keys are kept in this browser and nowhere else. They are sent with each
    request so the server can call Google or OpenAI on your behalf, and it
    stores nothing — close the tab and the server has no record of them.
  */
  const STORE_KEY = "codepilot.keys";

  function loadKeys() {
    try {
      return JSON.parse(localStorage.getItem(STORE_KEY)) || {};
    } catch (err) {
      return {};
    }
  }

  function saveKeys(keys) {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(keys));
      return true;
    } catch (err) {
      return false;   // private browsing, or storage disabled
    }
  }

  function clearKeys() {
    try { localStorage.removeItem(STORE_KEY); } catch (err) { /* nothing to do */ }
  }

  function headers(extra = {}) {
    const keys = loadKeys();
    const out = { ...extra };
    if (keys.gemini_key)   out["X-Gemini-Key"]   = keys.gemini_key;
    if (keys.openai_key)   out["X-OpenAI-Key"]   = keys.openai_key;
    if (keys.gemini_model) out["X-Gemini-Model"] = keys.gemini_model;
    if (keys.openai_model) out["X-OpenAI-Model"] = keys.openai_model;
    return out;
  }

  const json = (body) => ({
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });

  return {
    health:   ()      => request("/api/health"),
    chat:     (body)  => request("/api/chat", json(body)),
    compare:  (body)  => request("/api/compare", json(body)),
    loadKeys, saveKeys, clearKeys,
    upload:   (file)  => {
      const form = new FormData();
      form.append("file", file);
      return request("/api/upload", { method: "POST", body: form, headers: headers() });
    },
    clearIndex:   () => request("/api/clear-index", { method: "POST" }),
    testKey:      (body) => request("/api/test-key", json(body)),
    history:      () => request("/api/history"),
    clearHistory: () => request("/api/history", { method: "DELETE" }),
  };
})();


/* =========================================================
   2. MD — a small, deliberate Markdown renderer
   ========================================================= */
const MD = (() => {

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function inline(text) {
    return text
      .replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|\W)\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
               '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }

  function codeBlock(language, code) {
    const label = language || "code";
    return (
      `<div class="codeblock">` +
        `<div class="codeblock__bar"><span>${escapeHtml(label)}</span>` +
        `<button type="button" class="copy-btn">Copy</button></div>` +
        `<pre><code>${code}</code></pre>` +
      `</div>`
    );
  }

  /* Split on fences first so code is never touched by inline rules. */
  function render(raw) {
    const escaped = escapeHtml(raw || "");
    const parts = escaped.split(/```/);
    let html = "";

    parts.forEach((part, index) => {
      if (index % 2 === 1) {
        const newline = part.indexOf("\n");
        const language = newline === -1 ? "" : part.slice(0, newline).trim();
        const code = newline === -1 ? part : part.slice(newline + 1);
        html += codeBlock(language, code.replace(/\n$/, ""));
      } else {
        html += renderProse(part);
      }
    });
    return html;
  }

  function renderProse(text) {
    const lines = text.split("\n");
    let html = "";
    let listType = null;

    const closeList = () => {
      if (listType) { html += `</${listType}>`; listType = null; }
    };

    for (const line of lines) {
      const trimmed = line.trim();

      if (!trimmed) { closeList(); continue; }

      const heading = trimmed.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        closeList();
        const level = Math.min(heading[1].length + 1, 4);
        html += `<h${level}>${inline(heading[2])}</h${level}>`;
        continue;
      }

      const bullet = trimmed.match(/^[-*+]\s+(.*)$/);
      if (bullet) {
        if (listType !== "ul") { closeList(); html += "<ul>"; listType = "ul"; }
        html += `<li>${inline(bullet[1])}</li>`;
        continue;
      }

      const numbered = trimmed.match(/^\d+[.)]\s+(.*)$/);
      if (numbered) {
        if (listType !== "ol") { closeList(); html += "<ol>"; listType = "ol"; }
        html += `<li>${inline(numbered[1])}</li>`;
        continue;
      }

      closeList();
      html += `<p>${inline(trimmed)}</p>`;
    }

    closeList();
    return html;
  }

  return { render, escapeHtml };
})();


/* =========================================================
   3. UI — shared DOM helpers and the small bit of app state
   ========================================================= */
const UI = (() => {

  const $  = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const state = {
    mode: "ask",
    provider: "gemini",
    busy: false,
  };

  const chatWindow = () => $("#chatWindow");

  /* --- per-mode prompts shown on the blank page ------------------ */

  const EXAMPLES = {
    ask: {
      lead: "Ask anything about programming.",
      body: "Concepts, syntax, error messages, best practice — anything you would ask a mentor.",
      items: [
        "What is the difference between a list and a tuple in Python?",
        "When should I use async/await instead of threads?",
        "Why does JavaScript say 0.1 + 0.2 is not 0.3?",
      ],
    },
    explain: {
      lead: "Paste code. Get it explained.",
      body: "Works best with a whole function or class. The walkthrough follows the order the code runs in.",
      items: [
        "Explain what a Python decorator does, with an example.",
        "Walk me through what this list comprehension is doing.",
        "What does useEffect with an empty dependency array actually do?",
      ],
    },
    debug: {
      lead: "Paste the error. Get the cause.",
      body: "Include the full traceback if you have one — the line number is usually the fastest clue.",
      items: [
        "Why does my loop give IndexError on the last element?",
        "TypeError: NoneType object is not subscriptable — where do I start?",
        "My Flask route returns 404 but the URL looks right.",
      ],
    },
    optimize: {
      lead: "Paste slow code. Get it faster.",
      body: "You will get the complexity before and after, plus what the change costs you.",
      items: [
        "Make this O(n squared) duplicate check faster.",
        "This pandas loop takes 40 seconds on 100k rows.",
        "How do I speed up repeated string concatenation in a loop?",
      ],
    },
    generate: {
      lead: "Describe it. Get working code.",
      body: "Name the language and any constraints. You get something runnable, not a sketch.",
      items: [
        "A Python script that renames every .jpg in a folder by date taken.",
        "A React hook that debounces a search input.",
        "SQL to find the top 3 customers by revenue in each region.",
      ],
    },
  };

  /* Redraws the blank page for the chosen mode. Does nothing once the
     conversation has started, because the empty state is gone by then. */
  function renderEmptyState(mode) {
    const empty = $("#emptyState");
    if (!empty) return;

    const set = EXAMPLES[mode] || EXAMPLES.ask;
    const buttons = set.items
      .map((text) => `<li><button type="button" class="example">${MD.escapeHtml(text)}</button></li>`)
      .join("");

    empty.innerHTML =
      `<p class="empty__lead">${MD.escapeHtml(set.lead)}</p>` +
      `<p class="empty__body">${MD.escapeHtml(set.body)}</p>` +
      `<ul class="empty__examples">${buttons}</ul>`;
  }


  function hideEmptyState() {
    const empty = $("#emptyState");
    if (empty) empty.remove();
  }

  function scrollToBottom() {
    const win = chatWindow();
    win.scrollTop = win.scrollHeight;
  }

  /* --- messages ------------------------------------------------- */

  function addMessage({ who, label, body, tag, variant }) {
    hideEmptyState();
    const wrap = document.createElement("article");
    wrap.className = `msg msg--${who}${variant ? " msg--" + variant : ""}`;

    const head = document.createElement("div");
    head.className = "msg__who";
    head.innerHTML =
      `<span>${MD.escapeHtml(label)}</span>` +
      (tag ? `<span class="msg__tag">${MD.escapeHtml(tag)}</span>` : "");

    const content = document.createElement("div");
    content.className = "msg__body";
    if (who === "you") {
      content.textContent = body;
    } else {
      content.innerHTML = MD.render(body);
    }

    wrap.append(head, content);
    chatWindow().appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function addPending(label) {
    hideEmptyState();
    const wrap = document.createElement("article");
    wrap.className = "msg msg--ai is-pending";
    wrap.innerHTML =
      `<div class="msg__who"><span>${MD.escapeHtml(label)}</span></div>` +
      `<div class="msg__body"><span class="thinking">thinking</span></div>`;
    chatWindow().appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function addSources(messageEl, sources) {
    if (!sources || !sources.length) return;
    const box = document.createElement("div");
    box.className = "sources";
    box.innerHTML = `<span class="sources__label">Used</span>`;

    sources.forEach((source) => {
      if (source.kind === "web" && source.url) {
        const link = document.createElement("a");
        link.className = "chip chip--web";
        link.href = source.url;
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = source.label;
        box.appendChild(link);
      } else {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = source.label;
        box.appendChild(chip);
      }
    });

    $(".msg__body", messageEl).appendChild(box);
  }

  /* --- toast ---------------------------------------------------- */

  let toastTimer = null;
  function toast(text, variant = "") {
    const el = $("#toast");
    el.textContent = text;
    el.className = "toast" + (variant ? ` toast--${variant}` : "");
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 4200);
  }

  /* --- provider stamps, model picker, footer --------------------- */

  function applyProviders(providers) {
    Object.entries(providers).forEach(([name, ready]) => {
      const stamp = document.querySelector(`.stamp[data-provider="${name}"]`);
      if (stamp) stamp.classList.toggle("stamp--muted", !ready);
    });

    Array.from($("#providerSelect").options).forEach((option) => {
      const base = option.value === "gemini" ? "Gemini" : "OpenAI";
      option.textContent = providers[option.value] ? base : `${base} (no key)`;
    });

    // Point the picker at something that actually works.
    const current = $("#providerSelect").value;
    if (!providers[current]) {
      const ready = Object.entries(providers).find(([, ok]) => ok);
      if (ready) {
        $("#providerSelect").value = ready[0];
        state.provider = ready[0];
      }
    }

    const live = Object.entries(providers)
      .filter(([, ok]) => ok)
      .map(([name]) => (name === "gemini" ? "Gemini" : "OpenAI"));
    $("#footStatus").textContent = live.length
      ? `Connected: ${live.join(" + ")}`
      : "No API key set";
  }

  /* --- busy state ----------------------------------------------- */

  function setBusy(busy) {
    state.busy = busy;
    $("#sendBtn").disabled = busy;
    $("#compareBtn").disabled = busy;
    $("#sendBtn").textContent = busy ? "Sending" : "Send";
  }

  /* --- copy buttons (one listener for the whole window) ---------- */

  function wireCopyButtons() {
    chatWindow().addEventListener("click", (event) => {
      const button = event.target.closest(".copy-btn");
      if (!button) return;
      const code = button.closest(".codeblock").querySelector("code");
      navigator.clipboard.writeText(code.textContent).then(
        () => {
          button.textContent = "Copied";
          setTimeout(() => { button.textContent = "Copy"; }, 1600);
        },
        () => toast("Copying is blocked in this browser. Select the code instead.", "warn")
      );
    });
  }

  return {
    $, $$, state,
    addMessage, addPending, addSources, applyProviders, renderEmptyState,
    toast, setBusy, scrollToBottom, wireCopyButtons,
  };
})();


/* =========================================================
   4. CHAT — sending a question and rendering the answer
   ========================================================= */
const Chat = (() => {

  const { $, state } = UI;

  function currentInput() {
    return $("#messageInput").value.trim();
  }

  async function send() {
    if (state.busy) return;

    const message = currentInput();
    if (!message) {
      UI.toast("Write a question first.", "warn");
      $("#messageInput").focus();
      return;
    }

    const payload = {
      message,
      mode: state.mode,
      provider: state.provider,
      use_rag: $("#useRag").checked,
      use_web: $("#useWeb").checked,
    };

    UI.addMessage({ who: "you", label: "You", body: message, tag: state.mode });
    $("#messageInput").value = "";
    UI.setBusy(true);

    const providerName = $("#providerSelect").selectedOptions[0].textContent;
    const pending = UI.addPending(providerName);

    try {
      const data = await API.chat(payload);
      pending.remove();

      if (data.blocked) {
        UI.addMessage({
          who: "ai",
          label: "Safety check",
          body: data.reason,
          variant: "blocked",
        });
        return;
      }

      const el = UI.addMessage({ who: "ai", label: data.provider, body: data.answer });
      UI.addSources(el, data.sources);
      History.refresh();

    } catch (err) {
      pending.remove();
      UI.addMessage({ who: "ai", label: "Not sent", body: err.message, variant: "error" });
    } finally {
      UI.setBusy(false);
      $("#messageInput").focus();
    }
  }

  return { send, currentInput };
})();


/* =========================================================
   5. UPLOAD — file picking, drag and drop, the file list
   ========================================================= */
const Upload = (() => {

  const { $ } = UI;

  function renderFiles(files) {
    const list = $("#fileList");
    list.innerHTML = "";
    files.forEach((name) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${MD.escapeHtml(name)}</span><span>indexed</span>`;
      list.appendChild(li);
    });
    $("#clearFilesBtn").hidden = files.length === 0;
    if (files.length) $("#useRag").checked = true;
  }

  async function upload(file) {
    if (!file) return;
    UI.toast(`Reading ${file.name}…`);
    try {
      const data = await API.upload(file);
      renderFiles(data.indexed_files);
      UI.toast(`${data.filename} split into ${data.chunks} searchable chunks.`);
    } catch (err) {
      UI.toast(err.message, "warn");
    }
  }

  async function clearAll() {
    try {
      await API.clearIndex();
      renderFiles([]);
      $("#useRag").checked = false;
      UI.toast("Uploaded files removed.");
    } catch (err) {
      UI.toast(err.message, "warn");
    }
  }

  function wire() {
    const zone = $("#dropzone");
    const input = $("#fileInput");

    input.addEventListener("change", () => {
      upload(input.files[0]);
      input.value = "";           // allow re-uploading the same file
    });

    ["dragenter", "dragover"].forEach((type) =>
      zone.addEventListener(type, (event) => {
        event.preventDefault();
        zone.classList.add("is-over");
      })
    );

    ["dragleave", "drop"].forEach((type) =>
      zone.addEventListener(type, (event) => {
        event.preventDefault();
        zone.classList.remove("is-over");
      })
    );

    zone.addEventListener("drop", (event) => {
      upload(event.dataTransfer.files[0]);
    });

    $("#clearFilesBtn").addEventListener("click", clearAll);
  }

  return { wire, renderFiles };
})();


/* =========================================================
   6. COMPARE — the same question, both models, side by side
   ========================================================= */
const Compare = (() => {

  const { $ } = UI;

  function column(result) {
    const body = result.error
      ? `<p class="compare__error">${MD.escapeHtml(result.error)}</p>`
      : MD.render(result.text);
    return (
      `<div class="compare__col">` +
        `<div class="compare__name">${MD.escapeHtml(result.label)}</div>` +
        body +
      `</div>`
    );
  }

  async function run() {
    if (UI.state.busy) return;

    const message = Chat.currentInput();
    if (!message) {
      UI.toast("Write a question first.", "warn");
      return;
    }

    UI.addMessage({ who: "you", label: "You", body: message, tag: "compare" });
    $("#messageInput").value = "";
    UI.setBusy(true);
    const pending = UI.addPending("Gemini + OpenAI");

    try {
      const data = await API.compare({ message, mode: UI.state.mode });
      pending.remove();

      if (data.blocked) {
        UI.addMessage({
          who: "ai", label: "Safety check", body: data.reason, variant: "blocked",
        });
        return;
      }

      const wrap = document.createElement("article");
      wrap.className = "msg msg--ai";
      wrap.innerHTML =
        `<div class="msg__who"><span>Side by side</span></div>` +
        `<div class="compare">${column(data.results.gemini)}${column(data.results.openai)}</div>`;
      $("#chatWindow").appendChild(wrap);
      UI.scrollToBottom();

    } catch (err) {
      pending.remove();
      UI.addMessage({ who: "ai", label: "Not sent", body: err.message, variant: "error" });
    } finally {
      UI.setBusy(false);
    }
  }

  return { run };
})();


/* =========================================================
   7. HISTORY — earlier questions; click one to reuse it
   ========================================================= */
const History = (() => {

  const { $ } = UI;

  function render(items) {
    const list = $("#historyList");
    list.innerHTML = "";

    if (!items.length) {
      list.innerHTML = `<li class="history__empty">Nothing yet. Your questions will collect here.</li>`;
      return;
    }

    items.slice().reverse().slice(0, 25).forEach((item) => {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      // One line each; CSS truncates it, the tooltip carries the rest.
      button.title = item.question;
      button.innerHTML =
        `<span class="history__mode">${MD.escapeHtml(item.mode)} · ${MD.escapeHtml(item.provider)}</span>` +
        MD.escapeHtml(item.question);
      button.addEventListener("click", () => {
        $("#messageInput").value = item.question;
        $("#messageInput").focus();
      });
      li.appendChild(button);
      list.appendChild(li);
    });
  }

  async function refresh() {
    try {
      const data = await API.history();
      render(data.items || []);
    } catch (err) {
      /* history is a convenience — a failure here should not shout */
      render([]);
    }
  }

  async function clear() {
    try {
      await API.clearHistory();
      render([]);
      UI.toast("History cleared.");
    } catch (err) {
      UI.toast(err.message, "warn");
    }
  }

  return { refresh, clear, render };
})();


/* =========================================================
   8. SETTINGS — API keys, saved from the browser

   Keys are written to .env on the machine running the server and
   applied straight away, so no restart is needed. The server only
   ever sends keys back masked.
   ========================================================= */

const Settings = (() => {

  const { $ } = UI;

  const dialog = () => $("#settingsModal");

  function mask(value) {
    if (!value) return "not set";
    return value.length > 12
      ? `saved — ${value.slice(0, 4)}…${value.slice(-4)}`
      : "saved";
  }

  /* Which providers can this browser actually use? Either the visitor has
     pasted a key, or the server was deployed with one of its own. */
  function providerState(serverProviders = {}) {
    const keys = API.loadKeys();
    return {
      gemini: Boolean(keys.gemini_key) || Boolean(serverProviders.gemini),
      openai: Boolean(keys.openai_key) || Boolean(serverProviders.openai),
    };
  }

  let serverProviders = {};
  const setServerProviders = (value) => { serverProviders = value || {}; };

  function refreshStamps() {
    UI.applyProviders(providerState(serverProviders));
  }

  function open() {
    const keys = API.loadKeys();

    $("#geminiKey").value = "";
    $("#openaiKey").value = "";
    $("#geminiModel").value = keys.gemini_model || "";
    $("#openaiModel").value = keys.openai_model || "";
    $("#geminiCurrent").textContent = mask(keys.gemini_key);
    $("#openaiCurrent").textContent = mask(keys.openai_key);
    $("#testResult").hidden = true;

    // never leave a key visible from a previous visit
    UI.$$(".reveal").forEach((button) => {
      $(`#${button.dataset.target}`).type = "password";
      button.textContent = "Show";
    });

    dialog().showModal();
    $("#geminiKey").focus();
  }

  function close() {
    dialog().close();
  }

  /* Blank key boxes mean "leave what is already saved", so a visitor can
     change just the model without re-pasting their key. */
  function collect() {
    const keys = API.loadKeys();
    return {
      gemini_key:   $("#geminiKey").value.trim()   || keys.gemini_key   || "",
      openai_key:   $("#openaiKey").value.trim()   || keys.openai_key   || "",
      gemini_model: $("#geminiModel").value.trim() || "",
      openai_model: $("#openaiModel").value.trim() || "",
    };
  }

  function save() {
    const keys = collect();

    if (!keys.gemini_key && !keys.openai_key) {
      UI.toast("Paste at least one key first.", "warn");
      return;
    }

    if (!API.saveKeys(keys)) {
      UI.toast("This browser is blocking storage — try a normal window.", "warn");
      return;
    }

    refreshStamps();
    close();
    UI.toast("Saved in this browser. Ready to use.");
  }

  function forget() {
    API.clearKeys();
    refreshStamps();
    open();
    UI.toast("Keys removed from this browser.");
  }

  /* Show / Hide — lets you check what you actually pasted. */
  function toggleReveal(button) {
    const input = $(`#${button.dataset.target}`);
    const hidden = input.type === "password";
    input.type = hidden ? "text" : "password";
    button.textContent = hidden ? "Hide" : "Show";
  }

  function showResult(text, kind) {
    const box = $("#testResult");
    box.textContent = text;
    box.className = `modal__result ${kind}`;
    box.hidden = false;
  }

  /* Makes one real call so you know the key works, not just that it saved. */
  async function test(button) {
    const provider = button.dataset.provider;

    button.disabled = true;
    button.textContent = "…";
    showResult("Contacting the provider…", "");

    // Save first so the request carries what is on screen.
    const keys = collect();
    if (keys.gemini_key || keys.openai_key) API.saveKeys(keys);

    try {
      const data = await API.testKey({ provider });
      if (data.ok) {
        showResult(`Working — ${data.label} replied using ${data.model}.`, "is-ok");
        refreshStamps();
      } else {
        showResult(data.error, "is-bad");
      }
    } catch (err) {
      showResult(err.message, "is-bad");
    } finally {
      button.disabled = false;
      button.textContent = "Test";
    }
  }

  function wire() {
    $("#settingsBtn").addEventListener("click", open);
    $("#settingsCancel").addEventListener("click", close);
    $("#settingsSave").addEventListener("click", save);
    $("#settingsForget").addEventListener("click", forget);

    $("#settingsForm").addEventListener("click", (event) => {
      const reveal = event.target.closest(".reveal");
      if (reveal) return toggleReveal(reveal);

      const tester = event.target.closest(".test");
      if (tester) return test(tester);
    });
  }

  return { wire, open, setServerProviders, refreshStamps, providerState };
})();


/* =========================================================
   9. DRAWER — the cards panel slides in on phones
   ========================================================= */

const Drawer = (() => {

  const { $ } = UI;

  function set(open) {
    $("#cards").classList.toggle("is-open", open);
    $("#scrim").hidden = !open;
    $("#panelToggle").setAttribute("aria-expanded", String(open));
  }

  function wire() {
    $("#panelToggle").addEventListener("click", () => set(true));
    $("#drawerClose").addEventListener("click", () => set(false));
    $("#scrim").addEventListener("click", () => set(false));

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") set(false);
    });

    // Picking a file or a past question should get out of the way.
    $("#cards").addEventListener("click", (event) => {
      if (event.target.closest(".history button, .dropzone")) set(false);
    });
  }

  return { wire, close: () => set(false) };
})();


/* =========================================================
   8. STARTUP — wiring only, nothing here does real work
   ========================================================= */
document.addEventListener("DOMContentLoaded", () => {
  const { $, $$, state } = UI;

  /* --- mode buttons --- */
  $$(".mode").forEach((button) => {
    button.addEventListener("click", () => {
      $$(".mode").forEach((b) => {
        b.classList.remove("is-active");
        b.setAttribute("aria-checked", "false");
      });
      button.classList.add("is-active");
      button.setAttribute("aria-checked", "true");
      state.mode = button.dataset.mode;
      UI.renderEmptyState(state.mode);
    });
  });

  /* --- provider --- */
  $("#providerSelect").addEventListener("change", (event) => {
    state.provider = event.target.value;
  });

  /* --- send --- */
  $("#composer").addEventListener("submit", (event) => {
    event.preventDefault();
    Chat.send();
  });

  $("#messageInput").addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      Chat.send();
    }
  });

  /* --- compare --- */
  $("#compareBtn").addEventListener("click", Compare.run);

  /* --- example prompts (delegated: they are re-rendered per mode) --- */
  $("#chatWindow").addEventListener("click", (event) => {
    const button = event.target.closest(".example");
    if (!button) return;
    $("#messageInput").value = button.textContent.trim();
    $("#messageInput").focus();
  });

  /* --- history --- */
  $("#clearHistoryBtn").addEventListener("click", History.clear);

  /* --- uploads, copy buttons, settings, drawer --- */
  Upload.wire();
  UI.wireCopyButtons();
  Settings.wire();
  Drawer.wire();

  /* --- startup: which providers are configured? --- */
  API.health().then((data) => {
    // The server may or may not carry its own key; the browser may or may
    // not have one saved. Either counts as usable.
    Settings.setServerProviders(data.providers);
    Settings.refreshStamps();
    Upload.renderFiles(data.indexed_files || []);

    // Serverless hosts have no persistent disk, so say so rather than
    // letting files quietly disappear between requests.
    if (data.ephemeral) {
      const note = document.createElement("p");
      note.className = "ephemeral-note";
      note.textContent =
        "Uploaded files and history are temporary on this deployment and " +
        "reset when the server sleeps.";
      $("#dropzone").insertAdjacentElement("afterend", note);
    }

    // Nobody can use the app without a key somewhere, so ask up front.
    const usable = Settings.providerState(data.providers);
    if (!usable.gemini && !usable.openai) {
      UI.toast("Add your API key to get started — it stays in this browser.");
      Settings.open();
    }
  }).catch(() => {
    UI.toast("Could not reach the server. Is it still running?", "warn");
  });

  UI.renderEmptyState(state.mode);
  History.refresh();
  $("#messageInput").focus();
});
