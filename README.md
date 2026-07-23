# CodePilot AI

An intelligent coding mentor and debugging assistant. Ask a programming
question, paste code to explain, debug or optimize, upload your own files and
chat with them, search live documentation, and compare how two different
language models answer the same question.

Built as a full-stack capstone: plain HTML/CSS/JavaScript on the front,
Python + Flask on the back.

---

## What it does

| Feature | How it works |
|---|---|
| Ask, Explain, Debug, Optimize, Generate | Five prompt templates in `prompts/`, chosen by the mode buttons |
| Two LLMs | Google Gemini and OpenAI, behind one shared interface in `models/` |
| Compare both | Sends the same prompt to both models, renders answers side by side |
| Chat with your code (RAG) | Files are chunked, embedded locally, and stored in a FAISS index |
| Live documentation search (Agent) | DuckDuckGo search, with official docs sorted to the top |
| Safety guardrails | Blocks attack-tooling requests, strips prompt injection, masks leaked keys |
| Chat history | Saved to `data/history.json`, listed in the sidebar, click to reuse |
| API keys from the browser | The **API keys** button writes to `.env` and reloads the models — no restart, no text editor |

---

## Requirements

**Python 3.14 or newer.** Check yours:

```bash
python --version
```

If it prints anything below `3.14`, install it from
[python.org/downloads](https://www.python.org/downloads/) first — the rest of
the setup will not work otherwise.

The code uses two things that only exist in 3.14: `uuid.uuid7()` for
time-ordered chat-history ids, and deferred annotation evaluation (PEP 649),
which is why there is no `from __future__ import annotations` anywhere despite
the type hints throughout.

---

## Running it from the terminal

Copy these one line at a time.

**1. Go into the project folder**

```bash
cd path/to/CodePilot-AI
```

**2. Create a virtual environment** — keeps this project's packages separate
from the rest of your computer.

```bash
python -m venv venv
```

**3. Activate it.** Do this every time you open a new terminal.

```bash
# macOS / Linux
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows Command Prompt
venv\Scripts\activate.bat
```

Your prompt should now start with `(venv)`. That is how you know it worked.

**4. Install the packages**

```bash
pip install -r requirements.txt
```

This takes a few minutes — FAISS and the embedding library are large.

**5. Add your API keys**

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Open `.env` in any editor and paste in at least one key:

- `GEMINI_API_KEY` — free from [Google AI Studio](https://aistudio.google.com/app/apikey)

Model IDs move fast — Google retires old ones and they start returning 404.
The default is set in `.env` as `GEMINI_MODEL`. If it ever stops working,
check [the current list](https://ai.google.dev/gemini-api/docs/models) and
paste a new ID in.
- `OPENAI_API_KEY` — from [OpenAI](https://platform.openai.com/api-keys)

**6. Start the server**

```bash
python app.py
```

You should see:

```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

**7. Open it** — go to <http://127.0.0.1:5000> in your browser.

To stop the server, press `Ctrl + C` in the terminal. To leave the virtual
environment, type `deactivate`.

The first time you upload a file, the app downloads a small embedding model
(~90 MB). That happens once and runs on your machine, not through an API.

---

## If something goes wrong

| What you see | What it means |
|---|---|
| `python: command not found` | Try `py` instead of `python` on Windows |
| `404 ... model not found` | Google retired that model ID. Set `GEMINI_MODEL` in `.env` to a current one from [the models page](https://ai.google.dev/gemini-api/docs/models), then restart |
| `FutureWarning: google.generativeai has ended` | You have the old SDK. Run `pip install google-genai` |
| `duckduckgo_search has been renamed to ddgs` | Run `pip install ddgs` |
| `AttributeError: module 'uuid' has no attribute 'uuid7'` | You are on Python 3.13 or older. Install 3.14 and rebuild the virtual environment |
| `No module named flask` | The virtual environment is not active — redo step 3 |
| Header shows both models struck through | No keys found. Check `.env` is named exactly `.env`, then restart the server |
| `Gemini is not configured` | Same — the key is missing or has a typo |
| Upload says "search libraries are not installed" | `pip install -r requirements.txt` did not finish. Run it again |
| `Address already in use` | Another server is on port 5000. Change the port at the bottom of `app.py` |
| Blank page, no styling | Hard-refresh the browser: `Ctrl + Shift + R` |

---

## Setup summary

If you just want the short version:

```bash
cd CodePilot-AI
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your keys into .env
python app.py
```

---

## Project structure

```
CodePilot-AI/
├── app.py                  Flask routes — thin, they only validate and delegate
├── rag.py                  Chunking, embedding, FAISS index, retrieval
├── agent.py                Web search for documentation
├── safety.py               Prompt screening, injection stripping, key masking
├── utils.py                Prompt building, file rules, chat history
├── pyproject.toml          Project metadata; pins Python >= 3.14
├── requirements.txt
├── .env.example            Copy to .env and fill in
│
├── models/                 The one place a package earns its keep
│   ├── base.py             The contract both providers implement
│   ├── gemini.py           LLM #1
│   └── openai_model.py     LLM #2
│
├── templates/
│   └── index.html          The one page
│
├── static/
│   ├── css/
│   │   └── style.css       One stylesheet, 5 numbered sections
│   └── js/
│       └── script.js       One script, 8 numbered sections
│
├── prompts/                One text file per mode — edit these to change tone
├── uploads/                Your uploaded files land here
├── data/                   FAISS index and chat history
└── tests/                  pytest suite
```

---

## How a request flows

```
Browser
  └─ chat.js  ──POST /api/chat──▶  app.py
                                     │
                                     ├─ safety.py        block or clean the prompt
                                     ├─ rag.py           search your uploaded files
                                     ├─ agent.py         look up documentation
                                     ├─ utils.py         assemble the final prompt
                                     ├─ models/gemini.py call the model
                                     └─ safety.py        mask anything secret
  ◀── JSON {answer, sources} ────────┘
```

---

## Running the tests

```bash
pytest
```

The suite covers the safety layer (blocking, injection stripping, key masking)
and the helpers (prompt assembly, file rules, history).

---

## Notes

- `.env` is in `.gitignore`. Never commit real keys.
- Uploads are capped at 5 MB and limited to text and source-code extensions.
- The RAG index persists in `data/`. Use **Clear all files** in the sidebar to wipe it.
- If a model has no key, the app says so in the header instead of failing silently.

---

## Deploying

The app runs in two postures. Locally the settings panel may write `.env`,
because the only person who can reach `127.0.0.1` is you. In production that
route is closed — otherwise any visitor could overwrite your API keys.

Setting `FLASK_ENV=production` does three things:

- `/api/settings` and `/api/test-key` return 403, and the **API keys** button
  disappears from the UI
- the Werkzeug debugger cannot turn on (it allows remote code execution)
- the app refuses to start on the default `FLASK_SECRET_KEY`

### Free hosting on Render

Render is the only major host still offering a permanent free tier without a
card. It sleeps after 15 minutes idle and takes ~40 seconds to wake, which is
fine for a demo.

**1. Push to GitHub**

```bash
git init
git add .
git commit -m "CodePilot AI"
git branch -M main
git remote add origin https://github.com/<your-username>/CodePilot-AI.git
git push -u origin main
```

Check `.env` is *not* in the commit — `.gitignore` covers it, but verify with
`git status` before pushing.

**2. Create the service**

On [render.com](https://render.com): New → Web Service → connect your repo.
The included `render.yaml` sets everything except the secrets.

**3. Add your keys**

In the Render dashboard, under Environment, add:

| Key | Value |
|---|---|
| `GEMINI_API_KEY` | your key |
| `OPENAI_API_KEY` | your key, or leave it out |

`FLASK_SECRET_KEY` is generated for you. Deploy.

### Two things to know about the free tier

**Embeddings run through the API, not locally.** `sentence-transformers`
pulls in PyTorch at roughly 2 GB installed, which will not fit in 512 MB of
RAM. On deployment `EMBED_BACKEND=gemini` sends embedding work to the Gemini
API instead. Same RAG behaviour, no large dependency. To go back to local
embeddings on your own machine:

```bash
pip install -r requirements-local.txt   # then set EMBED_BACKEND=local
```

**Uploads do not survive a restart.** Render's free tier has no persistent
disk, so `uploads/` and `data/` are wiped whenever the service sleeps or
redeploys. Uploaded files and chat history are per-session. Fixing that
properly means a paid disk or moving the index to a hosted vector store —
worth mentioning as future work rather than pretending it is solved.

### Deploying to Vercel instead

Vercel runs Python as serverless functions rather than a long-lived server,
which changes what works. Python 3.14 is supported and Flask is detected
automatically from `app.py`, so the chat itself deploys cleanly. The included
`vercel.json` handles routing.

```bash
npm i -g vercel
vercel
vercel env add GEMINI_API_KEY
vercel env add FLASK_ENV        # value: production
vercel --prod
```

**What does not survive.** There is no persistent disk, so `uploads/` and
`data/` are redirected to `/tmp`, which is wiped on every cold start. Uploaded
files, the FAISS index and chat history last for the life of one warm
instance — often only minutes. The app detects this and shows a notice in the
sidebar rather than letting files vanish silently.

**The timeout.** Vercel allows 10 seconds per request on the free plan and 60
on Pro. A single model reply usually fits; a reply plus web search plus
retrieval may not. `vercel.json` requests 60 seconds, which only takes effect
on Pro. This is why the compare route calls both models concurrently rather
than one after the other.

**Which to choose.** If RAG and history matter — and they are two of the six
things this project sets out to demonstrate — Render is the better host,
because a normal server process keeps its disk. Vercel is the better choice
if you mainly want the chat, and want a fast global URL to share.

### Running it as a demo

Because the free tier sleeps, open the URL a minute before you present so the
first request has already woken it.

---

## License

MIT — see [LICENSE](LICENSE).

Copyright (c) 2026 Anand Tripathi.
