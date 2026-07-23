"""CodePilot AI — Flask application entry point.

Requires Python 3.14 or newer.

Every route is thin: it validates input, calls a module, returns JSON.
The real work lives in rag.py, agent.py, safety.py, utils.py and models/.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

try:
    from flask_cors import CORS
except ImportError:  # optional — only needed if you serve the UI elsewhere
    CORS = None

load_dotenv()

import rag
from agent import search_docs
from models.base import BaseModel
from models.gemini import GeminiModel
from models.openai_model import OpenAIModel
from safety import check_prompt, check_response
from utils import (
    ALLOWED_EXTENSIONS,
    IS_SERVERLESS,
    History,
    build_prompt,
    is_allowed_file,
    new_id,
    read_text_file,
    storage_dir,
)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
UPLOAD_DIR: Final[Path] = storage_dir("uploads")
MAX_UPLOAD_MB: Final[int] = 5

IS_PRODUCTION: Final[bool] = os.getenv("FLASK_ENV", "development") == "production"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")

if IS_PRODUCTION and app.config["SECRET_KEY"] == "dev-secret":
    raise RuntimeError(
        "Set FLASK_SECRET_KEY to a long random value before deploying. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
if CORS is not None:
    CORS(app)

ENV_PATH: Final[Path] = BASE_DIR / ".env"
DEFAULT_PROVIDER: Final[str] = "gemini"

# Rebuilt whenever keys change, so a new key takes effect without a restart.
MODELS: dict[str, BaseModel] = {}


def reload_models() -> None:
    """The server's own models, from environment variables. Optional —
    the app works with no server keys at all if visitors bring their own."""
    MODELS["gemini"] = GeminiModel()
    MODELS["openai"] = OpenAIModel()


reload_models()

KEY_HEADERS: Final[dict[str, tuple[str, str]]] = {
    "gemini": ("X-Gemini-Key", "X-Gemini-Model"),
    "openai": ("X-OpenAI-Key", "X-OpenAI-Model"),
}
BUILDERS: Final[dict[str, type[BaseModel]]] = {
    "gemini": GeminiModel,
    "openai": OpenAIModel,
}


def request_models() -> dict[str, BaseModel]:
    """The models to use for this one request.

    A visitor's key arrives as a header, is used for the length of the
    request and is never written down. If they did not send one, the
    server's own key is used instead — if it has one.
    """
    resolved: dict[str, BaseModel] = {}
    for name, (key_header, model_header) in KEY_HEADERS.items():
        key = request.headers.get(key_header, "").strip()
        model = request.headers.get(model_header, "").strip()
        if key:
            resolved[name] = BUILDERS[name](key, model)
        elif model and MODELS[name].available:
            resolved[name] = BUILDERS[name]("", model)
        else:
            resolved[name] = MODELS[name]
    return resolved

history = History(storage_dir("data") / "history.json")


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
@app.get("/")
def index() -> str:
    return render_template("index.html")


# --------------------------------------------------------------------------
# Status — the frontend uses this to show which providers are configured
# --------------------------------------------------------------------------
@app.get("/api/health")
def health() -> Response:
    return jsonify(
        ok=True,
        providers={name: model.available for name, model in MODELS.items()},
        index_ready=rag.has_index(),
        indexed_files=rag.list_files(),
        ephemeral=IS_SERVERLESS,
    )


# --------------------------------------------------------------------------
# Main chat
# --------------------------------------------------------------------------
@app.post("/api/chat")
def chat() -> tuple[Response, int] | Response:
    data: dict[str, Any] = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    mode = str(data.get("mode", "ask"))
    provider = str(data.get("provider", DEFAULT_PROVIDER))
    use_rag = bool(data.get("use_rag"))
    use_web = bool(data.get("use_web"))

    if not message:
        return jsonify(error="Type a question first."), 400

    verdict = check_prompt(message)
    if not verdict.allowed:
        return jsonify(blocked=True, reason=verdict.reason)

    context_parts: list[str] = []
    sources: list[dict[str, str]] = []

    if use_rag and rag.has_index():
        if hits := rag.search(message, k=4):
            context_parts.append(rag.as_context(hits))
            sources += [{"kind": "file", "label": h["source"]} for h in hits]

    if use_web:
        if results := search_docs(message, max_results=4):
            context_parts.append(
                "\n".join(
                    f"{r['title']} — {r['snippet']} ({r['url']})" for r in results
                )
            )
            sources += [
                {"kind": "web", "label": r["title"], "url": r["url"]} for r in results
            ]

    prompt = build_prompt(mode, verdict.cleaned_prompt, "\n\n".join(context_parts))
    models = request_models()
    model = models.get(provider, models[DEFAULT_PROVIDER])
    reply = model.generate(prompt)

    if reply["error"]:
        return jsonify(error=reply["error"]), 502

    answer = check_response(reply["text"])
    history.add(
        {
            "id": new_id(),
            "mode": mode,
            "provider": provider,
            "question": message,
            "answer": answer,
        }
    )
    return jsonify(answer=answer, provider=model.label, sources=sources)


# --------------------------------------------------------------------------
# Side-by-side comparison of both LLMs
# --------------------------------------------------------------------------
@app.post("/api/compare")
def compare() -> tuple[Response, int] | Response:
    data: dict[str, Any] = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    mode = str(data.get("mode", "ask"))

    if not message:
        return jsonify(error="Type a question first."), 400

    verdict = check_prompt(message)
    if not verdict.allowed:
        return jsonify(blocked=True, reason=verdict.reason)

    prompt = build_prompt(mode, verdict.cleaned_prompt)

    # Both providers are called at once rather than one after the other.
    # Sequentially this route takes as long as both replies added together,
    # which is the difference between fitting a 10 second serverless
    # timeout and not.
    models = request_models()
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = {
            name: pool.submit(model.generate, prompt)
            for name, model in models.items()
        }
        replies = {name: future.result() for name, future in futures.items()}

    results: dict[str, dict[str, str | None]] = {}
    for name, reply in replies.items():
        results[name] = {
            "label": models[name].label,
            "text": "" if reply["error"] else check_response(reply["text"]),
            "error": reply["error"],
        }
    return jsonify(results=results)


# --------------------------------------------------------------------------
# File upload -> RAG index
# --------------------------------------------------------------------------
@app.post("/api/upload")
def upload() -> tuple[Response, int] | Response:
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="No file was attached."), 400

    if not is_allowed_file(file.filename):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify(error=f"Only these types work: {allowed}"), 400

    filename = secure_filename(file.filename)
    path = UPLOAD_DIR / filename
    file.save(path)

    text = read_text_file(path)
    if not text.strip():
        return jsonify(error="That file is empty or unreadable."), 400

    try:
        chunks = rag.add_document(filename, text)
    except ModuleNotFoundError as exc:
        return jsonify(
            error=(
                f"The search libraries are not installed ({exc.name}). "
                "Run: pip install -r requirements.txt"
            )
        ), 503
    except Exception as exc:
        return jsonify(error=f"Could not index that file: {exc}"), 500

    return jsonify(filename=filename, chunks=chunks, indexed_files=rag.list_files())


@app.post("/api/clear-index")
def clear_index() -> Response:
    rag.reset()
    return jsonify(ok=True, indexed_files=[])


# --------------------------------------------------------------------------
# Key check
#
# There is deliberately no route that saves keys. Visitors keep their own
# key in their own browser and send it with each request, so the server has
# nothing worth stealing and no one can overwrite anyone else's settings.
# --------------------------------------------------------------------------
@app.post("/api/test-key")
def test_key() -> tuple[Response, int] | Response:
    """Make one tiny real call, so a visitor can confirm their key works."""
    data: dict[str, Any] = request.get_json(silent=True) or {}
    provider = str(data.get("provider", DEFAULT_PROVIDER))

    if provider not in BUILDERS:
        return jsonify(error=f"Unknown provider '{provider}'."), 400

    model = request_models()[provider]
    reply = model.generate("Reply with the single word: OK")
    if reply["error"]:
        return jsonify(ok=False, error=reply["error"])

    return jsonify(ok=True, model=model.model_name, label=model.label)


# --------------------------------------------------------------------------
# Chat history
# --------------------------------------------------------------------------
@app.get("/api/history")
def get_history() -> Response:
    return jsonify(items=history.all())


@app.delete("/api/history")
def clear_history() -> Response:
    history.clear()
    return jsonify(ok=True)


# --------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(_: Exception) -> tuple[Response, int]:
    return jsonify(error=f"That file is over the {MAX_UPLOAD_MB} MB limit."), 413


if __name__ == "__main__":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=not IS_PRODUCTION and os.getenv("FLASK_DEBUG", "1") == "1",
    )
