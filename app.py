"""CodePilot AI — Flask application entry point.

Requires Python 3.14 or newer.

Every route is thin: it validates input, calls a module, returns JSON.
The real work lives in rag.py, agent.py, safety.py, utils.py and models/.
"""

import os
import secrets
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
    mask_key,
    new_id,
    read_text_file,
    storage_dir,
    update_env,
)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
UPLOAD_DIR: Final[Path] = storage_dir("uploads")
MAX_UPLOAD_MB: Final[int] = 5

#
# Two modes. Locally the settings panel may write .env, because the only
# person who can reach 127.0.0.1 is you. On a public deployment that route
# would let any visitor overwrite your API keys, so it is off unless you
# deliberately switch it on.
#
IS_PRODUCTION: Final[bool] = os.getenv("FLASK_ENV", "development") == "production"

# Set ADMIN_PASSWORD to use the key editor on a live site. Without one the
# editor stays shut in production, because an unprotected route that writes
# .env would let any visitor take over your API keys.
ADMIN_PASSWORD: Final[str] = os.getenv("ADMIN_PASSWORD", "").strip()

ALLOW_KEY_EDITING: Final[bool] = os.getenv(
    "ALLOW_KEY_EDITING",
    "1" if (not IS_PRODUCTION or ADMIN_PASSWORD) else "0",
) == "1"

# Locally there is nothing to protect — only you can reach 127.0.0.1.
REQUIRE_ADMIN: Final[bool] = IS_PRODUCTION and ALLOW_KEY_EDITING

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
    MODELS["gemini"] = GeminiModel()
    MODELS["openai"] = OpenAIModel()


reload_models()

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
        key_editing=ALLOW_KEY_EDITING,
        admin_required=REQUIRE_ADMIN,
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
    model = MODELS.get(provider, MODELS[DEFAULT_PROVIDER])
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
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {
            name: pool.submit(model.generate, prompt)
            for name, model in MODELS.items()
        }
        replies = {name: future.result() for name, future in futures.items()}

    results: dict[str, dict[str, str | None]] = {}
    for name, reply in replies.items():
        results[name] = {
            "label": MODELS[name].label,
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
# Settings — lets the browser save API keys instead of editing .env by hand
#
# This writes to .env on the machine running the server. That is safe for
# local use, which is what this app is for. Do not expose this route on a
# public deployment without adding authentication first.
# --------------------------------------------------------------------------
ENV_FIELDS: Final[dict[str, str]] = {
    "gemini_key": "GEMINI_API_KEY",
    "openai_key": "OPENAI_API_KEY",
    "gemini_model": "GEMINI_MODEL",
    "openai_model": "OPENAI_MODEL",
}


def _key_guard() -> tuple[Response, int] | None:
    """None when the caller may edit keys, otherwise the refusal to send."""
    if not ALLOW_KEY_EDITING:
        return jsonify(
            error=(
                "Editing keys from the browser is switched off here. Set "
                "ADMIN_PASSWORD on the host to turn it on."
            )
        ), 403

    if REQUIRE_ADMIN:
        given = request.headers.get("X-Admin-Password", "")
        # compare_digest so a wrong guess takes the same time as a right one
        if not (given and secrets.compare_digest(given, ADMIN_PASSWORD)):
            return jsonify(error="Wrong admin password."), 401

    return None


@app.get("/api/settings")
def get_settings() -> tuple[Response, int] | Response:
    if (refusal := _key_guard()) is not None:
        return refusal
    """Current state. Keys come back masked — never in full."""
    return jsonify(
        gemini_key=mask_key(os.getenv("GEMINI_API_KEY", "")),
        openai_key=mask_key(os.getenv("OPENAI_API_KEY", "")),
        gemini_model=MODELS["gemini"].model_name,
        openai_model=MODELS["openai"].model_name,
        providers={name: model.available for name, model in MODELS.items()},
    )


@app.post("/api/settings")
def save_settings() -> tuple[Response, int] | Response:
    if (refusal := _key_guard()) is not None:
        return refusal

    data: dict[str, Any] = request.get_json(silent=True) or {}

    updates = {
        env_name: value
        for field, env_name in ENV_FIELDS.items()
        if (value := str(data.get(field, "")).strip())
    }
    if not updates:
        return jsonify(error="Nothing to save — every field was blank."), 400

    try:
        update_env(updates, ENV_PATH)
    except OSError as exc:
        return jsonify(error=f"Could not write .env: {exc}"), 500

    # Apply immediately so the user does not have to restart the server.
    os.environ.update(updates)
    reload_models()

    return jsonify(
        ok=True,
        saved=sorted(updates),
        providers={name: model.available for name, model in MODELS.items()},
    )


@app.post("/api/test-key")
def test_key() -> tuple[Response, int] | Response:
    """Make one tiny real call, so the user can confirm a key actually works."""
    if (refusal := _key_guard()) is not None:
        return refusal

    data: dict[str, Any] = request.get_json(silent=True) or {}
    provider = str(data.get("provider", DEFAULT_PROVIDER))

    model = MODELS.get(provider)
    if model is None:
        return jsonify(error=f"Unknown provider '{provider}'."), 400

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
