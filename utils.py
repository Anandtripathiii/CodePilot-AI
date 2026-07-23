"""Small shared helpers: prompt building, file rules, chat history."""

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Final

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
PROMPT_DIR: Final[Path] = BASE_DIR / "prompts"

# True on Vercel, AWS Lambda and similar. On these platforms the project
# directory is read-only and only /tmp can be written to — and even that is
# wiped between cold starts.
IS_SERVERLESS: Final[bool] = bool(
    os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
)


def storage_dir(name: str) -> Path:
    """Where a writable folder should live on this platform.

    Normally alongside the code. On serverless hosts that directory is
    read-only, so everything goes to /tmp instead — which works for the life
    of one warm instance and no longer.
    """
    root = Path(os.getenv("STORAGE_ROOT", "/tmp" if IS_SERVERLESS else BASE_DIR))
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path

ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h",
    ".cs", ".go", ".rb", ".php", ".rs", ".swift", ".kt", ".sql",
    ".html", ".css", ".json", ".yml", ".yaml", ".md", ".txt", ".sh",
})

MODE_FILES: Final[dict[str, str]] = {
    "ask": "ask",
    "explain": "explain",
    "debug": "debug",
    "optimize": "optimize",
    "generate": "generate",
}


def new_id() -> str:
    """A short, time-ordered id.

    uuid7 (new in Python 3.14) embeds a timestamp in its leading bits, so
    ids sort chronologically. Chat history is chronological, so this means
    the id alone is enough to order entries.
    """
    return uuid.uuid7().hex[:12]


def is_allowed_file(filename: str) -> bool:
    return Path(filename.lower()).suffix in ALLOWED_EXTENSIONS


def read_text_file(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# .env writing — lets the settings panel save keys without a text editor
# --------------------------------------------------------------------------
def update_env(updates: dict[str, str], path: Path) -> None:
    """Rewrite matching KEY=value lines in place, appending any that are new.

    Comments, blank lines and ordering are preserved, so the file stays
    readable after the UI has written to it.
    """
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    written: set[str] = set()
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}")
                written.add(key)
                continue
        out.append(line)

    out.extend(f"{key}={value}" for key, value in updates.items() if key not in written)
    path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")


def mask_key(value: str) -> str:
    """Show enough of a key to recognise it, never enough to use it."""
    value = value.strip()
    if not value:
        return ""
    return f"{value[:4]}…{value[-4:]}" if len(value) > 12 else "set"


# --------------------------------------------------------------------------
# Prompts — one text file per mode, loaded once and cached
# --------------------------------------------------------------------------
_cache: dict[str, str] = {}


def load_prompt(name: str) -> str:
    if name not in _cache:
        path = PROMPT_DIR / f"{name}.txt"
        _cache[name] = read_text_file(path) if path.exists() else ""
    return _cache[name]


def build_prompt(mode: str, question: str, context: str = "") -> str:
    """Assemble system prompt + task template + retrieved context + question."""
    parts = [
        load_prompt("system_prompt"),
        load_prompt(MODE_FILES.get(mode, "ask")),
    ]

    if context.strip():
        parts.append(
            "Reference material retrieved for this question. Use it if it is "
            "relevant, ignore it if it is not, and say so when it does not "
            "cover the question:\n"
            f"{context.strip()}"
        )

    parts.append(f"USER REQUEST:\n{question}")
    return "\n\n".join(part for part in parts if part.strip())


# --------------------------------------------------------------------------
# Chat history — a JSON file, newest last
# --------------------------------------------------------------------------
class History:
    """Append-only chat log, trimmed to the most recent `limit` entries."""

    def __init__(self, path: Path | str, limit: int = 200) -> None:
        self.path = Path(path)
        self.limit = limit
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _write(self, items: list[dict[str, Any]]) -> None:
        self.path.write_text(
            json.dumps(items[-self.limit:], indent=2), encoding="utf-8"
        )

    def add(self, item: dict[str, Any]) -> None:
        with self._lock:
            items = self._read()
            items.append(item)
            self._write(items)

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._read()

    def clear(self) -> None:
        with self._lock:
            self._write([])
