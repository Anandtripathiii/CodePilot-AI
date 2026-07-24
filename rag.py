"""Retrieval-augmented generation.

The whole pipeline in one place, in the order it runs:

    1. Chunking   — split a file into overlapping pieces
    2. Embedding  — turn each piece into a vector
    3. Storage    — keep the vectors in a FAISS index on disk
    4. Retrieval  — find the pieces closest to a question

There are two embedding backends:

    gemini  — calls the Gemini embedding API. No large dependency, so it
              runs in 512 MB of RAM. This is the default when a key exists
              and the only option that fits a free hosting tier.
    local   — sentence-transformers on your own machine. No API calls and
              works offline, but pulls in PyTorch (~2 GB installed).

Set EMBED_BACKEND=gemini or =local in .env to force one.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Final, TypedDict

import numpy as np

from utils import storage_dir

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = storage_dir("data")

GEMINI_EMBED_MODEL: Final[str] = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_DIM: Final[int] = int(os.getenv("EMBED_DIM", "768"))
LOCAL_EMBED_MODEL: Final[str] = "all-MiniLM-L6-v2"
LOCAL_DIM: Final[int] = 384

MIN_SCORE: Final[float] = 0.25       # cosine floor; below this a chunk is noise

_lock = threading.Lock()
_local_model: Any = None

# One index per visitor. Without this, every visitor would read and write
# the same vectors — meaning anyone could search anyone else's uploads.
# The key is the session id; the value is that visitor's loaded state.
_spaces: dict[str, dict[str, Any]] = {}


def _paths(space: str) -> tuple[Path, Path]:
    folder = DATA_DIR / space
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "faiss.index", folder / "chunks.json"


class Chunk(TypedDict):
    text: str
    source: str
    score: float


def backend() -> str:
    """Which embedding backend this deployment uses.

    Deliberately independent of who is asking. The vector dimension follows
    from this, and a stored index cannot change dimension halfway through —
    so it must not depend on which visitor happens to be uploading.
    """
    choice = os.getenv("EMBED_BACKEND", "gemini").strip().lower()
    return "local" if choice == "local" else "gemini"


def dimension() -> int:
    return GEMINI_DIM if backend() == "gemini" else LOCAL_DIM


# ==========================================================================
# 1. Chunking
# ==========================================================================
def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split code or prose into overlapping chunks.

    The separators are ordered so a split lands on a class or function
    boundary where possible, which keeps each chunk self-contained.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


# ==========================================================================
# 2. Embedding
# ==========================================================================
def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length.

    The index below uses inner product, which equals cosine similarity only
    for unit vectors. Gemini's shortened outputs are not normalised, so this
    is required rather than cosmetic.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class MissingKey(RuntimeError):
    """No Gemini key available to embed with. Raised with a message meant
    to be shown to the person, not logged and swallowed."""


def _embed_gemini(texts: list[str], is_query: bool, api_key: str = "") -> np.ndarray:
    # Checked before the import so a missing key is reported as a missing
    # key, rather than as whichever error happens to surface first.
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not key:
        raise MissingKey(
            "Indexing a file needs a Gemini key. Add one with the 'API keys' "
            "button — it is used to turn your code into searchable vectors."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)

    response = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=GEMINI_DIM,
            task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
        ),
    )
    return np.asarray([e.values for e in response.embeddings], dtype="float32")


def _embed_local(texts: list[str]) -> np.ndarray:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(LOCAL_EMBED_MODEL)
    return np.asarray(_local_model.encode(texts), dtype="float32")


def embed(
    texts: str | list[str], is_query: bool = False, api_key: str = ""
) -> np.ndarray:
    """Return a float32 array of shape (n, dimension()), unit length."""
    if isinstance(texts, str):
        texts = [texts]

    if backend() == "gemini":
        vectors = _embed_gemini(texts, is_query, api_key)
    else:
        vectors = _embed_local(texts)

    return _normalise(vectors)


# ==========================================================================
# 3. Storage
# ==========================================================================
def _faiss() -> Any:
    import faiss

    return faiss


def _disk_mtime(index_path: Path) -> float:
    """When this visitor's index last changed, or -1 if there is none."""
    try:
        return index_path.stat().st_mtime
    except OSError:
        return -1.0


def _load(space: str) -> dict[str, Any]:
    """This visitor's index, reloaded from disk if it has changed.

    The web server runs several worker processes, each with its own memory.
    An upload handled by one worker is invisible to the others unless they
    notice the file changed, so the timestamp is checked on every call. A
    stat is cheap; workers disagreeing about which files exist is not.

    An index built by a different embedder holds vectors the current one
    cannot read, so it is discarded rather than mixed.
    """
    index_path, meta_path = _paths(space)
    state = _spaces.get(space)
    mtime = _disk_mtime(index_path)

    if state is not None and state["mtime"] == mtime:
        return state

    faiss = _faiss()
    want = dimension()

    if mtime >= 0 and meta_path.exists():
        stored = faiss.read_index(str(index_path))
        if stored.d == want:
            state = {
                "index": stored,
                "meta": json.loads(meta_path.read_text(encoding="utf-8")),
                "mtime": mtime,
            }
            _spaces[space] = state
            return state

    state = {"index": faiss.IndexFlatIP(want), "meta": [], "mtime": mtime}
    _spaces[space] = state
    return state


def _save(space: str, state: dict[str, Any]) -> None:
    index_path, meta_path = _paths(space)
    # Metadata first: a worker reloading mid-write should never find a new
    # index pointing at old metadata.
    meta_path.write_text(json.dumps(state["meta"]), encoding="utf-8")
    _faiss().write_index(state["index"], str(index_path))
    state["mtime"] = _disk_mtime(index_path)


def add_document(space: str, filename: str, text: str, api_key: str = "") -> int:
    """Chunk, embed and store one file. Returns the number of chunks added."""
    with _lock:
        state = _load(space)
        chunks = split_text(text)
        if not chunks:
            return 0

        state["index"].add(embed(chunks, api_key=api_key))
        state["meta"].extend({"text": c, "source": filename} for c in chunks)
        _save(space, state)
        return len(chunks)


def has_index(space: str) -> bool:
    """False — rather than a crash — if FAISS is missing or nothing is stored."""
    try:
        return _load(space)["index"].ntotal > 0
    except Exception:
        return False


def list_files(space: str) -> list[str]:
    """Every filename in this visitor's index, in the order added."""
    try:
        meta = _load(space)["meta"]
    except Exception:
        return []
    return list(dict.fromkeys(item["source"] for item in meta))


def reset(space: str) -> None:
    """Wipe this visitor's index — used by the 'Clear all files' button."""
    with _lock:
        _spaces.pop(space, None)
        for path in _paths(space):
            path.unlink(missing_ok=True)


# ==========================================================================
# 4. Retrieval
# ==========================================================================
def search(
    space: str, question: str, k: int = 4, api_key: str = ""
) -> list[Chunk]:
    """The chunks most relevant to a question, weak matches dropped."""
    with _lock:
        state = _load(space)
        index = state["index"]
        if index.ntotal == 0:
            return []

        vector = embed(question, is_query=True, api_key=api_key)
        scores, ids = index.search(vector, min(k, index.ntotal))
        meta = state["meta"]

    hits: list[Chunk] = []
    for score, idx in zip(scores[0], ids[0], strict=True):
        if idx == -1 or float(score) < MIN_SCORE:
            continue
        item = meta[int(idx)]
        hits.append(Chunk(text=item["text"], source=item["source"], score=float(score)))
    return hits


def as_context(hits: list[Chunk]) -> str:
    """Format hits into a block that can be pasted into a prompt."""
    return "\n\n".join(f"--- from {h['source']} ---\n{h['text']}" for h in hits)
