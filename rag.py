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
INDEX_PATH: Final[Path] = DATA_DIR / "faiss.index"
META_PATH: Final[Path] = DATA_DIR / "chunks.json"

GEMINI_EMBED_MODEL: Final[str] = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_DIM: Final[int] = int(os.getenv("EMBED_DIM", "768"))
LOCAL_EMBED_MODEL: Final[str] = "all-MiniLM-L6-v2"
LOCAL_DIM: Final[int] = 384

MIN_SCORE: Final[float] = 0.25       # cosine floor; below this a chunk is noise

_lock = threading.Lock()
_local_model: Any = None
_gemini_client: Any = None
_index: Any = None
_meta: list[dict[str, Any]] = []


class Chunk(TypedDict):
    text: str
    source: str
    score: float


def backend() -> str:
    """Which embedding backend is in play right now."""
    choice = os.getenv("EMBED_BACKEND", "auto").strip().lower()
    if choice in {"gemini", "local"}:
        return choice
    return "gemini" if os.getenv("GEMINI_API_KEY", "").strip() else "local"


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


def _embed_gemini(texts: list[str], is_query: bool) -> np.ndarray:
    from google import genai
    from google.genai import types

    global _gemini_client
    if _gemini_client is None:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is needed for the gemini embedder.")
        _gemini_client = genai.Client(api_key=key)

    response = _gemini_client.models.embed_content(
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


def embed(texts: str | list[str], is_query: bool = False) -> np.ndarray:
    """Return a float32 array of shape (n, dimension()), unit length."""
    if isinstance(texts, str):
        texts = [texts]

    if backend() == "gemini":
        vectors = _embed_gemini(texts, is_query)
    else:
        vectors = _embed_local(texts)

    return _normalise(vectors)


# ==========================================================================
# 3. Storage
# ==========================================================================
def _faiss() -> Any:
    import faiss

    return faiss


def _load() -> None:
    """Read the index from disk, or start an empty one.

    If the stored index was built with a different embedder its vectors are
    meaningless to the current one, so it is discarded rather than mixed.
    """
    global _index, _meta
    if _index is not None:
        return

    faiss = _faiss()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    want = dimension()

    if INDEX_PATH.exists() and META_PATH.exists():
        stored = faiss.read_index(str(INDEX_PATH))
        if stored.d == want:
            _index = stored
            _meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            return

    _index = faiss.IndexFlatIP(want)
    _meta = []


def _save() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _faiss().write_index(_index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(_meta), encoding="utf-8")


def add_document(filename: str, text: str) -> int:
    """Chunk, embed and store one file. Returns the number of chunks added."""
    with _lock:
        _load()
        chunks = split_text(text)
        if not chunks:
            return 0

        _index.add(embed(chunks))
        _meta.extend({"text": c, "source": filename} for c in chunks)
        _save()
        return len(chunks)


def has_index() -> bool:
    """False — rather than a crash — if FAISS is missing or nothing is stored."""
    try:
        _load()
    except Exception:
        return False
    return _index.ntotal > 0


def list_files() -> list[str]:
    """Every filename currently in the index, in the order added."""
    try:
        _load()
    except Exception:
        return []

    seen: list[str] = []
    for item in _meta:
        if item["source"] not in seen:
            seen.append(item["source"])
    return seen


def reset() -> None:
    """Wipe everything — used by the 'Clear all files' button."""
    global _index, _meta
    with _lock:
        _meta = []
        try:
            _index = _faiss().IndexFlatIP(dimension())
        except ImportError:
            _index = None
        for path in (INDEX_PATH, META_PATH):
            path.unlink(missing_ok=True)


# ==========================================================================
# 4. Retrieval
# ==========================================================================
def search(question: str, k: int = 4) -> list[Chunk]:
    """The chunks most relevant to a question, weak matches dropped."""
    with _lock:
        _load()
        if _index.ntotal == 0:
            return []

        vector = embed(question, is_query=True)
        scores, ids = _index.search(vector, min(k, _index.ntotal))

    hits: list[Chunk] = []
    for score, idx in zip(scores[0], ids[0], strict=True):
        if idx == -1 or float(score) < MIN_SCORE:
            continue
        item = _meta[int(idx)]
        hits.append(Chunk(text=item["text"], source=item["source"], score=float(score)))
    return hits


def as_context(hits: list[Chunk]) -> str:
    """Format hits into a block that can be pasted into a prompt."""
    return "\n\n".join(f"--- from {h['source']} ---\n{h['text']}" for h in hits)
