"""Central configuration for TaxRAG: paths, model names, and retrieval defaults.

Every other module imports from here instead of hardcoding paths or magic
numbers, so a change (e.g. tuning k during the ablation study) happens in
exactly one place.
"""

import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHROMA_PERSIST_DIR = BASE_DIR / "data" / "chroma"

CHUNKS_PATH = DATA_PROCESSED_DIR / "chunks.json"

# --- Embedding model ---
# Local, free, runs via sentence-transformers — no API key required.
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# --- Retrieval defaults ---
K_RETRIEVE = 20  # candidates pulled before fusion/reranking (Phase 4/5)
K_FINAL = 5  # final chunks passed to generation


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader — avoids adding python-dotenv for something this
    small; we control the file's exact format ourselves, so we don't need
    a library that also handles quoting, comments, or multiline values."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(BASE_DIR / ".env")

# --- Generation model ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
REFUSAL_MARKER = "INSUFFICIENT_INFORMATION"
