"""Embeds every chunk in data/processed/chunks.json and writes the
vectors to data/processed/embeddings.npz, so a fresh deploy (or a fresh
clone) can load precomputed vectors at startup instead of running
bge-base-en-v1.5 over the whole corpus -- the difference between a
~90s cold start on Streamlit Community Cloud's free-tier CPU and a
sub-20s one.

Run after any re-chunk (chunks.json content or ordering changed):

    python -m scripts.precompute_embeddings

DenseRetriever._build_index() checks the resulting file's chunk IDs
against the live corpus on every startup and refuses to use it if
they've drifted -- so forgetting to re-run this after a re-chunk fails
loudly instead of silently mismatching vectors to text.
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import CHUNKS_PATH, EMBEDDING_MODEL_NAME, EMBEDDINGS_PATH


def main() -> None:
    raw_chunks = json.loads(CHUNKS_PATH.read_text())
    chunk_ids = [c["chunk_id"] for c in raw_chunks]
    texts = [c["text"] for c in raw_chunks]

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # No query-instruction prefix here -- bge's asymmetric convention
    # (see dense.py's _QUERY_INSTRUCTION) only applies that prefix to
    # queries, not indexed passages. This has to match dense.py's own
    # indexing call exactly, or query-time similarity scores would be
    # computed against vectors built a different way than queries are.
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    np.savez(
        EMBEDDINGS_PATH,
        chunk_ids=np.array(chunk_ids),
        embeddings=np.array(embeddings, dtype=np.float32),
    )
    print(f"Wrote {len(chunk_ids)} embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
