"""Dense retrieval: bge-base-en-v1.5 embeddings + ChromaDB.

A class, not a function or plain module — model loading and the vector DB
connection are both expensive one-time setup costs that should happen once
per process, not once per query.
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import CHROMA_PERSIST_DIR, CHUNKS_PATH, EMBEDDING_MODEL_NAME
from src.types import Chunk, RetrievalResult

# bge models are trained with an asymmetric convention: queries get this
# instruction prefix, indexed passages do not. Skipping it doesn't error —
# it just silently retrieves worse, since it no longer matches how the
# model was fine-tuned specifically for retrieval.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_COLLECTION_NAME = "taxrag_chunks"

# Chroma metadata values must be str/int/float/bool — None isn't allowed.
# -1 is a sentinel for "no year restriction," converted back to None on
# the way out.
_NO_YEAR = -1


class DenseRetriever:
    def __init__(self) -> None:
        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        if self._collection.count() == 0:
            self._build_index()

    def _build_index(self) -> None:
        """First-run bootstrap. data/chroma/ is gitignored -- it's fully
        rebuildable from the committed data/processed/chunks.json, so
        there's nothing worth version-controlling a binary index for.
        That means a fresh checkout (a Streamlit Cloud deploy, or anyone
        cloning the repo) starts with an empty collection; this embeds
        the committed corpus once, here, rather than requiring a manual
        build step no one would remember to run."""
        raw_chunks = json.loads(CHUNKS_PATH.read_text())
        self.index_chunks([Chunk(**c) for c in raw_chunks])

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """One-time build step: embed every chunk and store it, keyed by
        chunk_id, with its metadata alongside."""
        texts = [c.text for c in chunks]
        embeddings = self._model.encode(texts, normalize_embeddings=True).tolist()
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "citation": c.citation,
                    "source_type": c.source_type,
                    "authority_weight": c.authority_weight,
                    "tax_year_start": c.tax_year_start
                    if c.tax_year_start is not None
                    else _NO_YEAR,
                    "tax_year_end": c.tax_year_end
                    if c.tax_year_end is not None
                    else _NO_YEAR,
                    "url": c.url,
                }
                for c in chunks
            ],
        )

    def retrieve(
        self, query: str, k: int, tax_year: int | None = None
    ) -> list[RetrievalResult]:
        query_embedding = self._model.encode(
            _QUERY_INSTRUCTION + query, normalize_embeddings=True
        ).tolist()
        results = self._collection.query(query_embeddings=[query_embedding], n_results=k)

        out: list[RetrievalResult] = []
        for rank, (chunk_id, text, meta, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1,
        ):
            chunk = Chunk(
                chunk_id=chunk_id,
                text=text,
                citation=meta["citation"],
                source_type=meta["source_type"],
                authority_weight=meta["authority_weight"],
                tax_year_start=None
                if meta["tax_year_start"] == _NO_YEAR
                else meta["tax_year_start"],
                tax_year_end=None
                if meta["tax_year_end"] == _NO_YEAR
                else meta["tax_year_end"],
                url=meta["url"],
            )
            score = 1 - distance  # cosine distance -> similarity, higher is better
            out.append(RetrievalResult(chunk=chunk, score=score, rank=rank, retriever="dense"))
        return out
