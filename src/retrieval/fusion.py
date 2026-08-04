"""Reciprocal Rank Fusion: combine dense and sparse ranked lists using
only rank position, never the raw scores.

Why not blend the raw scores instead: BM25 scores are unbounded
term-frequency values (magnitude depends on document length and corpus
statistics) while cosine similarity is bounded to [-1, 1] -- averaging
or weighting them directly requires an arbitrary normalization constant
that would need re-tuning if the corpus or query distribution shifts.
RRF sidesteps that: it only asks "what rank did this chunk get from each
retriever," so nothing needs tuning, and a retriever that completely
whiffs on a query (returns something irrelevant at rank 1) barely
distorts the fused result the way a bad raw score could in a weighted
average.
"""

from src.types import Chunk, RetrievalResult

_K = 60  # standard RRF damping constant -- large enough that the gap
# between rank 1 and rank 2 doesn't dominate the fused score the way it
# would with a small constant; not tuned on this corpus, by design (the
# whole point of RRF is not needing to tune anything).


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    k: int = _K,
) -> list[RetrievalResult]:
    """Fuse two ranked lists into one, ranked by summed 1/(k + rank)
    across whichever list(s) each chunk appeared in."""
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}
    sources_by_id: dict[str, set[str]] = {}

    for results in (dense_results, sparse_results):
        for r in results:
            chunk_id = r.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + r.rank)
            chunks_by_id[chunk_id] = r.chunk
            sources_by_id.setdefault(chunk_id, set()).add(r.retriever)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    return [
        RetrievalResult(
            chunk=chunks_by_id[chunk_id],
            score=scores[chunk_id],
            rank=rank,
            # Which retriever(s) actually surfaced this chunk -- e.g.
            # "dense+sparse" for a chunk both agreed on, or just "sparse"
            # for one only BM25 found. Useful later for seeing which
            # retriever "saved" a given exact-token question.
            retriever="+".join(sorted(sources_by_id[chunk_id])),
        )
        for rank, chunk_id in enumerate(ranked_ids, start=1)
    ]
