"""Orchestrator: ties retrieval and generation together behind one call.

PipelineConfig carries every ablation flag this project will ever need,
defined now even though most do nothing yet (per ARCHITECTURE.md) — the
ablation study depends on every config being one flag flip, not a code
change written under deadline in a later phase.
"""

from dataclasses import dataclass

from src.generate import generate_answer
from src.retrieval.citation import exact_citation_lookup, parse_citation
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.sparse import SparseRetriever
from src.types import Answer, Chunk, RetrievalResult


@dataclass
class PipelineConfig:
    use_dense: bool = True
    use_sparse: bool = False  # Phase 4
    use_citation_lookup: bool = True
    use_rerank: bool = False  # Phase 5
    use_authority: bool = False  # Phase 5
    use_verifier: bool = False  # Phase 6
    k_retrieve: int = 20
    k_final: int = 5


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._dense = DenseRetriever() if self.config.use_dense else None
        self._sparse = SparseRetriever() if self.config.use_sparse else None

    def answer(self, query: str, tax_year: int | None = None) -> Answer:
        # A bigger candidate pool is only worth pulling when something
        # downstream will narrow it back down -- fusion (both retrievers
        # on) or reranking (Phase 5). A single retriever running alone
        # (dense-only or the Phase 4 "BM25 only" ablation row) still
        # retrieves k_final directly, same as before Phase 4.
        needs_candidate_pool = self.config.use_rerank or (
            self.config.use_dense and self.config.use_sparse
        )
        k = self.config.k_retrieve if needs_candidate_pool else self.config.k_final

        if self.config.use_dense and self.config.use_sparse:
            dense_results = self._dense.retrieve(query, k=k, tax_year=tax_year)
            sparse_results = self._sparse.retrieve(query, k=k, tax_year=tax_year)
            results = reciprocal_rank_fusion(dense_results, sparse_results)
        elif self.config.use_sparse:
            results = self._sparse.retrieve(query, k=k, tax_year=tax_year)
        else:
            results = self._dense.retrieve(query, k=k, tax_year=tax_year)

        if self.config.use_citation_lookup:
            parsed = parse_citation(query)
            if parsed is not None:
                citation_chunks = exact_citation_lookup(parsed)
                if citation_chunks:
                    results = prioritize_citation_chunks(citation_chunks, results)

        results = results[: self.config.k_final]

        return generate_answer(query, results)


def prioritize_citation_chunks(
    citation_chunks: list[Chunk], results: list[RetrievalResult]
) -> list[RetrievalResult]:
    """Citation-lookup chunks go first, then whatever slots are left over
    fill in from the retriever(s) that already ran -- so a query naming a
    provision doesn't lose the rest of hybrid retrieval's context, it
    just no longer depends on ranking to surface the provision itself.
    Any chunk citation lookup already found is dropped from the retrieved
    list rather than kept twice. Non-citation results keep their original
    score and retriever tag (e.g. "dense+sparse") -- only their rank
    changes, since citation.py's own confidence isn't comparable to
    either retriever's score.

    Not private (no leading underscore): evals/run_eval.py's
    _retrieve_for_config() applies the same citation-lookup step, in the
    same order, so retrieval-only scoring and the full pipeline never
    drift apart -- it imports this rather than re-deriving the merge."""
    citation_ids = {c.chunk_id for c in citation_chunks}
    citation_results = [
        RetrievalResult(chunk=chunk, score=1.0, rank=0, retriever="citation")
        for chunk in citation_chunks
    ]
    remaining_results = [r for r in results if r.chunk.chunk_id not in citation_ids]

    return [
        RetrievalResult(chunk=r.chunk, score=r.score, rank=rank, retriever=r.retriever)
        for rank, r in enumerate(citation_results + remaining_results, start=1)
    ]
