"""Orchestrator: ties retrieval and generation together behind one call.

PipelineConfig carries every ablation flag this project will ever need,
defined now even though most do nothing yet (per ARCHITECTURE.md) — the
ablation study depends on every config being one flag flip, not a code
change written under deadline in a later phase.
"""

from dataclasses import dataclass

from src.generate import generate_answer
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.sparse import SparseRetriever
from src.types import Answer


@dataclass
class PipelineConfig:
    use_dense: bool = True
    use_sparse: bool = False  # Phase 4
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

        results = results[: self.config.k_final]

        return generate_answer(query, results)
