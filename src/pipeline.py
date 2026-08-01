"""Orchestrator: ties retrieval and generation together behind one call.

PipelineConfig carries every ablation flag this project will ever need,
defined now even though most do nothing yet (per ARCHITECTURE.md) — the
ablation study depends on every config being one flag flip, not a code
change written under deadline in a later phase.
"""

from dataclasses import dataclass

from src.generate import generate_answer
from src.retrieval.dense import DenseRetriever
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

    def answer(self, query: str, tax_year: int | None = None) -> Answer:
        # Once fusion/reranking exist (Phase 4/5), this branch retrieves
        # k_retrieve candidates for them to narrow down. Until then,
        # retrieving more than k_final and discarding the rest would be
        # pure waste, so the simple path retrieves k_final directly.
        needs_candidate_pool = self.config.use_rerank or self.config.use_sparse
        k = self.config.k_retrieve if needs_candidate_pool else self.config.k_final

        results = self._dense.retrieve(query, k=k, tax_year=tax_year)
        results = results[: self.config.k_final]

        return generate_answer(query, results)
