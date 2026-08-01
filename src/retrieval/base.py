"""The Retriever contract every retrieval strategy implements.

A Protocol, not an ABC: structural typing means dense.py, sparse.py (Phase 4),
and any fused/hybrid retriever built later satisfy this type simply by
defining a matching retrieve() method — no shared parent class, no coupling
between implementations that otherwise share nothing.
"""

from typing import Protocol

from src.types import RetrievalResult


class Retriever(Protocol):
    def retrieve(
        self, query: str, k: int, tax_year: int | None = None
    ) -> list[RetrievalResult]: ...
