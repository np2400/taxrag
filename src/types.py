"""The three frozen data contracts every pipeline stage speaks.

Defined once in Phase 1 and then frozen (both figuratively — ARCHITECTURE.md
calls them fixed — and literally, via frozen=True). Later phases add new
retrievers and pipeline stages against this same vocabulary without changing
it. If a later phase seems to need a different shape, that's a signal the
contract was wrong, not a reason to quietly change it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    citation: str  # "IRC §280A(c)(1)"
    source_type: str  # statute | regulation | instruction | publication
    authority_weight: float
    tax_year_start: int | None
    tax_year_end: int | None
    url: str


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    retriever: str  # which retriever produced it — needed for ablation


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[str]
    refused: bool
    refusal_reason: str | None
    trace: dict  # per-stage timing + token cost
