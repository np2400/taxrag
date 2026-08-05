# TaxRAG — Code Architecture

Companion to `DESIGN.md`, which covers the reasoning behind these choices. This file says where things live and how the pieces fit together.

**Design principle:** data contracts are fixed early so later components slot in without refactoring. Any retriever added later satisfies the same interface as the one written first.

---

## Repo layout

```
taxrag/
├── CLAUDE.md               # agent operating rules
├── ARCHITECTURE.md         # this file — where & how
├── DESIGN.md               # why — rationale and known limitations
├── README.md
├── requirements.txt
├── app.py                  # Streamlit entry point
│
├── config/
│   ├── settings.py         # paths, model names, k values
│   └── tax_rates.yaml      # rates keyed by tax year — planned, not yet implemented
│
├── data/
│   ├── raw/                # downloaded IRC / Regs / Pubs (gitignored)
│   └── processed/          # chunked + metadata JSON (committed)
│
├── src/
│   ├── types.py            # Chunk, RetrievalResult, Answer — frozen contracts
│   ├── ingest/
│   │   ├── loader.py       # fetch + normalize source docs
│   │   ├── chunker.py      # section-aware splitting, header injection
│   │   └── metadata.py     # authority weights, tax-year tagging
│   ├── retrieval/
│   │   ├── base.py         # Retriever protocol — the key contract
│   │   ├── dense.py        # ChromaDB + embeddings
│   │   ├── sparse.py       # BM25
│   │   ├── fusion.py       # Reciprocal Rank Fusion
│   │   ├── citation.py     # exact citation lookup — parses §/IRC citations out of a query, resolves them by chunk citation field
│   │   ├── rerank.py       # cross-encoder — planned, not yet implemented
│   │   └── authority.py    # statute-over-publication reorder — planned, not yet implemented
│   ├── agents/
│   │   ├── tools.py        # SE tax, home office calculators — planned, not yet implemented
│   │   └── verifier.py     # citation check + retry — planned, not yet implemented
│   ├── generate.py         # synthesis with inline citations
│   └── pipeline.py         # orchestrator
│
├── evals/
│   ├── golden_set.json     # hand-written, 55 questions
│   ├── metrics.py          # Recall@5, MRR, groundedness, citation acc.
│   ├── run_eval.py         # CLI: python -m evals.run_eval --config hybrid
│   └── results/            # one timestamped JSON per run — never overwritten
│
└── tests/                  # planned, not yet implemented
    └── test_tools.py       # unit tests for tax math
```

---

## The three contracts

Everything else is built against these.

```python
# src/types.py

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    citation: str              # "IRC §280A(c)(1)"
    source_type: str           # statute | regulation | instruction | publication
    authority_weight: float
    tax_year_start: int | None
    tax_year_end: int | None
    url: str

@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    retriever: str             # which retriever produced it — needed for ablation

@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[str]
    refused: bool
    refusal_reason: str | None
    trace: dict                # per-stage timing + token cost
```

```python
# src/retrieval/base.py

class Retriever(Protocol):
    def retrieve(self, query: str, k: int,
                 tax_year: int | None = None) -> list[RetrievalResult]: ...
```

**Why this matters:** `dense.py`, `sparse.py`, `fusion.py`, and the planned `rerank.py` all take and return `list[RetrievalResult]`. That's what makes the ablation study possible — components swap in and out of the pipeline without touching anything else.

---

## Data flow

```
                    ┌─────────────────────────────────────┐
  raw docs ──────►  │  loader → chunker → metadata        │  (offline, run once)
                    └──────────────┬──────────────────────┘
                                   ▼
                            data/processed/*.json
                                   │
                                   ▼
                          ChromaDB + BM25 index
                                   │
  query ───────────────────────────┤
                                   ▼
                    ┌───────────────────────────────────────┐
                    │  dense.retrieve()  ─┐                 │
                    │  sparse.retrieve() ─┴─► fusion         │  implemented
                    │                          │             │
                    │                          ▼             │
                    │                   rerank.apply()       │  planned
                    │                          │             │
                    │                          ▼             │
                    │                authority.reorder()     │  planned
                    └──────────────┬──────────────────────────┘
                                   ▼
                       tools.py (if computational)             planned
                                   ▼
                            generate.py → draft                implemented
                                   ▼
                         verifier.check(draft)                 planned
                            │           │
                          pass        fail → retry once → refuse
                            ▼
                          Answer
```

Refusal itself is implemented independently of the verifier above — `generate.py` can already return `Answer(refused=True, ...)` for out-of-scope or underspecified questions. The planned verifier adds a second, citation-specific path to refusal (unsupported claim → retry → refuse).

---

## Ablation config

`pipeline.py` takes a config so every ablation row is one flag change, not a code change:

```python
@dataclass
class PipelineConfig:
    use_dense: bool = True
    use_sparse: bool = False           # implemented
    use_citation_lookup: bool = True   # implemented
    use_rerank: bool = False           # planned
    use_authority: bool = False        # planned
    use_verifier: bool = False         # planned
    k_retrieve: int = 20
    k_final: int = 5
```

```bash
python -m evals.run_eval --config dense_only
python -m evals.run_eval --config bm25_only
python -m evals.run_eval --config hybrid
```

Each run writes a timestamped JSON to `evals/results/`. The ablation table in the README is assembled from those files, never hand-typed. Flags for components that don't exist yet (`use_rerank`, `use_authority`, `use_verifier`) are defined now so adding the corresponding retriever or agent later means flipping a flag rather than restructuring the pipeline.

---

## Explicitly not in this architecture

No `api/`, no `auth/`, no `db/`, no `docker/`, no `.github/workflows/`, no `langchain` anywhere.

If a directory appears that isn't in the tree above, something went out of scope.
