# TaxRAG — Code Architecture

Companion to `SPEC.md`. That file says *what* to build and why; this one says *where it goes*.

**Design principle:** data contracts are fixed in Phase 1 so later phases slot in without refactoring. A new retriever added on Day 4 must satisfy the same interface as the one written on Day 1.

---

## Repo layout

```
taxrag/
├── CLAUDE.md               # agent operating rules (auto-loaded)
├── SPEC.md                 # what & why
├── ARCHITECTURE.md         # this file — where & how
├── SESSIONS.md             # daily prompts
├── README.md               # Phase 7 ONLY
├── requirements.txt
├── app.py                  # Streamlit entry point
│
├── config/
│   ├── settings.py         # paths, model names, k values
│   └── tax_rates.yaml      # rates keyed by tax year (Phase 6)
│
├── data/
│   ├── raw/                # downloaded IRC / Regs / Pubs (gitignored)
│   └── processed/          # chunked + metadata JSON (committed)
│
├── src/
│   ├── types.py            # Chunk, RetrievalResult, Answer  ← Phase 1, then frozen
│   ├── ingest/
│   │   ├── loader.py       # fetch + normalize source docs
│   │   ├── chunker.py      # section-aware splitting, header injection
│   │   └── metadata.py     # authority weights, tax-year tagging
│   ├── retrieval/
│   │   ├── base.py         # Retriever protocol  ← the key contract
│   │   ├── dense.py        # ChromaDB + embeddings      (Phase 1)
│   │   ├── sparse.py       # BM25                       (Phase 4)
│   │   ├── fusion.py       # Reciprocal Rank Fusion     (Phase 4)
│   │   ├── rerank.py       # cross-encoder              (Phase 5)
│   │   └── authority.py    # statute-over-pub reorder   (Phase 5)
│   ├── agents/
│   │   ├── tools.py        # SE tax, home office        (Phase 6)
│   │   └── verifier.py     # citation check + retry     (Phase 6)
│   ├── generate.py         # synthesis with inline citations
│   └── pipeline.py         # orchestrator — grows each phase
│
├── evals/
│   ├── golden_set.json     # ← HUMAN-WRITTEN. Phase 2. See CLAUDE.md rule 6.
│   ├── metrics.py          # Recall@5, MRR, groundedness, citation acc.
│   ├── run_eval.py         # CLI: python -m evals.run_eval --config hybrid
│   └── results/            # one timestamped JSON per run — never overwrite
│
└── tests/
    └── test_tools.py       # unit tests for tax math (Phase 6)
```

---

## The three contracts (Phase 1, then frozen)

Everything else is built against these. Define them first; don't change them later.

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

**Why this matters:** `dense.py`, `sparse.py`, `fusion.py`, and `rerank.py` all take and return `list[RetrievalResult]`. That's what makes the ablation study trivial — you swap components in the pipeline without touching anything else.

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
                    ┌──────────────────────────────────┐
                    │  dense.retrieve()  ─┐            │
                    │  sparse.retrieve() ─┴─► fusion   │  Phase 1 → 4
                    │                          │       │
                    │                          ▼       │
                    │                   rerank.apply() │  Phase 5
                    │                          │       │
                    │                          ▼       │
                    │                authority.reorder()│  Phase 5
                    └──────────────┬───────────────────┘
                                   ▼
                       tools.py (if computational)        Phase 6
                                   ▼
                            generate.py → draft
                                   ▼
                         verifier.check(draft)            Phase 6
                            │           │
                          pass        fail → retry once → refuse
                            ▼
                          Answer
```

---

## Ablation config — the piece that makes Phase 7 easy

`pipeline.py` takes a config so every ablation row is one flag change, not a code change:

```python
@dataclass
class PipelineConfig:
    use_dense: bool = True
    use_sparse: bool = False      # Phase 4
    use_rerank: bool = False      # Phase 5
    use_authority: bool = False   # Phase 5
    use_verifier: bool = False    # Phase 6
    k_retrieve: int = 20
    k_final: int = 5
```

```bash
python -m evals.run_eval --config dense_only
python -m evals.run_eval --config hybrid
python -m evals.run_eval --config hybrid_rerank
```

Each run writes a timestamped JSON to `evals/results/`. The ablation table in the README is assembled from those files — **never hand-typed.**

> Build this config object in Phase 1 even though most flags are false. Retrofitting it in Phase 7 means rewriting the pipeline under deadline.

---

## Phase ownership

Each phase touches only its own files. If a session wants to edit a file outside its row, stop and ask.

| Phase | Creates | Modifies |
|---|---|---|
| 1 | `types.py`, `base.py`, `ingest/*`, `dense.py`, `generate.py`, `pipeline.py`, `app.py` | — |
| 2 | `evals/golden_set.json` | — |
| 3 | `metrics.py`, `run_eval.py` | — |
| 4 | `sparse.py`, `fusion.py` | `pipeline.py` (flags only) |
| 5 | `rerank.py`, `authority.py` | `pipeline.py` (flags only) |
| 6 | `tools.py`, `verifier.py`, `tax_rates.yaml`, `test_tools.py` | `pipeline.py` (flags only) |
| 7 | `README.md` | `app.py` |
| 8 | — | — |

**After Phase 1, `pipeline.py` should only ever gain flag branches.** If a later phase needs to restructure it, the Phase 1 contracts were wrong — say so rather than quietly refactoring.

---

## Explicitly not in this architecture

No `api/`, no `auth/`, no `db/`, no `docker/`, no `.github/workflows/`, no `langchain` anywhere.

If a directory appears that isn't in the tree above, something went out of scope.
