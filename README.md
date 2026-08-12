# TaxRAG — Citation-Grounded Retrieval for Small-Business Federal Tax

**Live app:** https://taxrag.streamlit.app/

TaxRAG is a citation-grounded research assistant for Schedule C /
self-employed federal tax questions — the home office deduction,
vehicle/mileage deduction, and self-employment tax. It surfaces primary
tax authority with exact citations rather than giving advice, and it is
designed to refuse out-of-scope or underspecified questions rather than
guess.

## Why naive RAG fails on tax law

Chunk documents, embed them, retrieve top-k, generate — the default RAG
recipe breaks on tax questions in specific, measurable ways:

- **Hierarchical authority.** IRC statute, Treasury regulations, and IRS
  publications do not carry equal weight, but semantic similarity has no
  notion of that hierarchy — a plain-English publication can outrank the
  statute it is summarizing simply because its prose reads closer to how
  a taxpayer phrases a question.
- **Exact tokens carry legal meaning.** `§179` and `§199A` are
  near-identical vectors to an embedding model and legally unrelated.
- **Deterministic math.** Self-employment tax and the home-office
  deduction are computed, not looked up — a wrong LLM-generated number
  is worse than a refusal.

Concrete, traced example: asking "What does IRC §280A(c)(1) require for
the home-office exception to the general disallowance rule?" — dense
retrieval's top-5 never included the statute at all, only IRS
Publication chunks describing the same rule in plain English. That is
the hierarchical-authority failure above, caught in real eval data, not
hypothesized.

## Architecture

```
Query
  |
  |--> Retrieval  (metadata filter: tax_year applies to both retrievers)
  |      |-- BM25 (sparse, rank_bm25)   --+
  |      |                                |-- Reciprocal Rank Fusion --+
  |      |-- Dense (bge-base-en-v1.5)   --+                            |
  |      |                                                            |-- top 5
  |      +-- Exact citation lookup ---------------------------------- +
  |            (fires only when the query names a §/IRC provision;
  |             its chunks are placed first, remaining slots filled
  |             from the fused list)
  |
  |--> Synthesis (Groq, citation-grounded prompt)
  |
  +--> Answer (text + citations, or a designed refusal)
```

Chunking is section-aware (splits on structural boundaries like
`§280A(c)(1)`, never mid-provision), with the hierarchical path injected
into each chunk's header (`IRC §280A > (c) Exceptions > (1) Business
use`). Every chunk carries a citation, source type, and authority weight.

## Results

### Retrieval (Recall@5, MRR)

No LLM involved — pure comparison of required citation vs. retrieved
chunk IDs — so these runs are complete. Scored on the 49 of 55 golden-set
questions that have a required citation; the 6 `out_of_scope` questions
have no correct chunk by construction and are excluded from retrieval
scoring (they are exercised by the refusal metric instead). Numbers are
read directly from `evals/results/`, never hand-typed.

| Config | Recall@5 | MRR |
|---|---|---|
| Dense only | 0.689 | 0.767 |
| BM25 only | 0.624 | 0.643 |
| Hybrid (RRF) | 0.675 | 0.813 |
| **Hybrid + citation lookup** (current default) | **0.706** | **0.847** |

By category, all four configs:

| Category | n | Dense R@5 | BM25 R@5 | Hybrid R@5 | Hybrid+CL R@5 | Hybrid+CL MRR |
|---|---|---|---|---|---|---|
| exact_token | 12 | 0.722 | 0.681 | 0.722 | **0.847** | 0.861 |
| factual_lookup | 15 | 0.700 | 0.500 | 0.633 | 0.633 | 0.867 |
| computational | 8 | 0.750 | 0.750 | 0.750 | 0.750 | 0.938 |
| temporal | 5 | 0.800 | 0.800 | 0.800 | 0.800 | 0.667 |
| multi_hop | 5 | 0.517 | 0.583 | 0.583 | 0.583 | 1.000 |
| adversarial | 4 | 0.500 | 0.500 | 0.500 | 0.500 | 0.583 |

Honest read, in three parts:

**Hybrid alone does not beat dense on recall.** On aggregate Recall@5,
hybrid (0.675) is below dense-only (0.689), and on exact-token questions
— the category this whole hybrid argument is built on — it only ties
dense rather than beating it. What hybrid consistently delivers is better
*ranking*: when the correct chunk is already retrieved, fusing in BM25
tends to rank it higher (computational MRR 0.708 → 0.938, factual 0.811 →
0.867). That is a real, more modest result than "hybrid rescues
exact-token misses," reported as measured rather than reframed.

**Citation lookup is what actually moves recall, and it moves exactly one
category.** When a query names a specific IRC provision (`§280A(c)(1)`,
`IRC 1401(b)(2)`), the pipeline resolves it directly against each chunk's
citation field instead of depending on ranking to surface it — see Design
decisions below. This takes exact_token Recall@5 from 0.722 to 0.847 (MRR
0.722 → 0.861) and is the entire source of the aggregate gain from 0.675
to 0.706. Every other category is byte-identical with the flag on or off,
because the lookup only fires when a query actually parses out a §/IRC
citation.

**The weakest categories are `adversarial` (0.500) and `multi_hop`
(0.583), and no config moves either.** Both are structural rather than
tunable: adversarial questions embed a false premise that has to be
corrected before retrieval is even the right operation, and multi-hop
questions need two retrieval passes across different authorities. Neither
is addressable by better ranking of a single query.

### Generation (citation accuracy, groundedness, hallucination rate) — 15-question subset

| Config (n=15, q001-q015) | Citation acc. | Groundedness | Hallucinated-cite rate | Refused |
|---|---|---|---|---|
| Dense only | 0.067 | 0.867 | 0.067 | 0/15 |
| Hybrid (RRF) | 0.133 | 0.786 | 0.200 | 1/15 |

Citation accuracy and groundedness each need an LLM call per question
(generate, then a separate LLM-as-judge call to grade it), and a full
55-question run costs roughly 150k-220k tokens across both calls — enough
to hit Groq's free-tier daily token cap before every config finished.
This table uses only the 15 questions both `dense_only` and `hybrid` have
complete data for, so it is apples-to-apples on a smaller n rather than a
comparison across mismatched partial samples. BM25-only is excluded here
by design, not budget — nobody would ship BM25-only generation; its
retrieval contribution is already isolated above, without needing an LLM
at all.

**These numbers predate the citation-display fix (`bc71b4d`) and the
exact-citation-lookup feature, and should be treated as a lower bound
pending a re-run.** They are kept here rather than deleted because they
are what was actually measured at the time.

Why citation accuracy looks low: traced, not assumed. For the §280A(c)(1)
question above, the model correctly cited the Publication chunks it was
actually given — it just was not given the statute the golden set
specifically requires as ground truth. Recall that finding a relevant
source is not the same as generation citing the specific source a strict
answer key demands; some of this gap is a retrieval miss, some may be the
golden set being stricter than necessary.

## Design decisions

**Hybrid retrieval, fused with Reciprocal Rank Fusion, not weighted score
blending.** BM25 and dense embeddings fail on opposite query types —
dense handles paraphrase, BM25 handles exact tokens. RRF combines their
ranked lists using only rank position (sum of 1/(k + rank)), never raw
scores: BM25's unbounded term-frequency scores and cosine similarity's
[-1, 1] range are not comparable without an arbitrary tuned normalization
constant. RRF needs no tuning and degrades gracefully when one retriever
misses entirely.

**A real BM25 bug, found and fixed.** The first BM25 implementation had
no stopword filtering, on the assumption that IDF weighting alone would
discount common words. That broke here specifically: this corpus is
formal statutory prose that almost never phrases things as questions, so
words like "does" or "what" are actually rare in this corpus — BM25 gave
"does" an IDF of 3.17, nearly on par with the genuinely distinctive
"280a" at 3.40, which pushed the correct statute for the §280A(c)(1)
query to rank 50 of 1,531. Stopword filtering fixed it, verified on that
exact query before trusting the aggregate numbers.

**Exact citation lookup, a fragment bug found and fixed.** The traced
§280A(c)(1) failure above wasn't only about ranking — even when hybrid
retrieval did surface the statute, it returned exactly one of its three
alternative subparagraphs, (A), (B), or (C), and generation presented
that single prong as if it were the complete rule. A query that names a
specific provision doesn't need to be ranked into place:
`src/retrieval/citation.py` parses `§`/`IRC`-style citations out of the
query and resolves them directly against each chunk's own citation field,
returning the cited provision plus every sibling beneath it rather than
whichever one ranking happened to surface. "Beneath it" isn't a uniform
one-level-down rule — this corpus sometimes skips a level entirely
(`§1401(b)(2)` has no chunk of its own, and its two branches,
`§1401(b)(2)(B)` and `§1401(b)(2)(A)(i)-(iii)`, sit at different depths
below it), so the lookup walks to the closest existing chunk along every
branch instead of assuming every branch is the same depth.

**LLM-as-judge over RAGAS.** Groundedness and premise-correction are
graded by a custom prompt against the same Groq client already in use,
rather than adding RAGAS as a dependency — a hand-written rubric can be
quoted and defended directly; a scoring library's internals cannot be as
easily.

**Prefix-matched citations, not exact string match.** The golden set
cites at section level (`IRC §1401`); ingested chunks are pinpoint-cited
to subsections (`IRC §1401(c)(3)`). Exact-string matching would silently
undercount recall on every such question, so citations match on a
normalized token with prefix comparison in either direction — coarser for
Publications (collapses to document-level only) than for statutes.

## Known limitations

- Hybrid retrieval's benefit is real but modest — it improves ranking
  confidence for chunks already found, it does not rescue missed
  exact-token chunks in this eval. The recall gain comes from citation
  lookup, not from fusion.
- `adversarial` (0.500) and `multi_hop` (0.583) are the weakest
  categories and no config moves either; both need something other than
  better single-query ranking.
- Generation-metric coverage is partial: 15 of 55 questions for
  dense_only vs. hybrid, none for BM25-only, and all of it predates the
  citation-display fix. A full re-run is the top outstanding item.
- LLM-as-judge validity is unverified against human judgment — no
  hand-labeled subset has been checked against the judge's scores yet.
- The 55-question golden set was drafted by an LLM and reviewed in full by a CPA before use.
- `authority_weight` is assigned per source at ingestion (statute 1.0,
  publications 0.4) and carried on every chunk, but nothing currently
  ranks by it — the metadata exists ahead of the reordering step that
  would use it.
- BM25 tokenization is simple (lowercase, word-split, stopword removal,
  no stemming) — sufficient at this corpus size, not benchmarked further.
- Retrieval assumes the answer lives in one authority; multi-hop
  questions spanning statute, regulation, and publication are a measured,
  unsolved weak category.
- `data/raw/` is not tracked in git. The repo ships precomputed
  `chunks.json` and `embeddings.npz`, so the app and the eval harness run
  from a fresh clone, but re-running ingestion from source requires
  re-downloading the 12 source documents.

## What I'd add next

- Finish full-55 generation metrics for all three configs against the
  post-fix pipeline, and hand-label a subset to report actual judge/human
  agreement.
- Cross-encoder reranking and authority-weighted reordering — for queries
  that don't name a specific provision explicitly, the statute still
  doesn't reliably outrank the Publication describing it; exact citation
  lookup only helps when the query itself parses out a §/IRC citation.
- Deterministic calculation tools (self-employment tax, home-office
  deduction) so arithmetic never comes from the LLM, plus a citation
  verification step (entailment check, retry once, then refuse).
- Further BM25 tuning (larger RRF constant, different tokenizer) before
  concluding hybrid's ceiling here is "better MRR, not better recall."
- pgvector or Qdrant in place of local ChromaDB for concurrent users.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=..." > .env
streamlit run app.py
```

Run the eval harness:

```bash
# Recall@5 / MRR only — no LLM calls, no API cost, finishes in seconds
python -m evals.run_eval --config hybrid --retrieval-only

# Full run: retrieval + generation + LLM-as-judge.
# ~150k-220k tokens; will exhaust a Groq free-tier daily cap.
python -m evals.run_eval --config hybrid
```

Configs: `dense_only`, `bm25_only`, `hybrid`. Each run writes a
timestamped JSON to `evals/results/`.

`.github/workflows/keepalive.yml` pings the live app every 6 hours with a
headless browser so Streamlit Cloud's 12-hour sleep timer never fires —
see `scripts/keepalive.py`.
