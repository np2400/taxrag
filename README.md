# TaxRAG — Agentic Retrieval for Small-Business Federal Tax

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
  |--> Retrieval
  |      |-- BM25 (sparse, rank_bm25)     --+
  |      |-- Dense (bge-base-en-v1.5)       |-- Reciprocal Rank Fusion
  |      +-- (metadata filter: tax_year)  --+
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

### Retrieval (Recall@5, MRR) — full 55-question set

No LLM involved — pure comparison of required citation vs. retrieved
chunk IDs — so this table is complete.

| Config | Recall@5 | MRR |
|---|---|---|
| Dense only | 0.689 | 0.767 |
| BM25 only | 0.624 | 0.643 |
| Hybrid (RRF) | 0.675 | 0.813 |
| Hybrid + citation lookup | 0.706 | 0.847 |

By category (n=12 exact-token, n=8 computational, n=5 multi-hop, n=15
factual lookup):

| Category | Dense R@5 | BM25 R@5 | Hybrid R@5 | Dense MRR | BM25 MRR | Hybrid MRR |
|---|---|---|---|---|---|---|
| exact_token | 0.722 | 0.681 | 0.722 | 0.711 | 0.590 | 0.722 |
| computational | 0.750 | 0.750 | 0.750 | 0.708 | 0.562 | 0.938 |
| multi_hop | 0.517 | 0.583 | 0.583 | 1.000 | 0.500 | 1.000 |
| factual_lookup | 0.700 | 0.500 | 0.633 | 0.811 | 0.783 | 0.867 |

Honest read: hybrid does not beat dense-only on aggregate Recall@5, and
even on exact-token questions — the category this whole hybrid-retrieval
argument is built on — it only ties dense rather than clearly beating
it. What hybrid consistently delivers is better MRR: when the correct
chunk is already retrieved, fusing in BM25 tends to rank it higher
(exact-token 0.711 to 0.722, computational 0.708 to 0.938, factual 0.811
to 0.867). That is a real, more modest result than "hybrid rescues
exact-token misses," reported as measured rather than reframed.

**Citation lookup moves exactly one category, by design.** When a query
names a specific IRC provision (`§280A(c)(1)`, `IRC 1401(b)(2)`), the
pipeline now resolves it directly against each chunk's citation field
instead of depending on ranking to surface it — see Design decisions
below. On the retrieval-only harness this takes exact_token Recall@5
from 0.722 to 0.847 (MRR 0.722 → 0.861); every other category's numbers
are identical with the flag on or off, because citation lookup only
fires when a query actually parses out a §/IRC citation, which in this
golden set only happens on exact_token questions.

### Generation (citation accuracy, groundedness, hallucination rate) — 15-question subset

| Config (n=15, q001-q015) | Citation acc. | Groundedness | Hallucinated-cite rate | Refused |
|---|---|---|---|---|
| Dense only | 0.067 | 0.867 | 0.067 | 0/15 |
| Hybrid (RRF) | 0.133 | 0.786 | 0.200 | 1/15 |

Citation accuracy and groundedness each need an LLM call per question
(generate, then a separate LLM-as-judge call to grade it), and a full
55-question run costs roughly 150k-220k tokens across both calls —
enough to hit Groq's free-tier daily token cap before every config
finished. This table uses only the 15 questions both `dense_only` and
`hybrid` have complete data for, so it is apples-to-apples on a smaller
n rather than a comparison across mismatched partial samples. BM25-only
is excluded here by design, not budget — nobody would ship BM25-only
generation; its retrieval contribution is already isolated above,
without needing an LLM at all.

Why citation accuracy looks low: traced, not assumed. For the
§280A(c)(1) question above, the model correctly cited the Publication
chunks it was actually given — it just was not given the statute the
golden set specifically requires as ground truth. Recall finding a
relevant source is not the same as generation citing the specific source
a strict answer key demands; some of this gap is a retrieval miss, some
may be the golden set being stricter than necessary.

## Design decisions

**Hybrid retrieval, fused with Reciprocal Rank Fusion, not weighted score
blending.** BM25 and dense embeddings fail on opposite query types —
dense handles paraphrase, BM25 handles exact tokens. RRF combines their
ranked lists using only rank position (sum of 1/(k + rank)), never raw
scores: BM25's unbounded term-frequency scores and cosine similarity's
[-1, 1] range are not comparable without an arbitrary tuned
normalization constant. RRF needs no tuning and degrades gracefully when
one retriever misses entirely.

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
specific provision doesn't need to be ranked into place: `src/retrieval/citation.py`
parses `§`/`IRC`-style citations out of the query and resolves them
directly against each chunk's own citation field, returning the cited
provision plus every sibling beneath it rather than whichever one
ranking happened to surface. "Beneath it" isn't a uniform one-level-down
rule — this corpus sometimes skips a level entirely (`§1401(b)(2)` has
no chunk of its own, and its two branches, `§1401(b)(2)(B)` and
`§1401(b)(2)(A)(i)-(iii)`, sit at different depths below it), so the
lookup walks to the closest existing chunk along every branch instead of
assuming every branch is the same depth. Measured on the retrieval-only
harness: exact_token Recall@5 goes from 0.722 to 0.847 (MRR 0.722 →
0.861); every other category is unchanged, since this only fires on
queries that actually parse out a section citation.

**LLM-as-judge over RAGAS.** Groundedness and premise-correction are
graded by a custom prompt against the same Groq client already in use,
rather than adding RAGAS as a dependency — a hand-written rubric can be
quoted and defended directly; a scoring library's internals cannot be as
easily.

**Prefix-matched citations, not exact string match.** The golden set
cites at section level (`IRC §1401`); ingested chunks are pinpoint-cited
to subsections (`IRC §1401(c)(3)`). Exact-string matching would silently
undercount recall on every such question, so citations match on a
normalized token with prefix comparison in either direction — coarser
for Publications (collapses to document-level only) than for statutes.

## Known limitations

- Hybrid retrieval's benefit is real but modest — it improves ranking
  confidence for chunks already found, it does not rescue missed
  exact-token chunks in this eval.
- Generation-metric coverage is partial: 15 of 55 questions for
  dense_only vs. hybrid, none for BM25-only, both by design given Groq
  free-tier daily token limits (100k-200k tokens/day depending on model).
- LLM-as-judge validity is unverified against human judgment — no
  hand-labeled subset has been checked against the judge's scores yet.
- BM25 tokenization is simple (lowercase, word-split, stopword removal,
  no stemming) — sufficient at this corpus size, not benchmarked further.
- Retrieval assumes the answer lives in one authority; multi-hop
  questions spanning statute, regulation, and publication are a
  measured, unsolved weak category.

## What I'd add next

- Finish full-55 generation metrics for all three configs, and
  hand-label a subset to report actual judge/human agreement.
- Cross-encoder reranking and authority-weighted reordering — for
  queries that don't name a specific provision explicitly, the statute
  still doesn't reliably outrank the Publication describing it; exact
  citation lookup only helps when the query itself parses out a §/IRC
  citation.
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
python -m evals.run_eval --config hybrid --retrieval-only  # Recall@5/MRR, no LLM, seconds
python -m evals.run_eval --config dense_only                # full run: + generation + judge
```
