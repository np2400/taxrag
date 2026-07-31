# Design Notes — TaxRAG

Why this system is built the way it is. Companion to `ARCHITECTURE.md`, which covers file layout and data contracts.

---

## Why tax law is a hard retrieval problem

Naive RAG — chunk the documents, embed, retrieve top-k, generate — fails on tax questions in specific, reproducible ways. Each failure motivates a component in this system.

| Property of tax law | Consequence for retrieval |
|---|---|
| Hierarchical authority (IRC > Treas. Reg. > IRS publications) | Semantic similarity alone will surface a publication over the statute it summarizes |
| Provisions change by tax year | A correct 2023 answer can be wrong for 2025 |
| Exact tokens carry meaning: `§179`, `Form 8829`, `$0.67/mile` | Dense embeddings blur rare identifiers |
| Answers are worthless without citation | Fluency is not the quality bar; groundedness is |
| Much of the arithmetic is deterministic | Language models are the wrong tool for it |
| Some questions genuinely require a professional | The system needs a correct way to decline |

A concrete failure from early testing: a dense-only retriever asked about the **§179** expensing election returned chunks about **§199A**, because the two section identifiers are nearly indistinguishable in embedding space and are legally unrelated. That single failure is what motivated hybrid retrieval.

---

## Scope

**In scope — three topics, federal only:**
- Home office deduction (§280A), simplified and actual methods
- Vehicle and mileage deduction (§162, §274(d))
- Self-employment tax (§1401) and Schedule C basics

**Deliberately out of scope:** QBI/§199A, state tax, entity returns, depreciation beyond basics.

Narrow scope with measured results is more useful than broad scope with none. QBI in particular carries phase-in thresholds, SSTB classification, and W-2 wage limitations that would have consumed the evaluation budget without adding retrieval insight.

### Corpus

| Source | Authority weight |
|---|---|
| IRC §162, §179, §274(d), §280A, §1401 | 1.0 |
| Treas. Reg. §1.274-5 | 0.9 |
| Schedule C / SE / Form 8829 instructions | 0.5 |
| Publications 334, 463, 587 | 0.4 |

IRS publications are guidance, not authority. Where a publication and the statute it interprets appear to conflict, the system surfaces the statute.

**Note:** an initial version of this table also listed a Treas. Reg. §1.280A-2. During ingestion, a search of eCFR (the current, official compilation of adopted federal regulations) turned up no regulation under §280A — the home-office rules were only ever proposed, never finalized. The authority chain for home office is therefore IRC §280A → Publication 587 directly, with no regulation layer in between, which is a sharper illustration of the authority-hierarchy problem than a two-tier statute/reg split would have been.

---

## Retrieval design

### Hybrid: BM25 + dense, fused with Reciprocal Rank Fusion

Tax queries split roughly evenly between two types:

- **Conceptual** — *"can I write off my truck"* — handled well by dense embeddings
- **Exact-identifier** — *"what does §280A(c)(1) require"* — handled poorly by dense embeddings, handled well by BM25

Neither retriever covers both. Running them in parallel and fusing does.

**Why RRF rather than weighted score fusion:** BM25 scores and cosine similarities occupy incompatible ranges, so blending raw scores requires a normalization constant that doesn't transfer across query types. RRF operates only on rank position — no tuning, and it degrades gracefully when one retriever returns nothing useful.

### Cross-encoder reranking

Bi-encoders embed query and document independently, which is what makes them fast enough to scan a corpus. Cross-encoders attend over the pair jointly and are substantially more accurate, but too slow for full-corpus scan. Applying one to the top 20 candidates and returning 5 captures most of the accuracy gain at bounded cost.

### Authority-weighted reordering

When two chunks answer the same question, the one with higher authority weight is preferred. This is domain-specific and doesn't generalize outside legal or regulatory corpora, but within them it matters: a taxpayer relying on a publication where the regulation says something narrower has a real problem.

---

## Deterministic tools over generated arithmetic

Self-employment tax and the home office deduction are computed, not looked up. Both are implemented as unit-tested Python functions with rates in a version-keyed config, so a new tax year is a data change rather than a code change.

A language model producing a plausible but incorrect self-employment tax figure is worse than one that declines — the error is invisible to the person least equipped to catch it. Tools make the arithmetic auditable and correct by construction.

---

## Citation verification

Every generated answer passes through a verification step that checks:

1. Each factual claim carries a citation
2. The cited chunk actually supports the claim
3. The tax year in the answer matches the year in the question

Failures trigger one retry with a reformulated query, then an explicit refusal.

**Refusal is a designed outcome, not a failure mode.** In a regulated domain, declining to answer is frequently the correct behavior, and refusal precision is measured as a first-class metric alongside recall.

---

## Evaluation

A 50-question evaluation set with ground-truth answers and required citations, authored by hand against source documents and reviewed by a CPA. Categories: factual lookup, exact-identifier, computational, temporal, out-of-scope, and adversarial.

The adversarial category tests whether the system corrects a false premise rather than accommodating it — e.g. *"I can deduct my whole car since I use it for work, right?"* Agreement under social pressure is a measurable failure mode.

**Metrics** are reported separately for retrieval (Recall@5, MRR) and generation (groundedness, citation accuracy), plus behavioral metrics (refusal precision, hallucinated-citation rate).

An ablation study isolates each component's contribution by re-running the same evaluation set with components enabled progressively. Results are in the README.

---

## Known limitations

- **Multi-hop reasoning is the weakest category.** Questions requiring synthesis across statute, regulation, and a ruling that qualifies both are not handled well. Retrieval assumes the answer lives in a single authority. Graph-based retrieval over citation cross-references is the natural fix.
- **LLM-as-judge correlates imperfectly with human judgment.** A subset of the evaluation set was hand-labeled to measure agreement; the figure is reported in the README rather than assumed.
- **Single-user, local deployment.** ChromaDB runs in-process. Concurrent use would require pgvector or Qdrant.
- **Federal only, three topics.** Not a general tax research tool.

---

## Not legal or tax advice

This system surfaces primary tax authority and shows its sources. It does not advise. Questions requiring judgment about a specific taxpayer's circumstances are declined by design.
