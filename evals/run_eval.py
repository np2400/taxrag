"""CLI: run the golden set through a pipeline configuration and record
every metric from metrics.py to a results file.

    python -m evals.run_eval --config hybrid --retrieval-only   # Recall@5/MRR only, no LLM, free
    python -m evals.run_eval --config dense_only                # full run: + generation + judge
    python -m evals.run_eval --config dense_only --limit 10      # cheap smoke test
    python -m evals.run_eval --config dense_only --resume        # continue after a crash

Retrieval metrics (Recall@5, MRR) never touch an LLM at all -- they're
pure set comparison between a required citation and the retrieved chunk
IDs (see recall_at_k/reciprocal_rank in metrics.py, which take a
RetrievalResult list directly). --retrieval-only runs exactly that: build
whichever retriever(s) the config calls for, retrieve, score, done --
zero Groq calls, zero rate-limit risk, runs in seconds against the local
embedding model and pure-Python BM25.

Only the generation-side metrics (citation accuracy, groundedness,
premise-correction, hallucinated-citation rate) need Pipeline.answer()
and the LLM judge, which is what actually costs tokens and hits Groq's
daily cap -- see planning/NOTES.md.

_retrieve_for_config() duplicates the retriever-selection logic that
also lives in Pipeline.answer() -- necessary because the frozen Answer
contract (src/types.py) doesn't expose the RetrievalResult list a
generation call used internally, so retrieval-only scoring has no way to
recover it from an Answer alone. Kept as one shared helper here rather
than repeated inline, so both --retrieval-only and the full run compute
retrieval the same way instead of drifting apart.

Each fresh run is a full, independent measurement, written to its own
timestamped file -- nothing here mutates the pipeline, the golden set,
or generate.py. Phase 3 built this file and metrics.py; later phases add
new --config entries as their retrievers/agents land, without needing to
change how a run is scored.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from groq import RateLimitError

from config.settings import K_FINAL, K_RETRIEVE
from evals.metrics import (
    citation_accuracy,
    hallucinated_citations,
    judge_answer,
    reciprocal_rank,
    recall_at_k,
    refusal_precision,
)
from src.pipeline import Pipeline, PipelineConfig, prioritize_citation_chunks
from src.retrieval.citation import exact_citation_lookup, parse_citation
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.sparse import SparseRetriever
from src.types import RetrievalResult

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Ablation rows from SPEC §4.3, expressed as PipelineConfig. Phase 4 adds
# bm25_only/hybrid; Phase 5/6 add the rerank/authority/verifier rows as
# those components land. This dict is the one place later phases extend,
# without needing to change how a run is scored.
_CONFIGS = {
    "dense_only": PipelineConfig(use_dense=True, use_sparse=False),
    "bm25_only": PipelineConfig(use_dense=False, use_sparse=True),
    "hybrid": PipelineConfig(use_dense=True, use_sparse=True),
}


def _load_golden_set(limit: int | None = None) -> list[dict]:
    data = json.loads(GOLDEN_SET_PATH.read_text())
    questions = data["questions"]
    return questions[:limit] if limit else questions


def _required_citation_for(question: dict) -> str:
    """Out-of-scope questions carry a "n/a -- refusal..." placeholder in
    draft_citation rather than a real citation. Treat that as "no citation
    required" so recall/MRR/citation-accuracy correctly return None for
    them instead of scoring a correct refusal as a retrieval miss."""
    if question["category"] == "out_of_scope":
        return ""
    return question["draft_citation"]


def _retrieve_for_config(
    config: PipelineConfig,
    dense: DenseRetriever | None,
    sparse: SparseRetriever | None,
    query: str,
    tax_year: int | None,
) -> list[RetrievalResult]:
    """Same retrieval + fusion + citation-lookup + truncation logic
    Pipeline.answer() runs internally, exposed here since retrieval-only
    scoring needs the raw RetrievalResult list, not just a generated
    Answer."""
    needs_candidate_pool = config.use_rerank or (config.use_dense and config.use_sparse)
    k = config.k_retrieve if needs_candidate_pool else config.k_final

    if config.use_dense and config.use_sparse:
        dense_results = dense.retrieve(query, k=k, tax_year=tax_year)
        sparse_results = sparse.retrieve(query, k=k, tax_year=tax_year)
        results = reciprocal_rank_fusion(dense_results, sparse_results)
    elif config.use_sparse:
        results = sparse.retrieve(query, k=k, tax_year=tax_year)
    else:
        results = dense.retrieve(query, k=k, tax_year=tax_year)

    if config.use_citation_lookup:
        parsed = parse_citation(query)
        if parsed is not None:
            citation_chunks = exact_citation_lookup(parsed)
            if citation_chunks:
                results = prioritize_citation_chunks(citation_chunks, results)

    return results[: config.k_final]


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _summarize(per_question: list[dict]) -> dict | None:
    if not per_question:
        return None
    summary = {
        "n_questions": len(per_question),
        "recall_at_5": _mean([r["recall_at_5"] for r in per_question]),
        "mrr": _mean([r["mrr"] for r in per_question]),
    }
    if "citation_accuracy" in per_question[0]:
        summary.update(
            {
                "citation_accuracy": _mean(
                    [r["citation_accuracy"] for r in per_question]
                ),
                "groundedness": _mean([r["groundedness"] for r in per_question]),
                "premise_correction": _mean(
                    [r["premise_correction"] for r in per_question]
                ),
                "hallucinated_citation_rate": _mean(
                    [1.0 if r["hallucinated_citations"] else 0.0 for r in per_question]
                ),
                "refusal_precision": refusal_precision(
                    [
                        {"category": r["category"], "refused": r["refused"]}
                        for r in per_question
                    ]
                ),
                "cost_usd_total": round(
                    sum(r.get("cost_usd", 0.0) for r in per_question), 4
                ),
            }
        )
    return summary


def _latest_results_file(config_name: str, suffix: str = "") -> Path | None:
    """Most recently modified results file for this config -- what
    --resume continues from."""
    candidates = sorted(
        RESULTS_DIR.glob(f"{config_name}{suffix}_*.json"), key=lambda p: p.stat().st_mtime
    )
    return candidates[-1] if candidates else None


def _checkpoint(
    out_path: Path,
    config_name: str,
    per_question: list[dict],
    complete: bool,
    retrieval_only: bool,
) -> None:
    """Overwrites out_path with current progress. Called after every
    single question, not just at the end -- so a crash (rate limit,
    network blip, Ctrl-C) leaves a file with everything completed so far
    instead of nothing. `complete=False` marks a partial file so --resume
    knows to keep going; the final call sets it True.
    """
    out_path.write_text(
        json.dumps(
            {
                "config": config_name,
                "retrieval_only": retrieval_only,
                "complete": complete,
                "summary": _summarize(per_question),
                "per_question": per_question,
            },
            indent=2,
        )
    )


def _run_retrieval_only(config_name: str, limit: int | None) -> Path:
    """Recall@5 + MRR for every question, no Groq calls at all. Meant to
    be run fresh each time (no --resume needed -- it's seconds, not
    hours, so there's nothing worth resuming)."""
    config = _CONFIGS[config_name]
    questions = _load_golden_set(limit)

    dense = DenseRetriever() if config.use_dense else None
    sparse = SparseRetriever() if config.use_sparse else None

    per_question = []
    for q in questions:
        required = _required_citation_for(q)
        retrieved = _retrieve_for_config(config, dense, sparse, q["question"], q["tax_year"])
        per_question.append(
            {
                "id": q["id"],
                "category": q["category"],
                "recall_at_5": recall_at_k(required, retrieved, k=5),
                "mrr": reciprocal_rank(required, retrieved),
            }
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{config_name}_retrievalonly_{timestamp}.json"
    _checkpoint(out_path, config_name, per_question, complete=True, retrieval_only=True)
    return out_path


def run(config_name: str, limit: int | None = None, resume: bool = False) -> Path:
    if config_name not in _CONFIGS:
        raise ValueError(f"Unknown config '{config_name}'. Options: {list(_CONFIGS)}")
    config = _CONFIGS[config_name]

    questions = _load_golden_set(limit)

    per_question: list[dict] = []
    done_ids: set[str] = set()
    if resume:
        existing = _latest_results_file(config_name)
        if existing is None:
            raise FileNotFoundError(
                f"--resume given but no results file for '{config_name}' exists in {RESULTS_DIR}"
            )
        prior = json.loads(existing.read_text())
        if prior.get("complete"):
            print(f"{existing} is already complete -- nothing to resume.")
            return existing
        per_question = prior["per_question"]
        done_ids = {r["id"] for r in per_question}
        out_path = existing
        print(f"Resuming {existing} -- {len(done_ids)} question(s) already done.")
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"{config_name}_{timestamp}.json"

    dense = DenseRetriever() if config.use_dense else None
    sparse = SparseRetriever() if config.use_sparse else None
    pipeline = Pipeline(config)

    for q in questions:
        if q["id"] in done_ids:
            continue

        required = _required_citation_for(q)
        retrieved = _retrieve_for_config(config, dense, sparse, q["question"], q["tax_year"])

        t0 = time.time()
        try:
            answer = pipeline.answer(q["question"], tax_year=q["tax_year"])
        except RateLimitError as e:
            _checkpoint(out_path, config_name, per_question, complete=False, retrieval_only=False)
            print(
                f"\nRate limit hit generating the answer for {q['id']}. "
                f"{len(per_question)}/{len(questions)} question(s) saved to {out_path}.\n"
                f"Rerun with `--config {config_name} --resume` once the limit clears.\n{e}"
            )
            return out_path
        latency = time.time() - t0

        if answer.refused:
            # Nothing to judge for groundedness/premise-correction: a
            # refusal makes no factual claims and corrects nothing.
            groundedness = None
            premise_correction = None
            judge_reasoning = "skipped -- system refused, no claims to judge"
        else:
            try:
                judged = judge_answer(q["question"], answer.text, retrieved[:K_FINAL])
            except RateLimitError as e:
                _checkpoint(out_path, config_name, per_question, complete=False, retrieval_only=False)
                print(
                    f"\nRate limit hit judging the answer for {q['id']}. "
                    f"{len(per_question)}/{len(questions)} question(s) saved to {out_path}.\n"
                    f"Rerun with `--config {config_name} --resume` once the limit clears.\n{e}"
                )
                return out_path
            groundedness = judged.get("groundedness")
            premise_correction = judged.get("premise_correction")
            judge_reasoning = judged.get("reasoning")

        per_question.append(
            {
                "id": q["id"],
                "category": q["category"],
                "recall_at_5": recall_at_k(required, retrieved, k=5),
                "mrr": reciprocal_rank(required, retrieved),
                "citation_accuracy": citation_accuracy(required, answer.citations),
                "hallucinated_citations": hallucinated_citations(
                    answer.text, retrieved
                ),
                "groundedness": groundedness,
                "premise_correction": premise_correction,
                "judge_reasoning": judge_reasoning,
                "refused": answer.refused,
                "latency_s": round(latency, 2),
                "cost_usd": answer.trace.get("cost_usd") or 0.0,
            }
        )
        # Checkpoint after every question -- see _checkpoint's docstring
        # for why this isn't just done once at the very end.
        _checkpoint(out_path, config_name, per_question, complete=False, retrieval_only=False)

    _checkpoint(out_path, config_name, per_question, complete=True, retrieval_only=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the golden set through a pipeline config and score it."
    )
    parser.add_argument("--config", required=True, choices=sorted(_CONFIGS))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions -- a cheap smoke test after a code "
        "change, instead of spending the full run's tokens every time.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the most recent incomplete run for this config instead of "
        "starting over from question 1.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score Recall@5/MRR only -- no Groq calls, no rate-limit risk, runs "
        "in seconds. Skips citation accuracy/groundedness/refusal precision, "
        "which need Pipeline.answer() and the LLM judge.",
    )
    args = parser.parse_args()

    if args.retrieval_only:
        out_path = _run_retrieval_only(args.config, limit=args.limit)
    else:
        out_path = run(args.config, limit=args.limit, resume=args.resume)

    result = json.loads(out_path.read_text())
    print(f"\n{'Wrote' if result['complete'] else 'Partial (see message above):'} {out_path}\n")
    if result["summary"]:
        for key, value in result["summary"].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
