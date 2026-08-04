"""CLI: run the golden set through a pipeline configuration and record
every metric from metrics.py to a results file.

    python -m evals.run_eval --config dense_only
    python -m evals.run_eval --config dense_only --limit 10      # cheap smoke test
    python -m evals.run_eval --config dense_only --resume        # continue after a crash

Retrieval metrics (Recall@5, MRR) are measured against the retriever
directly, not through Pipeline.answer() -- the frozen Answer contract
(src/types.py) only carries the final generated text, citations, and a
refusal flag, not the underlying RetrievalResult list, so there's no way
to recover "what did retrieval return" from an Answer alone. This calls
DenseRetriever the same way Pipeline does internally, then separately
calls Pipeline.answer() for the generation-side metrics -- two calls to
the same deterministic retriever per question, not a race condition or
a modification to any Phase-1 file.

Each fresh run is a full, independent measurement, written to its own
timestamped file -- nothing here mutates the pipeline, the golden set,
or generate.py. This file and metrics.py are everything Phase 3 owns;
later phases add new --config entries as their retrievers/agents land,
without needing to change how a run is scored.

Every question costs two Groq calls (generate + judge; see judge_answer's
docstring in metrics.py), and a full 55-question run comes in well over
a free-tier daily token cap -- see planning/NOTES.md for the numbers and
why --limit/--resume/checkpointing below exist.
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
from src.pipeline import Pipeline, PipelineConfig
from src.retrieval.dense import DenseRetriever

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Ablation rows from SPEC §4.3, expressed as PipelineConfig. Only
# "dense_only" does anything different from the pipeline's own defaults
# right now -- Phase 4/5/6 add the retrievers and agents that make the
# other rows meaningful. This dict is the one place later phases extend,
# without needing to change how a run is scored.
_CONFIGS = {
    "dense_only": PipelineConfig(use_dense=True),
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


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _summarize(per_question: list[dict]) -> dict | None:
    if not per_question:
        return None
    return {
        "n_questions": len(per_question),
        "recall_at_5": _mean([r["recall_at_5"] for r in per_question]),
        "mrr": _mean([r["mrr"] for r in per_question]),
        "citation_accuracy": _mean([r["citation_accuracy"] for r in per_question]),
        "groundedness": _mean([r["groundedness"] for r in per_question]),
        "premise_correction": _mean([r["premise_correction"] for r in per_question]),
        "hallucinated_citation_rate": _mean(
            [1.0 if r["hallucinated_citations"] else 0.0 for r in per_question]
        ),
        "refusal_precision": refusal_precision(
            [
                {"category": r["category"], "refused": r["refused"]}
                for r in per_question
            ]
        ),
        "cost_usd_total": round(sum(r.get("cost_usd", 0.0) for r in per_question), 4),
    }


def _latest_results_file(config_name: str) -> Path | None:
    """Most recently modified results file for this config -- what
    --resume continues from."""
    candidates = sorted(
        RESULTS_DIR.glob(f"{config_name}_*.json"), key=lambda p: p.stat().st_mtime
    )
    return candidates[-1] if candidates else None


def _checkpoint(out_path: Path, config_name: str, per_question: list[dict], complete: bool) -> None:
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
                "complete": complete,
                "summary": _summarize(per_question),
                "per_question": per_question,
            },
            indent=2,
        )
    )


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

    retriever = DenseRetriever()
    pipeline = Pipeline(config)

    for q in questions:
        if q["id"] in done_ids:
            continue

        required = _required_citation_for(q)
        retrieved = retriever.retrieve(
            q["question"], k=K_RETRIEVE, tax_year=q["tax_year"]
        )

        t0 = time.time()
        try:
            answer = pipeline.answer(q["question"], tax_year=q["tax_year"])
        except RateLimitError as e:
            _checkpoint(out_path, config_name, per_question, complete=False)
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
                _checkpoint(out_path, config_name, per_question, complete=False)
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
        _checkpoint(out_path, config_name, per_question, complete=False)

    _checkpoint(out_path, config_name, per_question, complete=True)
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
        "change, instead of spending the full ~150-220k-token run every time.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the most recent incomplete run for this config instead of "
        "starting over from question 1.",
    )
    args = parser.parse_args()

    out_path = run(args.config, limit=args.limit, resume=args.resume)

    result = json.loads(out_path.read_text())
    print(f"\n{'Wrote' if result['complete'] else 'Partial (see message above):'} {out_path}\n")
    if result["summary"]:
        for key, value in result["summary"].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
