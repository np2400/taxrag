"""Answer synthesis with inline citations, via Groq's free-tier API.

The system prompt is the actual product here: it restricts the model to
the provided chunks only (never its own background knowledge), requires
every claim to carry an exact citation string, and defines a literal
refusal marker for when the provided context genuinely doesn't answer
the question.
"""

import sys
import time

_t0 = time.perf_counter()
from groq import Groq
print(f"[timing] import groq: {time.perf_counter() - _t0:.3f}s", file=sys.stderr)

from config.settings import GROQ_API_KEY, GROQ_MODEL_NAME, REFUSAL_MARKER
from src.types import Answer, RetrievalResult

_SYSTEM_PROMPT = f"""You are a tax research assistant. Answer the user's question
using ONLY the numbered source excerpts provided below — never your own
background knowledge of tax law, which may be outdated or wrong for this
system's purposes.

Rules:
- Every factual claim must be followed by its exact citation in parentheses,
  copied verbatim from the source excerpt it came from (e.g. "(IRC §280A(c)(1))").
- Do not invent or paraphrase a citation. Only cite sources that were provided.
- If the provided excerpts do not contain enough information to answer the
  question, respond with exactly: {REFUSAL_MARKER}
- This is a research aid, not tax advice. Do not tell the user what they
  "should" do — state what the authority provides.
"""


def _build_context(results: list[RetrievalResult]) -> str:
    parts = [f"[{r.chunk.citation}]\n{r.chunk.text}" for r in results]
    return "\n\n".join(parts)


def generate_answer(query: str, results: list[RetrievalResult]) -> Answer:
    """Synthesize a citation-grounded answer from retrieved chunks."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set — check .env exists and config.settings loaded it."
        )

    client = Groq(api_key=GROQ_API_KEY)
    context = _build_context(results)

    t0 = time.time()
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Source excerpts:\n\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0.0,
    )
    latency = time.time() - t0

    text = response.choices[0].message.content.strip()
    refused = REFUSAL_MARKER in text

    # Citation extraction, deliberately simple for Phase 1: a citation
    # counts if it (or its prefix, before " — ") appears in the answer
    # AND was actually retrieved. The prefix fallback matters for PDF
    # chunks: their internal citation is long and descriptive (e.g.
    # "Pub. 334 — Standard mileage rate..."), but a model naturally
    # writes the short form "(Pub. 334)" inline, not the full string —
    # verified by an actual run, not assumed. Whether a cited chunk
    # truly supports the specific claim next to it is verifier.py's
    # job (Phase 6).
    def _cited(citation: str) -> bool:
        return citation in text or citation.split(" — ")[0] in text

    cited = [r.chunk.citation for r in results if _cited(r.chunk.citation)]

    usage = response.usage
    trace = {
        "latency_s": round(latency, 2),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "cost_usd": 0.0,  # Groq free tier
        "model": GROQ_MODEL_NAME,
    }

    return Answer(
        text=text,
        citations=cited,
        refused=refused,
        refusal_reason="insufficient context" if refused else None,
        trace=trace,
    )
