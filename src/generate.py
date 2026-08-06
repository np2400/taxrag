"""Answer synthesis with inline citations, via Groq's free-tier API.

The system prompt is the actual product here: it restricts the model to
the provided chunks only (never its own background knowledge), requires
every claim to carry an exact citation string, and defines a literal
refusal marker for when the provided context genuinely doesn't answer
the question.
"""

import re
import time

from groq import Groq

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


# Citation-token pattern: what the model can plausibly write inline, given
# the bracket labels it was shown in _build_context. Deliberately independent
# of evals/metrics.py's _CITATION_TOKEN_RE (that one truncates Treas. Reg.
# paths for eval-time scoring against the golden set -- a separate, not yet
# fixed issue); this one preserves full statute/reg precision since it's
# reading what the model actually wrote, not comparing against a label.
_CITATION_MENTION_RE = re.compile(
    r"IRC\s+§\s?\d+[A-Za-z]*(?:\([a-zA-Z0-9]+\))*"
    r"|§\s?\d+[A-Za-z]*(?:\([a-zA-Z0-9]+\))*"
    r"|Treas\.\s?Reg\.\s?§[\d.]+-\d+(?:\([a-zA-Z0-9]+\))*"
    r"|Pub\.?\s?\d+"
    r"|Form\s?\d+"
    r"|Schedule C Instructions"
    r"|Schedule SE Instructions",
    re.IGNORECASE,
)


def _mention_key(mention: str) -> str:
    """Normalizes a mention (or a chunk's own citation) to a comparable key:
    strip whitespace/periods, lowercase, and drop a leading 'irc' so
    'IRC §280A(c)(1)' and a bare '§280A(c)(1)' key-match identically -- the
    bracket label always includes 'IRC', but nothing stops the model from
    dropping it when it writes the inline citation."""
    key = re.sub(r"[\s.]", "", mention).lower()
    return key[3:] if key.startswith("irc") else key


def _extract_citations(text: str, results: list[RetrievalResult]) -> list[str]:
    """Citations actually present in the generated text, not every retrieved
    chunk whose document prefix happens to appear in it. A mention is kept
    only if it's grounded -- its key is a prefix of, or equal to, at least
    one retrieved chunk's citation key -- which rejects hallucinated
    citations without expanding one generic mention (e.g. "(Pub. 463)")
    into every retrieved chunk that shares that document prefix: the
    retrieved chunk's own longer citation is never substituted in, only
    what the model actually wrote is kept. Order of first appearance is
    preserved; a citation mentioned more than once is kept only once."""
    retrieved_keys = {_mention_key(r.chunk.citation) for r in results}
    seen: set[str] = set()
    citations: list[str] = []
    for match in _CITATION_MENTION_RE.finditer(text):
        mention = match.group(0)
        key = _mention_key(mention)
        if key in seen:
            continue
        if not any(rk.startswith(key) or key.startswith(rk) for rk in retrieved_keys):
            continue
        seen.add(key)
        citations.append(mention)
    return citations


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

    # Whether a cited chunk truly supports the specific claim next to it
    # (not just that the citation was mentioned and grounded) is
    # verifier.py's job (Phase 6).
    cited = _extract_citations(text, results)

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
