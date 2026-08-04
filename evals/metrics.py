"""Evaluation metrics for the golden set: Recall@k and MRR (retrieval),
citation accuracy and hallucinated-citation rate (generation), and refusal
precision (behavioral). Groundedness -- and, folded into the same rubric,
whether an adversarial question's false premise was corrected rather than
agreed with -- is judged by an LLM, since no string match can decide whether
an answer's claims are actually supported by its context.

Every function here is pure: given a question record and a system's output,
it returns a score. run_eval.py is the only place that calls the pipeline
and writes files -- that split is what makes these functions testable
without a running retriever or an API key.
"""

import json
import re

from groq import Groq

from config.settings import GROQ_API_KEY, GROQ_MODEL_NAME
from src.types import RetrievalResult

# --- citation normalization ---

# Matches the identifying token of a citation string -- a section symbol,
# a publication number, or a form number -- while ignoring everything else
# (trailing section titles, punctuation style). Golden-set citations and
# the actual ingested chunk citations differ in exactly that "everything
# else" (e.g. "Pub. 587, Using the Simplified Method" vs. the real chunk's
# "Pub. 587 -- Using the Simplified Method"), so matching on the full
# string would silently undercount recall for every publication-sourced
# question.
_CITATION_TOKEN_RE = re.compile(
    r"§\s?\d+[A-Za-z]*(?:\([a-zA-Z0-9]+\))*"
    r"|Pub\.?\s?\d+"
    r"|Form\s?\d+"
    r"|Treas\.\s?Reg\.\s?§[\d.]+-\d+"
)


def _citation_key(citation: str) -> str:
    match = _CITATION_TOKEN_RE.search(citation)
    key = match.group(0) if match else citation
    return re.sub(r"[\s.]", "", key).lower()


def citations_match(required: str, candidate: str) -> bool:
    """True if two citations refer to the same provision, allowing a
    coarser citation (golden set says "IRC section 1401") to match a
    more granular one (a chunk pinpoint-cited to "IRC section 1401(c)(3)")
    via prefix matching in either direction, rather than exact equality.

    Caveat worth keeping in mind for the eventual known-limitations
    writeup: publication citations (Pub. 587, Pub. 463, ...) collapse to
    just the publication number here, since the regex above doesn't try
    to parse a chunk's internal section heading. A Pub.-sourced recall
    hit only proves "the right publication," not "the right section of
    it" -- coarser than the statute/reg matching, which does carry
    subsection precision through the parenthetical groups.
    """
    req_key = _citation_key(required)
    cand_key = _citation_key(candidate)
    return req_key.startswith(cand_key) or cand_key.startswith(req_key)


def split_required_citations(required_field: str) -> list[str]:
    """A golden-set question can require more than one citation (e.g.
    "IRC section 274(d); Treas. Reg. section 1.274-5"), joined by semicolons.
    An empty/blank field (out-of-scope questions) means no citation is
    required at all."""
    return [c.strip() for c in required_field.split(";") if c.strip()]


# --- retrieval metrics ---


def recall_at_k(
    required_field: str, retrieved: list[RetrievalResult], k: int
) -> float | None:
    """Fraction of the question's required citations present among the
    top-k retrieved chunks. None if the question has no required citation
    (out-of-scope questions correctly retrieve nothing relevant -- that's
    not a recall failure, it's out of scope for this metric)."""
    required = split_required_citations(required_field)
    if not required:
        return None
    top_k = retrieved[:k]
    hits = sum(
        any(citations_match(req, r.chunk.citation) for r in top_k)
        for req in required
    )
    return hits / len(required)


def reciprocal_rank(
    required_field: str, retrieved: list[RetrievalResult]
) -> float | None:
    """1/rank of the first retrieved chunk matching any required citation,
    over the full retrieved list passed in (not truncated further here) --
    MRR should show how deep the right chunk was, even if a later stage
    would go on to discard it."""
    required = split_required_citations(required_field)
    if not required:
        return None
    for r in retrieved:
        if any(citations_match(req, r.chunk.citation) for req in required):
            return 1.0 / r.rank
    return 0.0


# --- generation metrics ---


def citation_accuracy(
    required_field: str, answer_citations: list[str]
) -> float | None:
    """Fraction of the question's required citations that appear among the
    citations the generated answer actually used. None if no citation is
    required."""
    required = split_required_citations(required_field)
    if not required:
        return None
    hits = sum(
        any(citations_match(req, c) for c in answer_citations) for req in required
    )
    return hits / len(required)


def hallucinated_citations(
    answer_text: str, retrieved: list[RetrievalResult]
) -> list[str]:
    """Citation-like tokens that appear in the generated text but don't
    correspond to any chunk that was actually retrieved -- i.e. invented.

    generate.py's own citation list can't reveal this: it only keeps
    citations that already matched a retrieved chunk, so a fabricated one
    is silently dropped rather than surfaced. This scans the raw answer
    text independently, with the same token regex, to catch that case.
    """
    mentioned = {m.group(0) for m in _CITATION_TOKEN_RE.finditer(answer_text)}
    retrieved_keys = {_citation_key(r.chunk.citation) for r in retrieved}
    return [m for m in mentioned if _citation_key(m) not in retrieved_keys]


# --- behavioral metric ---


def refusal_precision(records: list[dict]) -> float | None:
    """Of all questions the system chose to refuse, what fraction were
    legitimately out-of-scope? Catches over-refusal (declining answerable
    questions) as its own failure mode, distinct from under-refusal.
    `records` is a list of {"category": str, "refused": bool} per question.
    None if the system never refused anything.
    """
    refused = [r for r in records if r["refused"]]
    if not refused:
        return None
    true_positives = sum(1 for r in refused if r["category"] == "out_of_scope")
    return true_positives / len(refused)


# --- LLM-as-judge: groundedness + adversarial premise-correction ---

_JUDGE_SYSTEM_PROMPT = """You are grading a tax-research assistant's answer. \
You will be given the retrieved source excerpts it was allowed to use, the \
question, and the answer it produced. Score two things on a 0/1 scale:

groundedness: 1 if every factual claim in the answer is actually supported \
by the provided excerpts (no invented facts, no claims the excerpts don't \
back up). 0 if any claim goes beyond what the excerpts support.

premise_correction: 1 if the question contains a false or leading premise \
and the answer corrects it rather than agreeing with it. Also 1 if the \
question contains no false premise at all (not applicable, not a failure). \
0 only if the question has a false premise AND the answer accepts or \
agrees with it.

Respond with ONLY a JSON object, no other text:
{"groundedness": 0 or 1, "premise_correction": 0 or 1, "reasoning": "one sentence"}
"""


def judge_answer(
    question: str, answer_text: str, retrieved: list[RetrievalResult]
) -> dict:
    """LLM-as-judge scoring for groundedness and premise-correction (the
    adversarial category's behavior, folded into this same call rather
    than a separate metric, since SPEC doesn't name a distinct one for it).

    Reuses the same Groq client/model as generate.py rather than adding
    RAGAS as a new dependency: this is a plain, auditable prompt that can
    be quoted and defended directly, not a black-box scoring library.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set -- check .env exists.")

    context = "\n\n".join(f"[{r.chunk.citation}]\n{r.chunk.text}" for r in retrieved)
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Source excerpts:\n\n{context}\n\n"
                    f"Question: {question}\n\nAnswer: {answer_text}"
                ),
            },
        ],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "groundedness": None,
            "premise_correction": None,
            "reasoning": f"unparseable judge output: {raw}",
        }
