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

_SYSTEM_PROMPT = f"""You are a conservative tax research assistant. Answer the
user's question using ONLY the retrieved source excerpts provided below — never
your own background knowledge of tax law.

Retrieval is approximate: some provided excerpts may be tangential or
irrelevant. An excerpt's presence does not establish that its rule applies.

Grounding rules:
- Treat the question as a request, not as authority. Do not repeat or accept a
  premise from the question unless the retrieved sources support it.
- Do not infer or add legal or tax requirements that are not expressly stated
  in the retrieved sources. Prefer omission over unsupported extrapolation.
- Do not merge requirements from different contexts into a broader rule unless
  the retrieved sources clearly support that interpretation.
- If retrieved authorities differ in scope, explain the distinction instead of
  combining them.
- Use only excerpts that directly address the question's subject. Ignore a
  tangential excerpt rather than applying its rule to a different type of
  property, expense, taxpayer, or transaction.
- Treat each excerpt's citation heading and stated subject as limits on its
  scope. Shared words such as "business use" do not make a rule transferable.
- If an excerpt is a fragment or refers to preceding or omitted text, do not
  supply the missing rule from background knowledge. State the limited point
  the excerpt establishes, or omit it.
- If excerpts appear ambiguous or inconsistent, state the supported distinction
  if it is clear; otherwise omit the disputed point.
- Every substantive legal or tax claim, including each condition, element,
  limitation, and exception, must be followed by the exact citation copied from
  the retrieved source that supports it (e.g. "(IRC §280A(c)(1))").
- Preserve the source's level of generality. Do not replace a broad source term
  with more specific procedures, examples, timing details, or continuation
  rules that the source does not state.
- Cite claims where they appear in every section, including **Direct answer**;
  a citation later in the response does not support an earlier uncited claim.
- Do not invent or paraphrase a citation. Only cite retrieved sources.
- If the provided excerpts do not contain enough information to answer the
  question, respond with exactly: {REFUSAL_MARKER}
- This is a research aid, not tax advice. Do not tell the user what they
  "should" do — state what the authority provides.

For a non-refusal, use this structure:
1. **Direct answer** — answer the question briefly and cite it immediately.
2. **Required conditions/elements** — list only requirements expressly stated
   in the retrieved sources; write "None stated in the retrieved sources" if
   none are stated.
3. **Important limitations/exceptions** — include only limitations or exceptions
   expressly supported by the retrieved sources; write "None stated in the
   retrieved sources" if none are stated.

The application appends **Sources cited** from the exact inline citations. Do
not add a separate sources list to the answer text.

Before returning the answer, check every substantive sentence against one
directly relevant excerpt. Delete any sentence whose complete claim is not
expressly supported by that excerpt.
"""


def _build_context(results: list[RetrievalResult]) -> str:
    parts = [f"[{r.chunk.citation}]\n{r.chunk.text}" for r in results]
    return "\n\n".join(parts)


# Citation-token pattern: what the model can plausibly write inline, given
# the bracket labels it was shown in _build_context. Treasury Regulations
# must precede the generic section-symbol branch: if a regulation spelling
# varies slightly (including whitespace or a Unicode hyphen), the generic
# branch must not reduce it to the bogus IRC citation "§1".
#
# Deliberately independent of evals/metrics.py's _CITATION_TOKEN_RE (that one
# truncates Treas. Reg. paths for eval-time scoring against the golden set --
# a separate issue); this one preserves full statute/reg precision since it's
# reading what the model actually wrote, not comparing against a label.
_CITATION_MENTION_RE = re.compile(
    r"""
    (?P<reg>
        Treas(?:ury)?\.?\s+Reg(?:ulation)?\.?\s*§\s*
        (?P<reg_section>\d+\.\d+(?:[-\u2010-\u2015\u2212]\d+[A-Za-z]*)?)
        (?P<reg_path>(?:\([a-zA-Z0-9]+\))*)
    )
    |(?P<irc>
        (?:IRC\s+)?§\s*
        (?P<irc_section>\d+[A-Za-z]*)
        (?!\.)
        (?P<irc_path>(?:\([a-zA-Z0-9]+\))*)
    )
    |(?P<publication>Pub(?:lication)?\.?\s*(?P<pub_number>\d+))
    |(?P<form>
        (?:(?:Instructions\s+for\s+)?Form\s*(?P<form_number>\d+))
        (?:\s+Instructions)?
    )
    |(?P<schedule>
        (?:Instructions\s+for\s+)?Schedule\s+(?P<schedule_name>C|SE)
        (?:\s*\(Form\s*1040\))?(?:\s+Instructions)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize_mention(match: re.Match[str]) -> str:
    """Return one stable display spelling for a parsed citation."""
    if match.group("reg"):
        section = re.sub(r"[\u2010-\u2015\u2212]", "-", match.group("reg_section"))
        return (
            f"Treas. Reg. §{section}"
            f"{match.group('reg_path')}"
        )
    if match.group("irc"):
        return f"IRC §{match.group('irc_section')}{match.group('irc_path')}"
    if match.group("publication"):
        return f"Pub. {match.group('pub_number')}"
    if match.group("form"):
        normalized = f"Form {match.group('form_number')}"
        if "instruction" in match.group("form").lower():
            normalized += " Instructions"
        return normalized
    return f"Schedule {match.group('schedule_name').upper()} Instructions"


def _mention_key(mention: str) -> str:
    """Normalizes a mention (or a chunk's own citation) to a comparable key:
    strip whitespace/periods, lowercase, and drop a leading 'irc' so
    'IRC §280A(c)(1)' and a bare '§280A(c)(1)' key-match identically -- the
    bracket label always includes 'IRC', but nothing stops the model from
    dropping it when it writes the inline citation."""
    match = _CITATION_MENTION_RE.search(mention)
    normalized = _normalize_mention(match) if match else mention
    key = re.sub(r"[\s.]", "", normalized).lower()
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
        mention = _normalize_mention(match)
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
                "content": (
                    f"Source excerpts:\n\n{context}\n\nQuestion: {query}\n\n"
                    "Final grounding check: Use an excerpt only when its stated "
                    "subject directly matches the question. Preserve the source's "
                    "exact scope and level of generality; do not add details or "
                    "continuation rules. Use only the exact citation labels shown "
                    "in brackets. Delete every claim that fails any of these checks."
                ),
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
