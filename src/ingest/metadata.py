"""Document-level metadata registry.

Maps each raw corpus file (data/raw/) to the authority weight, source type,
citation prefix, and source URL stamped onto every chunk produced from it.
Keyed by filename since loader.py already knows the filename when it reads
a file — the natural join key, no fragile string matching required.

tax_year_start/tax_year_end default to None (current, unrestricted) for
every document. Determining precise per-provision effective years requires
per-subsection legal research beyond a Phase 1 ingestion pass — a known,
documented simplification, not a fabricated fact. Revisit before the
temporal-filtering eval category (SPEC.md Sec 4.1) is treated as validated.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentMeta:
    citation_prefix: str
    source_type: str  # statute | regulation | instruction | publication
    authority_weight: float
    tax_year_start: int | None
    tax_year_end: int | None
    url: str


DOCUMENT_REGISTRY: dict[str, DocumentMeta] = {
    "irc-162.html": DocumentMeta(
        citation_prefix="IRC §162",
        source_type="statute",
        authority_weight=1.0,
        tax_year_start=None,
        tax_year_end=None,
        url="https://www.law.cornell.edu/uscode/text/26/162",
    ),
    "irc-179.html": DocumentMeta(
        citation_prefix="IRC §179",
        source_type="statute",
        authority_weight=1.0,
        tax_year_start=None,
        tax_year_end=None,
        url="https://www.law.cornell.edu/uscode/text/26/179",
    ),
    "irc-274.html": DocumentMeta(
        citation_prefix="IRC §274",
        source_type="statute",
        authority_weight=1.0,
        tax_year_start=None,
        tax_year_end=None,
        url="https://www.law.cornell.edu/uscode/text/26/274",
    ),
    "irc-280a.html": DocumentMeta(
        citation_prefix="IRC §280A",
        source_type="statute",
        authority_weight=1.0,
        tax_year_start=None,
        tax_year_end=None,
        url="https://www.law.cornell.edu/uscode/text/26/280A",
    ),
    "irc-1401.html": DocumentMeta(
        citation_prefix="IRC §1401",
        source_type="statute",
        authority_weight=1.0,
        tax_year_start=None,
        tax_year_end=None,
        url="https://www.law.cornell.edu/uscode/text/26/1401",
    ),
    "treas-reg-1.274-5.html": DocumentMeta(
        citation_prefix="Treas. Reg. §1.274-5",
        source_type="regulation",
        authority_weight=0.9,
        tax_year_start=None,
        tax_year_end=None,
        url="https://www.law.cornell.edu/cfr/text/26/1.274-5",
    ),
    "pub-334.pdf": DocumentMeta(
        citation_prefix="Pub. 334",
        source_type="publication",
        authority_weight=0.4,
        tax_year_start=None,
        tax_year_end=None,
        url="https://www.irs.gov/pub/irs-pdf/p334.pdf",
    ),
    "pub-463.pdf": DocumentMeta(
        citation_prefix="Pub. 463",
        source_type="publication",
        authority_weight=0.4,
        tax_year_start=None,
        tax_year_end=None,
        url="https://www.irs.gov/pub/irs-pdf/p463.pdf",
    ),
    "pub-587.pdf": DocumentMeta(
        citation_prefix="Pub. 587",
        source_type="publication",
        authority_weight=0.4,
        tax_year_start=None,
        tax_year_end=None,
        url="https://www.irs.gov/pub/irs-pdf/p587.pdf",
    ),
    "i1040sc-schedule-c-instructions.pdf": DocumentMeta(
        citation_prefix="Schedule C Instructions",
        source_type="instruction",
        authority_weight=0.5,
        tax_year_start=None,
        tax_year_end=None,
        url="https://www.irs.gov/pub/irs-pdf/i1040sc.pdf",
    ),
    "i1040sse-schedule-se-instructions.pdf": DocumentMeta(
        citation_prefix="Schedule SE Instructions",
        source_type="instruction",
        authority_weight=0.5,
        tax_year_start=None,
        tax_year_end=None,
        url="https://www.irs.gov/pub/irs-pdf/i1040sse.pdf",
    ),
    "i8829-form-8829-instructions.pdf": DocumentMeta(
        citation_prefix="Form 8829 Instructions",
        source_type="instruction",
        authority_weight=0.5,
        tax_year_start=None,
        tax_year_end=None,
        url="https://www.irs.gov/pub/irs-pdf/i8829.pdf",
    ),
}
