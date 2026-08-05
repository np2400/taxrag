"""Exact citation lookup: when a query names a specific IRC provision,
fetch it directly by the citation metadata every chunk already carries,
instead of hoping an embedding or BM25 score happens to rank it highly.

Motivated by a traced failure (see DESIGN.md): asking "What does IRC
§280A(c)(1) require?" returned only one of its three alternative
subparagraphs via hybrid retrieval, presenting a single prong as if it
were the whole rule. The user already told us exactly which provision
they want; this bypasses ranking for that one case rather than trying
to make embeddings or BM25 rank it correctly.
"""

import json
import re
from dataclasses import dataclass

from config.settings import CHUNKS_PATH
from src.types import Chunk

# Trigger: "IRC" (optionally followed by §) or a bare "§", then a section
# number (digits + optional trailing letter, e.g. "280A", "1401"), then
# zero or more parenthetical path segments: (c)(1)(A)(i). The negative
# lookahead right after the section number rejects Treasury-regulation
# style dotted numbers ("§1.274-5"): without it, "1" would get parsed out
# of "1.274-5" as a bogus section number, colliding every regulation
# subsection into one false match. That also means this module only ever
# resolves IRC statute citations, not regulations -- consistent with the
# failure mode it exists to fix.
_CITATION_RE = re.compile(
    r"(?:§\s?|IRC\s+(?:§\s?)?)(\d+[A-Za-z]*)(?!\.)((?:\([a-zA-Z0-9]+\))*)",
    re.IGNORECASE,
)
_PATH_SEGMENT_RE = re.compile(r"\(([a-zA-Z0-9]+)\)")


@dataclass(frozen=True)
class ParsedCitation:
    section: str
    path: tuple[str, ...]  # e.g. ("c", "1", "a") for §280A(c)(1)

    @property
    def subsection(self) -> str | None:
        return self.path[0] if len(self.path) > 0 else None

    @property
    def paragraph(self) -> str | None:
        return self.path[1] if len(self.path) > 1 else None

    @property
    def subparagraph(self) -> str | None:
        return self.path[2] if len(self.path) > 2 else None


def parse_citation(text: str) -> ParsedCitation | None:
    """Extract an IRC section citation from free text, or None if the
    text doesn't name one. Case-insensitive; normalizes to lowercase so
    "280A" and "280a" compare equal, matching chunker.py's own slugify
    (lowercase, non-alnum runs collapsed to a single dash) -- which is
    what makes the id built in _target_id line up with real chunk_ids
    without importing chunker.py itself."""
    match = _CITATION_RE.search(text)
    if not match:
        return None
    section = match.group(1).lower()
    path = tuple(seg.lower() for seg in _PATH_SEGMENT_RE.findall(match.group(2)))
    return ParsedCitation(section=section, path=path)


def _target_id(parsed: ParsedCitation) -> str:
    return "-".join(["irc", parsed.section, *parsed.path])


_chunks_cache: list[Chunk] | None = None


def _load_chunks() -> list[Chunk]:
    """Same load as sparse.py -- small enough (~1,500 chunks) that
    re-reading the committed JSON here, independent of whichever
    retrievers happen to be enabled, is simpler than threading a shared
    chunk list through Pipeline."""
    global _chunks_cache
    if _chunks_cache is None:
        raw = json.loads(CHUNKS_PATH.read_text())
        _chunks_cache = [Chunk(**c) for c in raw]
    return _chunks_cache


def _closest_descendants(target_id: str, by_id: dict[str, Chunk]) -> list[str]:
    """The nearest existing chunk along each branch below target_id, not
    just chunks exactly one path segment deeper. Some provisions in this
    corpus have no chunk of their own and jump straight to a grandchild
    -- e.g. §1401(b)(2) has no chunk, §1401(b)(2)(B) does (one level
    down), but §1401(b)(2)(A) doesn't either, so its own children,
    §1401(b)(2)(A)(i)-(iii), are two levels down. A strict
    depth-plus-one rule would silently drop that whole (A) branch."""
    candidates = sorted(
        (cid for cid in by_id if cid.startswith(target_id + "-")),
        key=lambda cid: cid.count("-"),
    )
    selected: list[str] = []
    for cid in candidates:
        if not any(cid.startswith(sel + "-") for sel in selected):
            selected.append(cid)
    return selected


def exact_citation_lookup(parsed: ParsedCitation) -> list[Chunk]:
    """Resolve a parsed citation against the corpus via each chunk's own
    citation field (re-parsed with the same rule, not string-matched --
    citation text formatting shouldn't have to exactly match query
    phrasing), applying the scope rule:

    - The cited provision has descendants (whether or not it has a chunk
      of its own): return its own chunk, if any, plus the closest
      existing descendant along every branch -- so a query about a
      subsection returns the whole subsection, not one arbitrary child.
    - The cited provision is a leaf: return it plus its immediate
      parent's chunk, which usually carries the introductory language
      the leaf's own text assumes as context. Top-level provisions
      (e.g. §280A(a)) have no parent chunk to add.
    - No chunk resolves at all: empty list, so the pipeline falls back
      to whatever hybrid retrieval already found.

    Chunks whose citation doesn't parse as an IRC section (regulations,
    publications, instructions) are never candidates -- parse_citation's
    own scoping keeps this path statute-only.
    """
    by_id: dict[str, Chunk] = {}
    for c in _load_chunks():
        p = parse_citation(c.citation)
        if p is None or p.section != parsed.section:
            continue
        by_id[_target_id(p)] = c

    target_id = _target_id(parsed)
    exact = by_id.get(target_id)
    descendant_ids = _closest_descendants(target_id, by_id)

    if descendant_ids:
        return ([exact] if exact else []) + [by_id[cid] for cid in descendant_ids]

    if exact is None:
        return []

    if not parsed.path:
        return [exact]
    parent_id = "-".join(["irc", parsed.section, *parsed.path[:-1]])
    parent = by_id.get(parent_id)
    return [exact, parent] if parent else [exact]
