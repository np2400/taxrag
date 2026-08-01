"""Section-aware chunking: splits each source document into leaf-level legal
provisions, never mid-provision, using each document's own structural markup
rather than a guessed regex or indentation heuristic.

Three parsing strategies live here behind one shared header-injection step:
- USC statute pages (Cornell's semantic div-role markup: subsection >
  paragraph > subparagraph > clause > subclause > item)
- CFR regulation pages (a different Cornell template — see parse_cfr)
- PDF Pubs/instructions (no structural markup — heuristic)    <- next step
"""

import re
from collections import Counter
from dataclasses import dataclass

from bs4 import Tag

from src.ingest.loader import LoadedDocument, load_all_documents
from src.ingest.metadata import DocumentMeta
from src.types import Chunk

# Cornell's USC template nests provisions in divs whose first class token
# names the hierarchical role. Depth comes from DOM nesting, not from any
# ordering of this set.
USC_STRUCTURAL_ROLES = {
    "subsection",
    "paragraph",
    "subparagraph",
    "clause",
    "subclause",
    "item",
    "subitem",
}

# Cornell's own editorial annotations — not part of the operative statutory
# text. Excluded so the corpus never blends commentary with the actual law.
USC_SKIP_ROLES = {"note", "notes", "sourceCredit"}


@dataclass
class ParsedSection:
    """One provision found during a DOM walk, before citation/header text
    is built and before it becomes a Chunk.

    path: hierarchical (role, label) pairs from the section root down to
    this node, e.g. [("subsection", "c"), ("paragraph", "1")].
    text: this node's own text only — excludes any nested structural
    children, which become their own separate ParsedSection entries.
    """

    path: list[tuple[str, str]]
    text: str


def _first_class(tag: Tag) -> str:
    classes = tag.get("class") or []
    return classes[0] if classes else ""


def _own_text(div: Tag) -> str:
    """Text directly owned by this div: its non-structural, non-annotation
    child divs (typically 'content' and 'continuation'), excluding any
    nested structural children (handled separately, by recursion)."""
    parts = []
    for child in div.find_all("div", recursive=False):
        role = _first_class(child)
        if role in USC_STRUCTURAL_ROLES or role in USC_SKIP_ROLES:
            continue
        text = child.get_text(" ", strip=True)
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _walk_usc(
    div: Tag, path: list[tuple[str, str]], out: list[ParsedSection]
) -> None:
    own = _own_text(div)
    if own:
        out.append(ParsedSection(path=path, text=own))

    for child in div.find_all("div", recursive=False):
        role = _first_class(child)
        if role not in USC_STRUCTURAL_ROLES:
            continue
        label_span = child.find("span", class_="num")
        label = label_span.get("value") if label_span else "?"
        _walk_usc(child, path + [(role, label)], out)


def parse_usc(loaded: LoadedDocument) -> list[ParsedSection]:
    """Parse a Cornell USC-template statute page into ParsedSections."""
    section_div = loaded.soup.find("div", class_="section")
    if section_div is None:
        raise ValueError(f"No section root found in {loaded.filename!r}")
    out: list[ParsedSection] = []
    _walk_usc(section_div, [], out)
    return out


def _psection_level(p: Tag) -> int:
    """Read the nesting depth from a 'psection-N' class token."""
    for cls in p.get("class") or []:
        if cls.startswith("psection-"):
            return int(cls.split("-")[1])
    return 0


def parse_cfr(loaded: LoadedDocument) -> list[ParsedSection]:
    """Parse a Cornell CFR-template regulation page into ParsedSections.

    Unlike the USC template's nested divs, CFR paragraphs are a *flat*
    sequence of <p class="psection-N"> tags in document order, each
    carrying its own nesting depth N via the class name — and depth is
    not monotonic (a level can jump from 3 straight to 5, then back to 4
    in a different branch). Hierarchy is rebuilt with a small stack keyed
    by level number: a new label at level N replaces slot N and discards
    anything deeper (now stale) — the same technique used to turn a flat
    sequence of heading levels into a tree.
    """
    paras = loaded.soup.find_all(
        "p", class_=lambda c: bool(c) and c.startswith("psection-")
    )
    current: dict[int, str] = {}
    out: list[ParsedSection] = []

    for p in paras:
        level = _psection_level(p)
        enum_span = p.find("span", class_="enumxml")
        label = enum_span.get_text(strip=True).strip("()") if enum_span else "?"
        if enum_span:
            enum_span.extract()  # drop the label from the body text
        text = p.get_text(" ", strip=True)

        for stale_level in [k for k in current if k > level]:
            del current[stale_level]
        current[level] = label

        if text:
            path = [(str(lvl), current[lvl]) for lvl in sorted(current)]
            out.append(ParsedSection(path=path, text=text))

    return out


def _page_body_size(words: list[dict]) -> float:
    """Most common font size on the page — treated as the body-text
    baseline. Computed per page rather than hardcoded, since documents
    (and even pages within one document) can differ."""
    if not words:
        return 10.0
    sizes = [round(w["size"], 1) for w in words]
    return Counter(sizes).most_common(1)[0][0]


def _group_into_lines(words: list[dict]) -> list[list[dict]]:
    """Group words into lines by rounded vertical position, left to right."""
    lines: dict[int, list[dict]] = {}
    for w in words:
        key = round(w["top"])
        lines.setdefault(key, []).append(w)
    return [sorted(lines[k], key=lambda w: w["x0"]) for k in sorted(lines)]


_PAGE_FURNITURE_PATTERNS = [
    re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"\d{1,2}:\d{2}\s*-\s*\d{1,2}-[A-Za-z]{3}-\d{4}"),
    re.compile(r"AH XSL/XML", re.IGNORECASE),
    re.compile(r"Publication\s+\d+\s*\(\d{4}\)", re.IGNORECASE),
    re.compile(r"\b(Userid|Fileid|Schema|Leadpct)\s*:", re.IGNORECASE),
    re.compile(r"Draft Ok to Print", re.IGNORECASE),
]


def _is_page_furniture(line_text: str) -> bool:
    """IRS PDF-generator boilerplate (page stamps, internal watermarks) —
    not publication content at all, heading or body. Found by inspecting
    real false-positive headings against actual output, not anticipated
    upfront."""
    return any(p.search(line_text) for p in _PAGE_FURNITURE_PATTERNS)


def _has_real_word(line_text: str) -> bool:
    """At least one run of 3+ letters — screens out stray bullet glyphs
    and punctuation-only lines from ever being classified as headings."""
    return bool(re.search(r"[A-Za-z]{3,}", line_text))


def _is_heading_line(line: list[dict], body_size: float) -> bool:
    """Short line that's either notably larger than body text or
    predominantly bold-fontnamed — verified signal, not an ALL-CAPS guess.
    Must also contain a real word, screening out stray bullet glyphs."""
    if not line or len(line) > 12:
        return False
    line_text = " ".join(w["text"] for w in line)
    if not _has_real_word(line_text):
        return False
    avg_size = sum(w["size"] for w in line) / len(line)
    bold_count = sum(
        1 for w in line if "Bold" in w["fontname"] or "-Bd" in w["fontname"]
    )
    is_bold_line = bold_count >= len(line) * 0.6
    is_larger = avg_size > body_size + 1.0
    return is_larger or is_bold_line


def parse_pdf(loaded: LoadedDocument) -> list[ParsedSection]:
    """Best-effort heading-based chunking for PDFs with no structural
    markup. Meaningfully lower precision than the two HTML parsers by
    necessity — there is no numbered hierarchy to exploit here, only
    visual formatting. Content accumulates under the most recent heading
    across page breaks, since a section's text shouldn't fragment just
    because it happens to cross a page boundary.
    """
    out: list[ParsedSection] = []
    current_heading = "Introduction"
    current_parts: list[str] = []

    def flush() -> None:
        text = " ".join(current_parts).strip()
        if text:
            out.append(ParsedSection(path=[("heading", current_heading)], text=text))

    for page_words in loaded.pages or []:
        body_size = _page_body_size(page_words)
        for line in _group_into_lines(page_words):
            line_text = " ".join(w["text"] for w in line).strip()
            if not line_text or _is_page_furniture(line_text):
                continue
            if _is_heading_line(line, body_size):
                flush()
                current_heading = line_text
                current_parts = []
            else:
                current_parts.append(line_text)
    flush()
    return out


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_chunks(
    filename: str, meta: DocumentMeta, sections: list[ParsedSection]
) -> list[Chunk]:
    """Turn any parser's ParsedSections into final Chunk objects: builds
    the citation, injects the hierarchical header into the chunk text, and
    generates a stable chunk_id. The one place all three parsing strategies
    (USC, CFR, PDF) converge into the same output type.
    """
    chunks: list[Chunk] = []
    seen_slugs: dict[str, int] = {}
    for section in sections:
        is_pdf_style = len(section.path) == 1 and section.path[0][0] == "heading"

        if is_pdf_style:
            heading = section.path[0][1]
            citation = f"{meta.citation_prefix} — {heading}"
            header = f"{meta.citation_prefix} > {heading}"
        else:
            suffix = "".join(f"({label})" for _role, label in section.path)
            citation = f"{meta.citation_prefix}{suffix}"
            header = " > ".join(
                [meta.citation_prefix] + [f"({label})" for _role, label in section.path]
            )

        base_slug = _slugify(citation)
        seen_slugs[base_slug] = seen_slugs.get(base_slug, 0) + 1
        occurrence = seen_slugs[base_slug]
        # chunk_id is a storage key, not shown to users — disambiguated
        # with a counter since PDF headings (unlike numbered HTML
        # citations) aren't guaranteed unique within one document.
        chunk_id = base_slug if occurrence == 1 else f"{base_slug}-{occurrence}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=f"{header} — {section.text}",
                citation=citation,
                source_type=meta.source_type,
                authority_weight=meta.authority_weight,
                tax_year_start=meta.tax_year_start,
                tax_year_end=meta.tax_year_end,
                url=meta.url,
            )
        )
    return chunks


def build_all_chunks() -> list[Chunk]:
    """Run the right parser for every document in the registry (dispatched
    by source_type, reusing metadata.py rather than hardcoding filenames
    again) and return the full corpus as Chunks."""
    all_chunks: list[Chunk] = []
    for loaded in load_all_documents():
        if loaded.meta.source_type == "statute":
            sections = parse_usc(loaded)
        elif loaded.meta.source_type == "regulation":
            sections = parse_cfr(loaded)
        else:  # "instruction" | "publication"
            sections = parse_pdf(loaded)
        all_chunks.extend(build_chunks(loaded.filename, loaded.meta, sections))
    return all_chunks
