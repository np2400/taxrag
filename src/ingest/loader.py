"""Reads each raw corpus file and extracts it into a normalized form the
chunker can walk — format detection and raw extraction only. Any structural
intelligence (finding section markers, building hierarchy) belongs in
chunker.py, not here.
"""

from dataclasses import dataclass

import pdfplumber
from bs4 import BeautifulSoup

from config.settings import DATA_RAW_DIR
from src.ingest.metadata import DOCUMENT_REGISTRY, DocumentMeta


@dataclass
class LoadedDocument:
    """Handoff shape between loader.py and chunker.py.

    Not one of ARCHITECTURE.md's three frozen contracts — a private,
    internal detail scoped to the ingest pipeline only.
    """

    filename: str
    meta: DocumentMeta
    format: str  # "html" | "pdf"
    soup: BeautifulSoup | None = None  # populated when format == "html"
    pages: list[list[dict]] | None = None
    # populated when format == "pdf": each page's words (text/size/fontname/
    # position), not plain text — chunker.py needs font metadata to detect
    # headings, which a plain string would have already thrown away.


def load_document(filename: str) -> LoadedDocument:
    """Read one file from data/raw/ and extract it into a LoadedDocument."""
    meta = DOCUMENT_REGISTRY[filename]
    path = DATA_RAW_DIR / filename

    if filename.endswith(".html"):
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        return LoadedDocument(filename=filename, meta=meta, format="html", soup=soup)

    if filename.endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            pages = [
                page.extract_words(extra_attrs=["size", "fontname"])
                for page in pdf.pages
            ]
        return LoadedDocument(filename=filename, meta=meta, format="pdf", pages=pages)

    raise ValueError(f"Unsupported file type for {filename!r}")


def load_all_documents() -> list[LoadedDocument]:
    """Load every document in the registry, in registry order."""
    return [load_document(filename) for filename in DOCUMENT_REGISTRY]
