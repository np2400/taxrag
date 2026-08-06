"""Sparse retrieval: BM25 over the same chunk corpus dense.py embeds.

A class, like DenseRetriever, because building the BM25 index (tokenizing
every chunk) is a one-time cost that should happen once per process, not
once per query. Unlike Chroma, rank_bm25 has no persistence layer of its
own -- the corpus is small enough (~1,500 chunks) that rebuilding the
index from data/processed/chunks.json at construction time is fast, so
there's nothing to persist to disk in the first place.
"""

import json
import re

from rank_bm25 import BM25Okapi

from config.settings import CHUNKS_PATH
from src.types import Chunk, RetrievalResult

_TOKEN_RE = re.compile(r"\w+")

# Without this, BM25's IDF weighting backfires on this corpus: the golden
# set's questions are natural-language ("What does...", "How would...")
# but the corpus is formal statutory/publication prose that almost never
# phrases things as questions. That makes interrogative words like "does"
# or "how" *rare in this corpus* even though they're meaningless -- BM25
# has no way to tell "rare because distinctive" (like "280a") apart from
# "rare because it's the wrong register" (like "does"), and gave "does" an
# IDF of 3.17 against "280a"'s 3.40, nearly on par. Verified concretely:
# without this filter, the query for IRC §280A(c)(1) ranked the actual
# §280A(c)(1) chunk 50th out of 1,531, behind chunks that shared "does"/
# "rule"/"office" but not the one term that actually mattered.
_STOPWORDS = frozenset(
    """
    a an the and or but if is are was were be been being do does did
    what which who whom this that these those to of in on at for from
    with as by it its into about how when where why not no can could
    should would will shall may might must have has had i you he she
    we they them his her their your my our
    """.split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric characters, drop stopwords.
    Otherwise deliberately simple -- no stemming; BM25's term-frequency
    scoring does the rest of the useful work once stopword noise is gone."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class SparseRetriever:
    def __init__(self) -> None:
        raw_chunks = json.loads(CHUNKS_PATH.read_text())
        self._chunks = [Chunk(**c) for c in raw_chunks]
        tokenized_corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
        self, query: str, k: int, tax_year: int | None = None
    ) -> list[RetrievalResult]:
        # tax_year isn't filtered on yet -- matching dense.py's current
        # behavior (Phase 1 accepts the parameter but doesn't apply it
        # either). Adding filtering to only one retriever would make the
        # two sides of the Phase 4 ablation not actually comparable.
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        return [
            RetrievalResult(
                chunk=self._chunks[idx],
                score=float(scores[idx]),
                rank=rank,
                retriever="sparse",
            )
            for rank, idx in enumerate(ranked_indices, start=1)
        ]
