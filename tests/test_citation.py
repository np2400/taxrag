"""Tests for src/retrieval/citation.py.

Run with: python -m unittest tests.test_citation -v
No pytest dependency added -- unittest is stdlib, and this is the only
test in the repo so far (see ARCHITECTURE.md, tests/ is otherwise
planned but not yet implemented).
"""

import unittest

from src.retrieval.citation import exact_citation_lookup, parse_citation


class TestExactCitationLookupUnevenDepth(unittest.TestCase):
    """§1401(b)(2) has no chunk of its own, and its two branches aren't
    the same depth below it: §1401(b)(2)(B) is one level down, but
    §1401(b)(2)(A) also has no chunk of its own -- only its children
    §1401(b)(2)(A)(i)-(iii), two levels down, exist. A lookup that only
    checked one level below the cited provision would find (B) and
    silently drop the entire (A) branch."""

    def test_both_branches_returned(self) -> None:
        parsed = parse_citation("What does §1401(b)(2) require?")
        self.assertIsNotNone(parsed)

        chunks = exact_citation_lookup(parsed)
        chunk_ids = {c.chunk_id for c in chunks}

        self.assertEqual(
            chunk_ids,
            {
                "irc-1401-b-2-b",
                "irc-1401-b-2-a-i",
                "irc-1401-b-2-a-ii",
                "irc-1401-b-2-a-iii",
            },
        )


if __name__ == "__main__":
    unittest.main()
