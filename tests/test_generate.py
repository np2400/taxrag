"""Tests for src/generate.py's citation extraction.

Run with: python -m unittest tests.test_generate -v
Only _extract_citations (and its helper _mention_key) are exercised here --
they're pure functions, so no Groq API key or live retriever is needed,
matching evals/metrics.py's own testability principle.
"""

import unittest

from src.generate import _extract_citations
from src.types import Chunk, RetrievalResult


def _result(citation: str, source_type: str = "publication") -> RetrievalResult:
    chunk = Chunk(
        chunk_id=citation,
        text="irrelevant for these tests",
        citation=citation,
        source_type=source_type,
        authority_weight=0.4,
        tax_year_start=None,
        tax_year_end=None,
        url="https://example.gov",
    )
    return RetrievalResult(chunk=chunk, score=1.0, rank=1, retriever="dense")


class TestExtractCitations(unittest.TestCase):
    def test_only_treasury_regulation_citation(self) -> None:
        retrieved = [
            _result("Treas. Reg. §1.274-5(j)(2)", source_type="regulation")
        ]
        # Groq commonly emits U+2011 (non-breaking hyphen) in prose. The UI
        # renders it like a normal hyphen, which hid the production mismatch.
        text = "Adequate records are required (Treas. Reg. § 1.274‑5(j)(2))."

        self.assertEqual(
            _extract_citations(text, retrieved), ["Treas. Reg. §1.274-5(j)(2)"]
        )

    def test_publication_and_treasury_regulation_citations(self) -> None:
        retrieved = [
            _result("Pub. 463 — Standard Mileage Rate"),
            _result("Treas. Reg. §1.274-5(j)(2)", source_type="regulation"),
        ]
        text = (
            "The mileage method is described in Publication 463. "
            "Its recordkeeping rule appears in Treasury Regulation "
            "§ 1.274-5(j)(2)."
        )

        self.assertEqual(
            _extract_citations(text, retrieved),
            ["Pub. 463", "Treas. Reg. §1.274-5(j)(2)"],
        )

    def test_irc_treasury_regulation_and_publication_citations(self) -> None:
        retrieved = [
            _result("IRC §274(d)", source_type="statute"),
            _result("Treas. Reg. §1.274-5", source_type="regulation"),
            _result("Pub. 463 — Adequate Records"),
        ]
        text = (
            "The statute imposes substantiation requirements (§274(d)); "
            "the regulation supplies the rules (Treas. Reg. §1.274-5), "
            "and the IRS publication summarizes them (Pub 463)."
        )

        self.assertEqual(
            _extract_citations(text, retrieved),
            ["IRC §274(d)", "Treas. Reg. §1.274-5", "Pub. 463"],
        )

    def test_duplicate_citation_spellings_collapse(self) -> None:
        retrieved = [
            _result("Treas. Reg. §1.274-5(j)(2)", source_type="regulation")
        ]
        text = (
            "The rule is in Treas. Reg. §1.274-5(j)(2). "
            "See Treasury Regulation § 1.274-5(j)(2) again."
        )

        self.assertEqual(
            _extract_citations(text, retrieved), ["Treas. Reg. §1.274-5(j)(2)"]
        )

    def test_retrieved_but_uncited_sources_are_excluded(self) -> None:
        retrieved = [
            _result("Pub. 463 — Standard Mileage Rate"),
            _result("Treas. Reg. §1.274-5(j)(2)", source_type="regulation"),
            _result("IRC §274(d)", source_type="statute"),
        ]
        text = "The mileage method is described in Pub. 463."

        self.assertEqual(_extract_citations(text, retrieved), ["Pub. 463"])

    def test_irs_instruction_citations_are_preserved(self) -> None:
        retrieved = [
            _result("Schedule C Instructions — Line 9", source_type="instruction"),
            _result("Form 8829 Instructions — Purpose", source_type="instruction"),
        ]
        text = (
            "Report the expense as described in Instructions for Schedule C "
            "(Form 1040) and Instructions for Form 8829."
        )

        self.assertEqual(
            _extract_citations(text, retrieved),
            ["Schedule C Instructions", "Form 8829 Instructions"],
        )

    def test_generic_mention_does_not_fan_out_across_retrieved_chunks(self) -> None:
        """Bug C regression: one generic "(Pub. 463)" mention must not turn
        into one entry per retrieved Pub. 463 chunk."""
        retrieved = [
            _result("Pub. 463 — Introduction"),
            _result("Pub. 463 — Future Developments"),
            _result("Pub. 463 — What's New"),
            _result("Pub. 463 — Expenses"),
            _result("Pub. 463 — For use in preparing"),
        ]
        text = "The standard mileage rate applies here (Pub. 463)."

        self.assertEqual(_extract_citations(text, retrieved), ["Pub. 463"])

    def test_full_form_mention_normalizes_to_same_single_citation(self) -> None:
        """Whether the model writes the short or long form, the result is
        the same single normalized citation -- no fan-out either way."""
        retrieved = [
            _result("Pub. 463 — Introduction"),
            _result("Pub. 463 — Standard Mileage Rate"),
        ]
        text = "See (Pub. 463 — Standard Mileage Rate) for the current rate."

        self.assertEqual(_extract_citations(text, retrieved), ["Pub. 463"])

    def test_pinpoint_treas_reg_citation_keeps_full_precision(self) -> None:
        retrieved = [
            _result("Treas. Reg. §1.274-5(j)(2)", source_type="regulation"),
            _result("Treas. Reg. §1.274-5(a)", source_type="regulation"),
        ]
        text = "Substantiation requires adequate records (Treas. Reg. §1.274-5(j)(2))."

        self.assertEqual(
            _extract_citations(text, retrieved), ["Treas. Reg. §1.274-5(j)(2)"]
        )

    def test_unsupported_citation_is_excluded(self) -> None:
        """A citation the model mentions but that was never retrieved
        (hallucinated) must not reach Answer.citations."""
        retrieved = [_result("IRC §280A(c)(1)", source_type="statute")]
        text = "The QBI deduction may apply (IRC §199A)."

        self.assertEqual(_extract_citations(text, retrieved), [])

    def test_duplicate_mention_kept_once(self) -> None:
        retrieved = [_result("IRC §280A(c)(1)", source_type="statute")]
        text = (
            "The exclusive-use rule applies (IRC §280A(c)(1)). "
            "This is required by (IRC §280A(c)(1))."
        )

        self.assertEqual(_extract_citations(text, retrieved), ["IRC §280A(c)(1)"])

    def test_first_appearance_order_preserved(self) -> None:
        retrieved = [
            _result("IRC §280A(c)(1)", source_type="statute"),
            _result("Pub. 587 — Simplified Method"),
        ]
        # Mentioned in the reverse of retrieval order.
        text = "See (Pub. 587) for the simplified method, per (IRC §280A(c)(1))."

        self.assertEqual(
            _extract_citations(text, retrieved), ["Pub. 587", "IRC §280A(c)(1)"]
        )

    def test_bare_section_symbol_matches_irc_prefixed_chunk(self) -> None:
        """The model may drop the leading 'IRC' word even though the
        bracket label always includes it -- _mention_key must still match."""
        retrieved = [_result("IRC §280A(c)(1)", source_type="statute")]
        text = "This rule applies (§280A(c)(1))."

        self.assertEqual(_extract_citations(text, retrieved), ["IRC §280A(c)(1)"])


if __name__ == "__main__":
    unittest.main()
