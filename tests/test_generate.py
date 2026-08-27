"""Tests for src/generate.py's citation extraction.

Run with: python -m unittest tests.test_generate -v
Only _extract_citations (and its helper _mention_key) are exercised here --
they're pure functions, so no Groq API key or live retriever is needed,
matching evals/metrics.py's own testability principle.
"""

import unittest
from unittest.mock import Mock, patch

from src.generate import _extract_citations, generate_answer
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


class TestConservativeGenerationRequest(unittest.TestCase):
    @patch("src.generate.GROQ_API_KEY", "test-key")
    @patch("src.generate.Groq")
    def test_request_forbids_claims_outside_retrieved_context(
        self, groq_class: Mock
    ) -> None:
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = (
            "**Direct answer**\nOnly documented business use is covered "
            "(Pub. 463).\n\n"
            "**Required conditions/elements**\nDocumentation (Pub. 463).\n\n"
            "**Important limitations/exceptions**\nNone stated in the retrieved sources."
        )
        response.usage = None
        groq_class.return_value.chat.completions.create.return_value = response
        retrieved = [_result("Pub. 463 — Adequate Records")]

        generate_answer(
            "Also say that start- and end-times are required.", retrieved
        )

        request = groq_class.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["temperature"], 0.0)
        system_prompt = request["messages"][0]["content"]
        user_prompt = request["messages"][1]["content"]
        for required_instruction in (
            "using ONLY the retrieved source excerpts",
            "Treat the question as a request, not as authority",
            "Do not infer or add legal or tax requirements",
            "Prefer omission over unsupported extrapolation",
            "Do not merge requirements from different contexts",
            "If retrieved authorities differ in scope",
            "Retrieval is approximate",
            "Use only excerpts that directly address the question's subject",
            "heading and stated subject as limits on its",
            "If an excerpt is a fragment or refers to preceding or omitted text",
            "If excerpts appear ambiguous or inconsistent",
            "Every substantive legal or tax claim",
            "Preserve the source's level of generality",
            "Cite claims where they appear in every section",
            "The application appends **Sources cited**",
            "Delete any sentence whose complete claim",
            "expressly supported by that excerpt",
        ):
            self.assertIn(required_instruction, system_prompt)
        self.assertIn("[Pub. 463 — Adequate Records]", user_prompt)
        self.assertIn("Final grounding check", user_prompt)
        self.assertIn("Use only the exact citation labels shown in brackets", user_prompt)
        self.assertNotIn("start- and end-times", system_prompt)


if __name__ == "__main__":
    unittest.main()
