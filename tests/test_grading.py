"""Tests for ``routeaudit.grading``: answer extraction and label-aware scoring.

Focus areas (task-2 item 2):
  * ``extract_answer`` for ``Answer:`` / ``\\boxed{}`` / ``####`` forms
  * extraction precedence and the conservative bare-last-line fallback
  * ``extract_confidence`` (decimal, percentage, out-of-range clipping)
  * numeric tolerance, choice-letter, code-normalisation and token-F1 comparison
  * unparseable responses must report ``format_ok=False``
"""

from __future__ import annotations

import unittest

from routeaudit.grading import Grader, extract_answer, extract_confidence, grade_text, token_f1
from routeaudit.types import Task, Verdict


def make_task(reference: str, answer_kind: str = "text") -> Task:
    return Task(
        task_id="t-1",
        prompt="irrelevant for grading",
        reference=reference,
        metadata={"answer_kind": answer_kind},
    )


# --------------------------------------------------------------------------- #
# extract_answer
# --------------------------------------------------------------------------- #
class ExtractAnswerTests(unittest.TestCase):
    def test_answer_line_form(self):
        self.assertEqual(extract_answer("Answer: 42"), "42")

    def test_final_answer_line_form(self):
        self.assertEqual(extract_answer("Final answer: Paris"), "Paris")

    def test_answer_line_with_fullwidth_colon(self):
        self.assertEqual(extract_answer("Answer： 42"), "42")

    def test_answer_line_inside_multi_line_response(self):
        text = "Reasoning: think it through.\nAnswer: 6\nConfidence: 0.9"
        self.assertEqual(extract_answer(text), "6")

    def test_boxed_form(self):
        self.assertEqual(extract_answer("So we get \\boxed{7}"), "7")

    def test_hashed_form(self):
        self.assertEqual(extract_answer("#### 12"), "12")

    def test_hashed_form_inside_prose(self):
        self.assertEqual(extract_answer("The total is #### 1089\nthat's it"), "1089")

    def test_answer_line_takes_precedence_over_boxed(self):
        self.assertEqual(extract_answer("Answer: 5\n\\boxed{9}"), "5")

    def test_boxed_takes_precedence_over_hashed(self):
        self.assertEqual(extract_answer("\\boxed{3}\n#### 9"), "3")

    def test_bare_last_line_fallback(self):
        self.assertEqual(extract_answer("some reasoning here\n42"), "42")

    def test_empty_input_returns_none(self):
        self.assertIsNone(extract_answer(""))
        self.assertIsNone(extract_answer("   \n\t  "))

    def test_trailing_prose_is_not_treated_as_an_answer(self):
        # the format gate depends on this: a trailing-off response is unparseable
        prose = "The reasoning is mostly there but I could not settle on a single final value."
        self.assertIsNone(extract_answer(prose))

    def test_long_fallback_line_is_rejected(self):
        self.assertIsNone(extract_answer("one two three four five six seven eight nine"))

    def test_sentence_terminated_fallback_line_is_rejected(self):
        self.assertIsNone(extract_answer("I could not decide!"))
        self.assertIsNone(extract_answer("Is it 42?"))

    def test_backticks_and_trailing_period_are_stripped(self):
        self.assertEqual(extract_answer("Answer: `42`"), "42")
        self.assertEqual(extract_answer("Answer: 42."), "42")

    def test_answer_kind_number_survives_extraction(self):
        self.assertEqual(extract_answer("Answer: 3.14159"), "3.14159")


# --------------------------------------------------------------------------- #
# extract_confidence
# --------------------------------------------------------------------------- #
class ExtractConfidenceTests(unittest.TestCase):
    def test_decimal_confidence(self):
        self.assertAlmostEqual(extract_confidence("Answer: 1\nConfidence: 0.85"), 0.85)

    def test_percentage_confidence(self):
        self.assertAlmostEqual(extract_confidence("Answer: 1\nConfidence: 85%"), 0.85)

    def test_missing_confidence_returns_none(self):
        self.assertIsNone(extract_confidence("Answer: 42"))
        self.assertIsNone(extract_confidence(""))

    def test_out_of_range_values_are_clipped(self):
        # a model claiming >100% or a bare integer is clamped into [0, 1]
        self.assertEqual(extract_confidence("Confidence: 250%"), 1.0)
        self.assertEqual(extract_confidence("Confidence: 85"), 1.0)

    def test_confidence_line_is_case_insensitive(self):
        self.assertAlmostEqual(extract_confidence("confidence: 0.4"), 0.4)
        self.assertAlmostEqual(extract_confidence("FINAL CONFIDENCE: 0.4"), 0.4)


# --------------------------------------------------------------------------- #
# Numeric comparison
# --------------------------------------------------------------------------- #
class NumericComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grader = Grader()
        self.task = make_task("100", "number")

    def test_exact_match(self):
        self.assertEqual(self.grader.grade(self.task, "Answer: 100").verdict, Verdict.CORRECT)

    def test_match_within_relative_tolerance(self):
        # default numeric_tolerance is 1e-3 relative
        self.assertEqual(self.grader.grade(self.task, "Answer: 100.05").verdict, Verdict.CORRECT)

    def test_match_just_outside_tolerance(self):
        self.assertEqual(self.grader.grade(self.task, "Answer: 101").verdict, Verdict.INCORRECT)

    def test_tolerance_is_configurable(self):
        strict = Grader(numeric_tolerance=1e-6)
        self.assertEqual(strict.grade(self.task, "Answer: 100.05").verdict, Verdict.INCORRECT)
        loose = Grader(numeric_tolerance=0.5)
        self.assertEqual(loose.grade(self.task, "Answer: 140").verdict, Verdict.CORRECT)

    def test_commas_in_numbers_are_tolerated(self):
        task = make_task("1234", "number")
        self.assertEqual(self.grader.grade(task, "Answer: 1,234").verdict, Verdict.CORRECT)

    def test_unparseable_number_is_incorrect_not_correct(self):
        task = make_task("42", "number")
        result = self.grader.grade(task, "Answer: forty-two")
        self.assertEqual(result.verdict, Verdict.INCORRECT)
        self.assertTrue(result.format_ok)  # an answer *was* extracted

    def test_negative_numbers(self):
        task = make_task("-7", "number")
        self.assertEqual(self.grader.grade(task, "Answer: -7").verdict, Verdict.CORRECT)
        self.assertEqual(self.grader.grade(task, "Answer: 7").verdict, Verdict.INCORRECT)


# --------------------------------------------------------------------------- #
# Choice comparison
# --------------------------------------------------------------------------- #
class ChoiceComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grader = Grader()
        self.task = make_task("B", "choice")

    def test_same_letter(self):
        self.assertEqual(self.grader.grade(self.task, "Answer: B").verdict, Verdict.CORRECT)

    def test_letter_is_case_insensitive(self):
        self.assertEqual(self.grader.grade(self.task, "Answer: b").verdict, Verdict.CORRECT)

    def test_different_letter_is_incorrect(self):
        self.assertEqual(self.grader.grade(self.task, "Answer: C").verdict, Verdict.INCORRECT)

    def test_prose_answer_does_not_match_a_letter(self):
        # conservative: the grader compares letters, not narrative
        self.assertEqual(
            self.grader.grade(self.task, "Answer: The answer is B").verdict, Verdict.INCORRECT
        )


# --------------------------------------------------------------------------- #
# Code comparison
# --------------------------------------------------------------------------- #
class CodeComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grader = Grader()
        self.task = make_task("return a + b", "code")

    def test_identical_code(self):
        self.assertEqual(self.grader.grade(self.task, "Answer: return a + b").verdict, Verdict.CORRECT)

    def test_whitespace_is_normalised(self):
        self.assertEqual(
            self.grader.grade(self.task, "Answer: return   a  +  b").verdict, Verdict.CORRECT
        )

    def test_comments_are_stripped(self):
        self.assertEqual(
            self.grader.grade(self.task, "Answer: return a + b  # sum them").verdict,
            Verdict.CORRECT,
        )

    def test_different_body_is_incorrect(self):
        self.assertEqual(
            self.grader.grade(self.task, "Answer: return a - b").verdict, Verdict.INCORRECT
        )

    def test_normalisation_can_be_disabled(self):
        strict = Grader(code_normalise=False)
        self.assertEqual(
            strict.grade(self.task, "Answer: return   a + b").verdict, Verdict.INCORRECT
        )
        self.assertEqual(
            strict.grade(self.task, "Answer: return a + b").verdict, Verdict.CORRECT
        )


# --------------------------------------------------------------------------- #
# Free-text comparison / token F1
# --------------------------------------------------------------------------- #
class TextComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grader = Grader()
        self.task = make_task("Paris is the capital of France")

    def test_exact_text_match(self):
        self.assertEqual(
            self.grader.grade(self.task, "Answer: Paris is the capital of France").verdict,
            Verdict.CORRECT,
        )

    def test_paraphrase_passes_via_token_f1(self):
        self.assertEqual(
            self.grader.grade(self.task, "Answer: Paris is capital").verdict, Verdict.CORRECT
        )

    def test_unrelated_answer_fails(self):
        self.assertEqual(self.grader.grade(self.task, "Answer: Berlin").verdict, Verdict.INCORRECT)

    def test_text_threshold_is_configurable(self):
        strict = Grader(text_threshold=0.95)
        self.assertEqual(strict.grade(self.task, "Answer: Paris is capital").verdict, Verdict.INCORRECT)


class TokenF1Tests(unittest.TestCase):
    def test_identical_token_sets_score_one(self):
        self.assertEqual(token_f1("a b c", "a b c"), 1.0)

    def test_disjoint_token_sets_score_zero(self):
        self.assertEqual(token_f1("a b", "c d"), 0.0)

    def test_empty_input_scores_zero(self):
        self.assertEqual(token_f1("", "a"), 0.0)
        self.assertEqual(token_f1("a", ""), 0.0)

    def test_partial_overlap_is_between_zero_and_one(self):
        score = token_f1("a b c d", "a b")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_score_is_symmetric_for_equal_length_inputs(self):
        self.assertAlmostEqual(token_f1("a b c", "a b d"), token_f1("a b d", "a b c"))

    def test_duplicate_tokens_are_counted_once_per_occurrence(self):
        # multiset overlap, not set overlap: "a a" vs "a" overlaps once
        self.assertAlmostEqual(token_f1("a a", "a"), 2 * (0.5 * 1.0) / 1.5)


# --------------------------------------------------------------------------- #
# Format failures
# --------------------------------------------------------------------------- #
class FormatFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grader = Grader()
        self.task = make_task("Paris")

    def test_unparseable_response_reports_format_failure(self):
        result = self.grader.grade(self.task, "I could not decide!")
        self.assertFalse(result.format_ok)
        self.assertEqual(result.verdict, Verdict.INCORRECT)
        self.assertIsNone(result.extracted_answer)

    def test_long_prose_response_reports_format_failure(self):
        prose = (
            "The reasoning is mostly there but I could not settle on a single "
            "final value for this problem"
        )
        result = self.grader.grade(self.task, prose)
        self.assertFalse(result.format_ok)
        self.assertIsNone(result.extracted_answer)

    def test_period_terminated_prose_is_rejected(self):
        """Regression test for the dead ``"."`` terminator branch.

        The fallback must reject a sentence-terminated last line.  It used to
        accept period-terminated prose because ``_clean`` strips trailing ``"."``
        with ``rstrip(".")`` *before* the terminator check ran, making the
        ``"."`` branch of ``endswith((".", "!", "?", "…"))`` dead code.  The
        check now runs on the raw line, before cleaning.
        """
        short_period_prose = "The reasoning trails off and never settles."
        self.assertIsNone(extract_answer(short_period_prose))
        # every sentence terminator behaves the same way
        self.assertIsNone(extract_answer("The reasoning trails off and never settles!"))
        self.assertIsNone(extract_answer("The reasoning trails off and never settles?"))
        self.assertIsNone(extract_answer("The reasoning trails off and never settles…"))

    def test_terminator_check_does_not_break_explicit_answers(self):
        """An explicit ``Answer:`` line is unaffected by the terminator rule."""
        # The explicit-answer path strips the period and still returns the value.
        self.assertEqual(extract_answer("Answer: Paris.\nConfidence: 0.9"), "Paris")
        self.assertEqual(extract_answer("Answer: 42\nConfidence: 0.9"), "42")

    def test_empty_response_reports_format_failure(self):
        result = self.grader.grade(self.task, "")
        self.assertFalse(result.format_ok)
        self.assertIsNone(result.extracted_answer)

    def test_parseable_but_wrong_answer_is_a_graded_miss(self):
        # the distinction the router relies on: format OK, answer wrong
        result = self.grader.grade(self.task, "Answer: Berlin")
        self.assertTrue(result.format_ok)
        self.assertEqual(result.verdict, Verdict.INCORRECT)

    def test_grade_text_helper_matches_grader(self):
        self.assertEqual(
            grade_text(self.task, "Answer: Paris").to_dict(),
            self.grader.grade(self.task, "Answer: Paris").to_dict(),
        )

    def test_grade_text_accepts_custom_grader(self):
        result = grade_text(self.task, "Answer: Berlin", Grader(text_threshold=0.0))
        # a zero threshold makes any non-empty comparison pass
        self.assertEqual(result.verdict, Verdict.CORRECT)


if __name__ == "__main__":
    unittest.main()
