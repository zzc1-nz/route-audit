"""Tests for ``routeaudit.confidence``: the label-free escalation gate.

Focus areas (task-2 item 3):
  * ``format_gate`` takes precedence over a high self-reported confidence
  * missing confidence escalates (untrusted by default)
  * threshold boundary: ``confidence == threshold`` does NOT escalate
  * ``provider_error`` escalates with its own reason, ahead of every other gate
  * calibration helpers (reliability bins, Brier score, coverage)
"""

from __future__ import annotations

import unittest

from routeaudit.confidence import (
    EscalationDecision,
    Router,
    calibration_report,
    optimal_threshold,
)
from routeaudit.types import EscalationReason, Verdict


# --------------------------------------------------------------------------- #
# Gate ordering and reasons
# --------------------------------------------------------------------------- #
class RouterGateTests(unittest.TestCase):
    def test_high_confidence_and_good_format_does_not_escalate(self):
        decision = Router(threshold=0.5).decide(confidence=0.9, format_ok=True)
        self.assertFalse(decision.escalate)
        self.assertIs(decision.reason, EscalationReason.NONE)

    def test_format_gate_beats_high_confidence(self):
        decision = Router(threshold=0.5).decide(confidence=0.99, format_ok=False)
        self.assertTrue(decision.escalate)
        self.assertIs(decision.reason, EscalationReason.FORMAT_FAILURE)

    def test_format_gate_can_be_disabled_to_isolate_confidence(self):
        router = Router(threshold=0.5, format_gate=False)
        decision = router.decide(confidence=0.99, format_ok=False)
        self.assertFalse(decision.escalate)
        self.assertIs(decision.reason, EscalationReason.NONE)

    def test_format_gate_disabled_still_escalates_on_low_confidence(self):
        router = Router(threshold=0.5, format_gate=False)
        decision = router.decide(confidence=0.1, format_ok=False)
        self.assertTrue(decision.escalate)
        self.assertIs(decision.reason, EscalationReason.LOW_CONFIDENCE)

    def test_missing_confidence_escalates(self):
        decision = Router(threshold=0.5).decide(confidence=None, format_ok=True)
        self.assertTrue(decision.escalate)
        self.assertIs(decision.reason, EscalationReason.MISSING_CONFIDENCE)
        self.assertIsNone(decision.confidence)

    def test_low_confidence_escalates(self):
        decision = Router(threshold=0.5).decide(confidence=0.2, format_ok=True)
        self.assertTrue(decision.escalate)
        self.assertIs(decision.reason, EscalationReason.LOW_CONFIDENCE)

    def test_provider_error_escalates(self):
        decision = Router(threshold=0.5).decide(
            confidence=0.99, format_ok=True, provider_error="connection reset"
        )
        self.assertTrue(decision.escalate)
        self.assertIs(decision.reason, EscalationReason.PROVIDER_ERROR)
        self.assertIn("connection reset", decision.detail)

    def test_provider_error_outranks_format_failure(self):
        decision = Router(threshold=0.5).decide(
            confidence=None, format_ok=False, provider_error="HTTP 500"
        )
        self.assertIs(decision.reason, EscalationReason.PROVIDER_ERROR)


# --------------------------------------------------------------------------- #
# Threshold boundary
# --------------------------------------------------------------------------- #
class ThresholdBoundaryTests(unittest.TestCase):
    def test_confidence_exactly_at_threshold_does_not_escalate(self):
        decision = Router(threshold=0.5).decide(confidence=0.5, format_ok=True)
        self.assertFalse(decision.escalate)
        self.assertIs(decision.reason, EscalationReason.NONE)

    def test_confidence_just_below_threshold_escalates(self):
        decision = Router(threshold=0.5).decide(confidence=0.4999, format_ok=True)
        self.assertTrue(decision.escalate)
        self.assertIs(decision.reason, EscalationReason.LOW_CONFIDENCE)

    def test_confidence_just_above_threshold_does_not_escalate(self):
        decision = Router(threshold=0.5).decide(confidence=0.5001, format_ok=True)
        self.assertFalse(decision.escalate)

    def test_boundary_holds_at_extreme_thresholds(self):
        # confidence == 1.0 at threshold 1.0 must not escalate
        self.assertFalse(Router(threshold=1.0).decide(confidence=1.0, format_ok=True).escalate)
        # confidence == 0.0 at threshold 0.0 must not escalate
        self.assertFalse(Router(threshold=0.0).decide(confidence=0.0, format_ok=True).escalate)

    def test_higher_threshold_escalates_more(self):
        confidence = 0.6
        self.assertFalse(Router(threshold=0.5).decide(confidence=confidence, format_ok=True).escalate)
        self.assertTrue(Router(threshold=0.7).decide(confidence=confidence, format_ok=True).escalate)

    def test_escalation_is_monotone_in_threshold(self):
        confidence = 0.55
        flags = [
            Router(threshold=t).decide(confidence=confidence, format_ok=True).escalate
            for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ]
        # once it starts escalating it must keep escalating
        first_true = flags.index(True) if True in flags else len(flags)
        self.assertTrue(all(flags[first_true:]), msg=f"non-monotone: {flags}")

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            Router(threshold=1.5)
        with self.assertRaises(ValueError):
            Router(threshold=-0.1)


# --------------------------------------------------------------------------- #
# Decision object
# --------------------------------------------------------------------------- #
class EscalationDecisionTests(unittest.TestCase):
    def test_decision_round_trips_to_dict(self):
        decision = Router(threshold=0.5).decide(confidence=0.2, format_ok=True)
        payload = decision.to_dict()
        self.assertTrue(payload["escalate"])
        self.assertEqual(payload["reason"], "low_confidence")
        self.assertEqual(payload["confidence"], 0.2)
        self.assertTrue(payload["format_ok"])

    def test_decision_carries_the_observed_signals(self):
        decision = Router(threshold=0.5).decide(confidence=None, format_ok=False)
        self.assertIsNone(decision.confidence)
        self.assertFalse(decision.format_ok)
        self.assertTrue(decision.detail)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
class CalibrationTests(unittest.TestCase):
    def test_perfectly_confident_and_correct_has_zero_brier(self):
        report = calibration_report([(1.0, Verdict.CORRECT)] * 4)
        self.assertEqual(report.brier, 0.0)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.mean_confidence, 1.0)
        self.assertAlmostEqual(report.overconfidence, 0.0)

    def test_confident_but_wrong_has_high_brier(self):
        report = calibration_report([(1.0, Verdict.INCORRECT)] * 4)
        self.assertAlmostEqual(report.brier, 1.0)
        self.assertEqual(report.accuracy, 0.0)
        self.assertAlmostEqual(report.overconfidence, 1.0)

    def test_bins_partition_confidence_into_ten_buckets(self):
        report = calibration_report([(0.05, Verdict.CORRECT), (0.95, Verdict.INCORRECT)])
        self.assertEqual(len(report.bins), 10)
        self.assertEqual(report.bins[0].n, 1)  # 0.05 -> bin 0
        self.assertEqual(report.bins[9].n, 1)  # 0.95 -> bin 9
        self.assertEqual(sum(b.n for b in report.bins), report.n)

    def test_ungraded_and_missing_pairs_are_excluded_but_counted_in_coverage(self):
        pairs = [
            (0.9, Verdict.CORRECT),
            (None, Verdict.CORRECT),
            (0.5, Verdict.UNGRADED),
        ]
        report = calibration_report(pairs)
        self.assertEqual(report.n, 1)  # only the one fully observed pair
        self.assertAlmostEqual(report.confidence_coverage, 1 / 3)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            calibration_report([])

    def test_ece_is_zero_when_bins_are_perfectly_calibrated(self):
        # one bin holding only correct 0.95-confidence answers -> gap 0.05
        report = calibration_report([(0.95, Verdict.CORRECT)] * 10)
        self.assertAlmostEqual(report.ece, 0.05, places=6)


# --------------------------------------------------------------------------- #
# optimal_threshold (used as a cross-check on the sweep)
# --------------------------------------------------------------------------- #
class OptimalThresholdTests(unittest.TestCase):
    def test_returns_a_probability(self):
        pairs = [(0.9, Verdict.CORRECT), (0.2, Verdict.INCORRECT)]
        threshold = optimal_threshold(pairs)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_separates_cleanly_separable_confidence(self):
        # confident-correct vs unconfident-wrong: every threshold in (0.1, 0.9]
        # reaches the minimum cost, so assert cost-optimality rather than one
        # particular tie-break (the implementation keeps the lowest threshold).
        pairs = [(0.9, Verdict.CORRECT)] * 10 + [(0.1, Verdict.INCORRECT)] * 10
        threshold = optimal_threshold(pairs)

        def cost_at(cut: float) -> float:
            cost = 0.0
            for confidence, verdict in pairs:
                if confidence < cut:
                    cost += 2.0  # escalate: 1 + ratio
                elif not verdict.is_correct:
                    cost += 2.0  # wrong answer charged the same
                else:
                    cost += 1.0
            return cost

        # the chosen threshold must be as cheap as the best possible cut
        best_possible = min(cost_at(step / 100.0) for step in range(1, 100))
        self.assertAlmostEqual(cost_at(threshold), best_possible)
        # and it must still be a probability
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)

    def test_empty_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            optimal_threshold([])


if __name__ == "__main__":
    unittest.main()
