"""Tests for ``routeaudit.analysis``: routing-error cost analysis.

Focus areas (task-2 item 6):
  * ``over_escalation`` tasks satisfy ``cheap_correct and escalated``
  * ``over_downgrade`` tasks satisfy ``cheap_incorrect and escalated``
  * ``wasted_token_share`` stays inside [0, 1]
  * ``regret_cost_usd >= 0`` (the oracle is a lower bound on cost)
  * ``sensitivity.points`` is sorted by threshold
"""

from __future__ import annotations

import unittest

from routeaudit.analysis import (
    analyse,
    compare,
    domain_breakdown,
    oracle_plan,
    regret_analysis,
    routing_error_analysis,
    threshold_sensitivity,
)
from routeaudit.config import AuditConfig
from routeaudit.engine import run_audit
from routeaudit.types import AuditRun, Strategy, Verdict

# Small, fast audit whose threshold ladder actually produces escalations.
N_TASKS = 12
THRESHOLDS = (0.9, 0.7, 0.5)
HEADLINE = 0.9


def build_audit(**overrides) -> AuditRun:
    config = dict(n_tasks=N_TASKS, thresholds=THRESHOLDS, seed=20260101)
    config.update(overrides)
    return run_audit(AuditConfig(**config))


_AUDIT: AuditRun | None = None


def get_audit() -> AuditRun:
    global _AUDIT
    if _AUDIT is None:
        _AUDIT = build_audit()
    return _AUDIT


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = get_audit()

    def test_compare_selects_a_b_and_the_headline_cascade(self):
        comparison = compare(self.audit)
        self.assertEqual(comparison.cheap_key, "A")
        self.assertEqual(comparison.premium_key, "B")
        self.assertEqual(comparison.cascade_key, "C@0.5")

    def test_compare_honours_an_explicit_threshold(self):
        self.assertEqual(compare(self.audit, threshold=0.9).cascade_key, "C@0.9")

    def test_ordered_lists_a_then_b_then_cascade_ascending(self):
        ordered = compare(self.audit).ordered()
        keys = [s.strategy.value for s in ordered]
        self.assertEqual(keys[:2], ["A", "B"])
        cascade_thresholds = [s.threshold for s in ordered if s.strategy is Strategy.CASCADE]
        self.assertEqual(cascade_thresholds, sorted(cascade_thresholds))

    def test_premium_accuracy_is_at_least_cheap_accuracy(self):
        """Checked in aggregate: at ``n_tasks=12`` the binomial noise of a
        single seed can put premium below cheap (~1 in 40 seeds), so the
        invariant is asserted over a larger sample."""
        audit = run_audit(AuditConfig(n_tasks=200, thresholds=(0.5,), seed=20260101))
        comparison = compare(audit)
        self.assertGreaterEqual(comparison.premium.accuracy, comparison.cheap.accuracy)

    def test_cascade_savings_against_premium_is_a_fraction(self):
        comparison = compare(self.audit)
        self.assertGreaterEqual(comparison.cascade_savings_vs_premium, 0.0)
        self.assertLessEqual(comparison.cascade_savings_vs_premium, 1.0)

    def test_compare_rejects_an_audit_without_a_cascade(self):
        audit = run_audit(
            AuditConfig(n_tasks=4, thresholds=(0.5,), include_strategies=("A", "B"))
        )
        with self.assertRaises(ValueError):
            compare(audit)

    def test_compare_rejects_an_unknown_threshold(self):
        with self.assertRaises(ValueError):
            compare(self.audit, threshold=0.42)


# --------------------------------------------------------------------------- #
# Routing-error attribution
# --------------------------------------------------------------------------- #
class RoutingErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = get_audit()
        cls.analysis = routing_error_analysis(cls.audit, threshold=HEADLINE)
        cls.cascade = cls.audit.by_task(Strategy.CASCADE, HEADLINE)
        cls.cheap = cls.audit.by_task(Strategy.CHEAP)

    def test_over_escalation_tasks_were_escalated_but_cheap_was_correct(self):
        for task_id in self.analysis.wasted_escalation_tasks:
            run = self.cascade[task_id]
            self.assertTrue(run.escalated, msg=task_id)
            self.assertIs(run.cheap_verdict, Verdict.CORRECT, msg=task_id)
            self.assertIs(self.cheap[task_id].final_verdict, Verdict.CORRECT, msg=task_id)

    def test_over_downgrade_tasks_were_escalated_because_cheap_was_wrong(self):
        for task_id in self.analysis.wasted_downgrade_tasks:
            run = self.cascade[task_id]
            self.assertTrue(run.escalated, msg=task_id)
            self.assertIs(run.cheap_verdict, Verdict.INCORRECT, msg=task_id)
            self.assertIs(self.cheap[task_id].final_verdict, Verdict.INCORRECT, msg=task_id)

    def test_missed_escalation_tasks_were_not_escalated_but_cheap_was_wrong(self):
        for task_id in self.analysis.missed_escalation_tasks:
            run = self.cascade[task_id]
            self.assertFalse(run.escalated, msg=task_id)
            self.assertIs(run.cheap_verdict, Verdict.INCORRECT, msg=task_id)

    def test_bucket_task_counts_match_the_task_id_lists(self):
        self.assertEqual(
            self.analysis.over_escalation.n_tasks,
            len(self.analysis.wasted_escalation_tasks),
        )
        self.assertEqual(
            self.analysis.over_downgrade.n_tasks,
            len(self.analysis.wasted_downgrade_tasks),
        )
        self.assertEqual(
            self.analysis.missed_escalation.n_tasks,
            len(self.analysis.missed_escalation_tasks),
        )

    def test_over_escalation_tokens_are_the_escalation_tokens_of_those_tasks(self):
        expected = sum(
            self.cascade[t].tokens.escalated_tokens
            for t in self.analysis.wasted_escalation_tasks
        )
        self.assertEqual(self.analysis.over_escalation.tokens, expected)

    def test_over_downgrade_tokens_are_the_cheap_tokens_of_those_tasks(self):
        expected = sum(
            self.cascade[t].tokens.primary_tokens
            for t in self.analysis.wasted_downgrade_tasks
        )
        self.assertEqual(self.analysis.over_downgrade.tokens, expected)

    def test_n_escalated_matches_the_runs(self):
        self.assertEqual(
            self.analysis.n_escalated,
            sum(1 for r in self.cascade.values() if r.escalated),
        )

    def test_escalation_precision_uses_the_over_escalation_count(self):
        expected = (
            (self.analysis.n_escalated - self.analysis.over_escalation.n_tasks)
            / self.analysis.n_escalated
        )
        self.assertAlmostEqual(self.analysis.escalation_precision, expected)

    def test_wasted_token_share_is_within_the_unit_interval(self):
        self.assertGreaterEqual(self.analysis.wasted_token_share, 0.0)
        self.assertLessEqual(self.analysis.wasted_token_share, 1.0)

    def test_wasted_cost_share_is_within_the_unit_interval(self):
        self.assertGreaterEqual(self.analysis.wasted_cost_share, 0.0)
        self.assertLessEqual(self.analysis.wasted_cost_share, 1.0)

    def test_wasted_cost_is_the_sum_of_both_directions(self):
        self.assertAlmostEqual(
            self.analysis.wasted_cost_usd,
            self.analysis.wasted_escalation_cost_usd
            + self.analysis.wasted_downgrade_cost_usd,
        )

    def test_useful_tokens_is_the_remainder(self):
        self.assertEqual(
            self.analysis.useful_tokens,
            self.analysis.total_tokens
            - self.analysis.wasted_escalation_tokens
            - self.analysis.wasted_downgrade_tokens,
        )

    def test_wasted_share_matches_the_token_definition(self):
        expected = (
            self.analysis.wasted_escalation_tokens + self.analysis.wasted_downgrade_tokens
        ) / self.analysis.total_tokens
        self.assertAlmostEqual(self.analysis.wasted_token_share, expected)

    def test_escalation_reasons_are_consistent_with_the_runs(self):
        self.assertEqual(
            sum(self.analysis.n_escalation_reasons.values()), self.analysis.n_escalated
        )

    def test_rates_stay_in_the_unit_interval(self):
        for rate in (
            self.analysis.over_escalation_rate,
            self.analysis.over_downgrade_rate,
            self.analysis.missed_escalation_rate,
            self.analysis.escalation_precision,
        ):
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)

    def test_higher_threshold_escalates_at_least_as_often(self):
        counts = {
            threshold: routing_error_analysis(self.audit, threshold=threshold).n_escalated
            for threshold in THRESHOLDS
        }
        self.assertGreaterEqual(counts[0.9], counts[0.7])
        self.assertGreaterEqual(counts[0.7], counts[0.5])


# --------------------------------------------------------------------------- #
# Harmful escalation (escalation that *loses* accuracy)
# --------------------------------------------------------------------------- #
class HarmfulEscalationTests(unittest.TestCase):
    """An escalation can make the answer worse: the cheap answer was right, the
    premium attempt came back wrong.  That is a distinct failure mode from
    over-escalation (which only wastes tokens), so it gets its own accounting."""

    @classmethod
    def setUpClass(cls) -> None:
        from routeaudit.accounting import attribute

        # A seed/threshold pair where the phenomenon actually occurs.
        cls.audit = run_audit(AuditConfig(n_tasks=20, thresholds=(0.9,), seed=0))
        cls.threshold = 0.9
        cls.analysis = routing_error_analysis(cls.audit, threshold=cls.threshold)
        cls.cascade = cls.audit.by_task(Strategy.CASCADE, cls.threshold)
        cls.ledgers = {t: attribute(r) for t, r in cls.cascade.items()}

    def test_fixture_actually_contains_a_harmful_escalation(self):
        self.assertGreater(self.analysis.n_harmful_escalations, 0)

    def test_harmful_escalation_requires_cheap_correct_and_final_incorrect(self):
        for task_id, ledger in self.ledgers.items():
            if ledger.harmful_escalation:
                self.assertTrue(ledger.escalated, msg=task_id)
                self.assertIs(ledger.cheap_correct, True, msg=task_id)
                self.assertIs(ledger.final_correct, False, msg=task_id)

    def test_harmful_escalation_does_not_overlap_over_escalation(self):
        # over-escalation means the cheap answer was right *and was kept*;
        # harmful escalation means the escalation replaced it with a wrong one
        for task_id, ledger in self.ledgers.items():
            if ledger.harmful_escalation:
                self.assertFalse(
                    ledger.final_correct, msg=f"{task_id} cannot be both"
                )

    def test_count_matches_a_manual_scan(self):
        manual = sum(1 for ledger in self.ledgers.values() if ledger.harmful_escalation)
        self.assertEqual(self.analysis.n_harmful_escalations, manual)

    def test_rate_is_relative_to_tasks(self):
        """Documented as a share of *tasks*, not of escalations."""
        self.assertAlmostEqual(
            self.analysis.harmful_escalation_rate,
            self.analysis.n_harmful_escalations / self.analysis.n_tasks,
        )

    def test_rate_stays_in_the_unit_interval(self):
        self.assertGreaterEqual(self.analysis.harmful_escalation_rate, 0.0)
        self.assertLessEqual(self.analysis.harmful_escalation_rate, 1.0)

    def test_harmful_escalation_is_serialised(self):
        payload = self.analysis.to_dict()
        self.assertIn("n_harmful_escalations", payload)
        self.assertIn("harmful_escalation_rate", payload)

    def test_harmful_escalation_never_exceeds_escalation_count(self):
        analysis = routing_error_analysis(self.audit, threshold=self.threshold)
        self.assertLessEqual(analysis.n_harmful_escalations, analysis.n_escalated)


# --------------------------------------------------------------------------- #
# Threshold sensitivity
# --------------------------------------------------------------------------- #
class SensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = get_audit()
        cls.sensitivity = threshold_sensitivity(cls.audit)

    def test_points_are_sorted_by_threshold(self):
        thresholds = [point.threshold for point in self.sensitivity.points]
        self.assertEqual(thresholds, sorted(thresholds))

    def test_one_point_per_configured_threshold(self):
        self.assertEqual(
            [p.threshold for p in self.sensitivity.points], sorted(THRESHOLDS)
        )

    def test_operating_threshold_defaults_to_half(self):
        self.assertAlmostEqual(self.sensitivity.operating_threshold, 0.5)

    def test_point_lookup_by_threshold(self):
        point = self.sensitivity.point(0.7)
        self.assertAlmostEqual(point.threshold, 0.7)
        with self.assertRaises(KeyError):
            self.sensitivity.point(0.42)

    def test_point_metrics_match_the_underlying_summary(self):
        from routeaudit.accounting import summarise_audit

        summaries = summarise_audit(self.audit)
        for point in self.sensitivity.points:
            summary = summaries[f"C@{point.threshold:.1f}"]
            self.assertAlmostEqual(point.accuracy, summary.accuracy)
            self.assertEqual(point.total_tokens, summary.total_tokens)
            self.assertEqual(point.n_escalated, summary.n_escalated)

    def test_cheap_and_premium_baselines_are_reported(self):
        self.assertAlmostEqual(
            self.sensitivity.cheap_accuracy,
            compare(self.audit).cheap.accuracy,
        )
        self.assertAlmostEqual(
            self.sensitivity.premium_accuracy,
            compare(self.audit).premium.accuracy,
        )

    def test_best_and_cheapest_points_are_real_points(self):
        self.assertIn(self.sensitivity.best_accuracy_point, self.sensitivity.points)
        self.assertIn(self.sensitivity.cheapest_point, self.sensitivity.points)

    def test_cheapest_point_has_the_minimum_cost(self):
        minimum = min(p.total_cost_usd for p in self.sensitivity.points)
        self.assertAlmostEqual(self.sensitivity.cheapest_point.total_cost_usd, minimum)

    def test_marginal_economics_are_computed_for_later_points(self):
        # the first point has no predecessor
        self.assertIsNone(self.sensitivity.points[0].marginal_cost_per_point)
        self.assertIsNone(self.sensitivity.points[0].marginal_tokens_per_point)
        for point in self.sensitivity.points[1:]:
            # either a finite marginal or explicitly None on a flat accuracy step
            if point.marginal_cost_per_point is not None:
                self.assertIsInstance(point.marginal_cost_per_point, float)

    def test_sensitivity_requires_a_cascade(self):
        audit = run_audit(
            AuditConfig(n_tasks=4, thresholds=(0.5,), include_strategies=("A", "B"))
        )
        with self.assertRaises(ValueError):
            threshold_sensitivity(audit)


# --------------------------------------------------------------------------- #
# Regret vs the label-aware oracle
# --------------------------------------------------------------------------- #
class RegretTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = get_audit()
        cls.regret = regret_analysis(cls.audit, threshold=HEADLINE)

    def test_regret_is_non_negative(self):
        # the oracle is a lower bound on cost for the accuracy it must reach
        self.assertGreaterEqual(self.regret.regret_cost_usd, -1e-12)

    def test_regret_is_non_negative_across_seeds_and_thresholds(self):
        for seed in (1, 2, 3):
            audit = run_audit(
                AuditConfig(n_tasks=12, thresholds=(0.9, 0.7, 0.5), seed=seed)
            )
            for threshold in THRESHOLDS:
                regret = regret_analysis(audit, threshold=threshold)
                self.assertGreaterEqual(
                    regret.regret_cost_usd,
                    -1e-12,
                    msg=f"seed={seed} threshold={threshold}",
                )

    def test_regret_cost_share_is_within_the_unit_interval(self):
        self.assertGreaterEqual(self.regret.regret_cost_share, 0.0)
        self.assertLessEqual(self.regret.regret_cost_share, 1.0)

    def test_oracle_cost_does_not_exceed_the_premium_baseline(self):
        # always-premium is always feasible, so the oracle cannot cost more
        self.assertLessEqual(
            self.regret.oracle_cost_usd, self.regret.premium_cost_usd + 1e-12
        )

    def test_cascade_cost_matches_the_summary(self):
        from routeaudit.accounting import summarise_audit

        cascade = summarise_audit(self.audit)[f"C@{HEADLINE:.1f}"]
        self.assertAlmostEqual(self.regret.cascade_cost_usd, cascade.total_cost_usd)
        self.assertEqual(self.regret.cascade_tokens, cascade.total_tokens)
        self.assertEqual(self.regret.cascade_n_escalations, cascade.n_escalated)

    def test_routing_efficiency_is_bounded(self):
        self.assertGreaterEqual(self.regret.routing_efficiency, 0.0)
        self.assertLessEqual(self.regret.routing_efficiency, 1.0)

    def test_oracle_never_needs_more_escalations_than_tasks(self):
        self.assertLessEqual(self.regret.oracle_n_escalations, self.regret.n_tasks)

    def test_oracle_accuracy_never_exceeds_one(self):
        self.assertLessEqual(self.regret.oracle_accuracy, 1.0)
        self.assertGreaterEqual(self.regret.oracle_accuracy, 0.0)

    def test_oracle_plan_on_a_synthetic_matrix(self):
        """``oracle_plan`` must escalate cheapest-first until the target is met."""

        def run(verdict: Verdict, cost: float, tokens: int):
            from routeaudit.types import (
                Attempt,
                EscalationReason,
                ModelResponse,
                TaskRun,
                TokenBreakdown,
                Usage,
            )

            attempt = Attempt(
                index=0,
                stage="primary",
                model="m",
                tier="cheap",
                response=ModelResponse(
                    text="Answer: x",
                    usage=Usage(tokens, 0),
                    latency_ms=1.0,
                    model="m",
                    provider="mock",
                ),
                verdict=verdict,
                confidence=0.5,
                format_ok=True,
            )
            return TaskRun(
                task_id="t",
                strategy=Strategy.CHEAP,
                threshold=None,
                attempts=[attempt],
                final_answer="",
                final_model="m",
                final_tier="cheap",
                final_verdict=verdict,
                escalated=False,
                escalation_reason=EscalationReason.NONE,
                cheap_verdict=verdict,
                premium_verdict=None,
                tokens=TokenBreakdown.from_attempts([attempt]),
                cheap_cost_usd=cost,
                premium_cost_usd=0.0,
            )

        cheap_runs = {
            "a": run(Verdict.INCORRECT, 1.0, 100),
            "b": run(Verdict.INCORRECT, 1.0, 100),
            "c": run(Verdict.CORRECT, 1.0, 100),
        }
        premium_runs = {
            "a": run(Verdict.CORRECT, 5.0, 300),
            "b": run(Verdict.CORRECT, 9.0, 300),
            "c": run(Verdict.CORRECT, 5.0, 300),
        }

        # target already met by the cheap baseline -> no escalations
        plan = oracle_plan(cheap_runs, premium_runs, 1)
        self.assertEqual(plan.n_escalations, 0)
        self.assertEqual(plan.n_correct, 1)
        self.assertAlmostEqual(plan.cost_usd, 3.0)
        self.assertEqual(plan.max_correct, 3)  # 1 already right + 2 fixable

        # one more correct answer: pick the cheaper escalation ("a", delta 4)
        plan = oracle_plan(cheap_runs, premium_runs, 2)
        self.assertEqual(plan.n_escalations, 1)
        self.assertEqual(plan.n_correct, 2)
        self.assertAlmostEqual(plan.cost_usd, 3.0 + 4.0)

        # both escalations needed: 4 + 8
        plan = oracle_plan(cheap_runs, premium_runs, 3)
        self.assertEqual(plan.n_escalations, 2)
        self.assertEqual(plan.n_correct, 3)
        self.assertAlmostEqual(plan.cost_usd, 3.0 + 4.0 + 8.0)

    def test_oracle_plan_supports_attribute_access(self):
        """The plan is a named record, so callers can read fields by name."""
        from routeaudit.analysis import OraclePlan

        plan = oracle_plan({}, {}, 0)
        self.assertIsInstance(plan, OraclePlan)
        for field in ("cost_usd", "tokens", "n_correct", "n_escalations", "max_correct"):
            self.assertTrue(hasattr(plan, field), msg=field)

    def test_oracle_plan_reports_an_unreachable_target_honestly(self):
        """An unreachable target must not be silently rounded up."""
        from routeaudit.types import (
            Attempt,
            EscalationReason,
            ModelResponse,
            TaskRun,
            TokenBreakdown,
            Usage,
        )

        def run(verdict: Verdict, cost: float):
            attempt = Attempt(
                index=0,
                stage="primary",
                model="m",
                tier="cheap",
                response=ModelResponse(
                    text="Answer: x",
                    usage=Usage(100, 0),
                    latency_ms=1.0,
                    model="m",
                    provider="mock",
                ),
                verdict=verdict,
                confidence=0.5,
                format_ok=True,
            )
            return TaskRun(
                task_id="t",
                strategy=Strategy.CHEAP,
                threshold=None,
                attempts=[attempt],
                final_answer="",
                final_model="m",
                final_tier="cheap",
                final_verdict=verdict,
                escalated=False,
                escalation_reason=EscalationReason.NONE,
                cheap_verdict=verdict,
                premium_verdict=None,
                tokens=TokenBreakdown.from_attempts([attempt]),
                cheap_cost_usd=cost,
                premium_cost_usd=0.0,
            )

        # both models wrong on every task -> nothing can be fixed
        cheap_runs = {"a": run(Verdict.INCORRECT, 1.0)}
        premium_runs = {"a": run(Verdict.INCORRECT, 5.0)}
        plan = oracle_plan(cheap_runs, premium_runs, 1)
        self.assertEqual(plan.n_correct, 0)
        self.assertEqual(plan.n_escalations, 0)
        self.assertEqual(plan.max_correct, 0)

    def test_oracle_accuracy_never_exceeds_cascade_accuracy(self):
        """The oracle may fall short of the cascade, never exceed it."""
        for seed in (1, 2, 3, 4):
            audit = run_audit(
                AuditConfig(n_tasks=12, thresholds=(0.9, 0.7, 0.5), seed=seed)
            )
            for threshold in THRESHOLDS:
                regret = regret_analysis(audit, threshold=threshold)
                self.assertLessEqual(
                    regret.oracle_accuracy,
                    regret.cascade_accuracy + 1e-9,
                    msg=f"seed={seed} threshold={threshold}",
                )

    def test_regret_flags_when_the_oracle_could_not_match_accuracy(self):
        """The shortfall flag must agree with the reported accuracies."""
        for seed in (1, 2, 3):
            audit = run_audit(
                AuditConfig(n_tasks=12, thresholds=(0.9, 0.7, 0.5), seed=seed)
            )
            for threshold in THRESHOLDS:
                regret = regret_analysis(audit, threshold=threshold)
                matched = abs(regret.oracle_accuracy - regret.cascade_accuracy) < 1e-9
                self.assertEqual(regret.oracle_accuracy_matched, matched)
                self.assertAlmostEqual(
                    regret.oracle_accuracy_gap,
                    regret.cascade_accuracy - regret.oracle_accuracy,
                )
                # a positive gap must be reflected in the flag
                if regret.oracle_accuracy_gap > 1e-9:
                    self.assertFalse(regret.oracle_accuracy_matched)

    def test_oracle_max_correct_bounds_the_attainable_accuracy(self):
        regret = regret_analysis(self.audit, threshold=HEADLINE)
        self.assertLessEqual(regret.oracle_max_correct, regret.n_tasks)
        self.assertGreaterEqual(regret.oracle_max_correct, 0)
        # the oracle never claims more correct answers than its own ceiling
        self.assertLessEqual(
            regret.oracle_accuracy * regret.n_tasks,
            regret.oracle_max_correct + 1e-9,
        )


# --------------------------------------------------------------------------- #
# Domain breakdown and the analysis bundle
# --------------------------------------------------------------------------- #
class DomainAndBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = get_audit()

    def test_domain_slices_cover_every_task_once(self):
        slices = domain_breakdown(self.audit, threshold=HEADLINE)
        self.assertEqual(sum(s.n_tasks for s in slices), N_TASKS)

    def test_domain_slices_are_sorted_by_name(self):
        slices = domain_breakdown(self.audit, threshold=HEADLINE)
        self.assertEqual([s.domain for s in slices], sorted(s.domain for s in slices))

    def test_domain_metrics_are_bounded(self):
        for slice_ in domain_breakdown(self.audit, threshold=HEADLINE):
            for rate in (
                slice_.cheap_accuracy,
                slice_.premium_accuracy,
                slice_.cascade_accuracy,
                slice_.cascade_escalation_rate,
                slice_.wasted_token_share,
            ):
                self.assertGreaterEqual(rate, 0.0, msg=slice_.domain)
                self.assertLessEqual(rate, 1.0, msg=slice_.domain)

    def test_analyse_bundle_is_internally_consistent(self):
        bundle = analyse(self.audit, threshold=HEADLINE)
        self.assertEqual(bundle.comparison.cascade_key, f"C@{HEADLINE:.1f}")
        self.assertEqual(
            bundle.routing_errors.total_tokens, bundle.comparison.cascade.total_tokens
        )
        self.assertAlmostEqual(
            bundle.regret.cascade_cost_usd, bundle.comparison.cascade.total_cost_usd
        )
        self.assertEqual(
            len(bundle.sensitivity.points), len(THRESHOLDS)
        )

    def test_analyse_bundle_serialises_to_dict(self):
        payload = analyse(self.audit, threshold=HEADLINE).to_dict()
        for key in ("comparison", "routing_errors", "sensitivity", "regret", "domains"):
            self.assertIn(key, payload)
        self.assertEqual(payload["n_tasks"], N_TASKS)
        self.assertEqual(payload["seed"], 20260101)


if __name__ == "__main__":
    unittest.main()
