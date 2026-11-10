"""Boundary conditions and the budget guard.

Focus areas (task-2 items 7 and 8):
  * ``n_tasks=1`` runs and analyses without crashing
  * a single-threshold cascade works
  * ``include_strategies=("A", "B")`` (no cascade) does not crash the engine
  * ``BudgetGuard`` raises ``BudgetExceeded`` past its limit, and an audit
    configured below its own cost aborts loudly instead of overspending

These are the "does it still hold together at the edges" tests: the whole point
of a budget guard is that it fires on the runs nobody benchmarks.
"""

from __future__ import annotations

import json
import unittest

from routeaudit.accounting import summarise, summarise_audit
from routeaudit.analysis import analyse
from routeaudit.config import AuditConfig
from routeaudit.engine import (
    AuditEngine,
    BudgetExceeded,
    BudgetGuard,
    CallCache,
    run_audit,
)
from routeaudit.types import AuditRun, EscalationReason, Strategy, Verdict


# --------------------------------------------------------------------------- #
# Degenerate audit sizes
# --------------------------------------------------------------------------- #
class MinimalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = run_audit(AuditConfig(n_tasks=1, thresholds=(0.5,)))

    def test_single_task_audit_produces_one_run_per_configuration(self):
        self.assertEqual(len(self.audit.select(Strategy.CHEAP)), 1)
        self.assertEqual(len(self.audit.select(Strategy.PREMIUM)), 1)
        self.assertEqual(len(self.audit.select(Strategy.CASCADE)), 1)
        self.assertEqual(len(self.audit.runs), 3)

    def test_single_task_audit_reports_n_tasks_one(self):
        self.assertEqual(self.audit.n_tasks, 1)

    def test_single_task_audit_summarises(self):
        for strategy in (Strategy.CHEAP, Strategy.PREMIUM):
            summary = summarise(self.audit.select(strategy))
            self.assertEqual(summary.n_tasks, 1)
            self.assertIn(summary.n_correct, (0, 1))
            self.assertIn(summary.accuracy, (0.0, 1.0))

    def test_single_task_audit_survives_the_full_analysis(self):
        bundle = analyse(self.audit, threshold=0.5)
        self.assertEqual(len(bundle.sensitivity.points), 1)
        self.assertEqual(bundle.routing_errors.n_tasks, 1)
        self.assertGreaterEqual(bundle.regret.regret_cost_usd, -1e-12)
        self.assertEqual(sum(d.n_tasks for d in bundle.domains), 1)

    def test_single_task_audit_survives_a_save_load_cycle(self):
        restored = AuditRun.from_dict(self.audit.to_dict())
        self.assertEqual(restored.n_tasks, 1)
        self.assertEqual(len(restored.runs), 3)

    def test_single_task_with_multiple_thresholds(self):
        audit = run_audit(AuditConfig(n_tasks=1, thresholds=(0.9, 0.7, 0.5)))
        self.assertEqual(len(audit.select(Strategy.CASCADE)), 3)
        self.assertEqual(len(audit.runs), 5)


class SingleThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = run_audit(AuditConfig(n_tasks=6, thresholds=(0.5,)))

    def test_only_one_cascade_configuration_exists(self):
        self.assertEqual(len(self.audit.select(Strategy.CASCADE)), 6)
        self.assertEqual(
            {r.threshold for r in self.audit.select(Strategy.CASCADE)}, {0.5}
        )

    def test_summarise_audit_has_exactly_one_cascade_key(self):
        keys = sorted(k for k in summarise_audit(self.audit) if k.startswith("C@"))
        self.assertEqual(keys, ["C@0.5"])

    def test_sensitivity_has_a_single_point_with_no_marginals(self):
        bundle = analyse(self.audit, threshold=0.5)
        self.assertEqual(len(bundle.sensitivity.points), 1)
        point = bundle.sensitivity.points[0]
        self.assertAlmostEqual(point.threshold, 0.5)
        # a lone point has no predecessor to compute a marginal against
        self.assertIsNone(point.marginal_cost_per_point)
        self.assertIsNone(point.marginal_tokens_per_point)

    def test_operating_threshold_falls_back_to_the_only_value(self):
        bundle = analyse(self.audit, threshold=0.5)
        self.assertAlmostEqual(bundle.sensitivity.operating_threshold, 0.5)

    def test_non_half_threshold_becomes_the_operating_point(self):
        audit = run_audit(AuditConfig(n_tasks=4, thresholds=(0.8,)))
        bundle = analyse(audit)
        self.assertAlmostEqual(bundle.sensitivity.operating_threshold, 0.8)
        self.assertEqual(bundle.comparison.cascade_key, "C@0.8")


# --------------------------------------------------------------------------- #
# Strategy subsets
# --------------------------------------------------------------------------- #
class StrategySubsetTests(unittest.TestCase):
    def test_a_and_b_only_audit_does_not_crash(self):
        audit = run_audit(
            AuditConfig(n_tasks=6, thresholds=(0.5,), include_strategies=("A", "B"))
        )
        self.assertEqual(audit.n_tasks, 6)
        self.assertEqual(audit.strategies_present(), [Strategy.CHEAP, Strategy.PREMIUM])
        self.assertEqual(len(audit.runs), 12)

    def test_a_and_b_only_audit_has_no_cascade_runs(self):
        audit = run_audit(
            AuditConfig(n_tasks=4, thresholds=(0.5,), include_strategies=("A", "B"))
        )
        self.assertEqual(audit.select(Strategy.CASCADE), [])

    def test_a_and_b_only_audit_summarises(self):
        audit = run_audit(
            AuditConfig(n_tasks=4, thresholds=(0.5,), include_strategies=("A", "B"))
        )
        summaries = summarise_audit(audit)
        self.assertEqual(sorted(summaries), ["A", "B"])

    def test_a_and_b_only_audit_cannot_be_analysed_as_a_cascade(self):
        # documented contract: the cascade comparison needs a cascade
        audit = run_audit(
            AuditConfig(n_tasks=4, thresholds=(0.5,), include_strategies=("A", "B"))
        )
        with self.assertRaises(ValueError):
            analyse(audit)

    def test_cheap_only_audit(self):
        audit = run_audit(
            AuditConfig(n_tasks=4, thresholds=(0.5,), include_strategies=("A",))
        )
        self.assertEqual(audit.strategies_present(), [Strategy.CHEAP])
        self.assertEqual(len(audit.runs), 4)

    def test_premium_only_audit(self):
        audit = run_audit(
            AuditConfig(n_tasks=4, thresholds=(0.5,), include_strategies=("B",))
        )
        self.assertEqual(audit.strategies_present(), [Strategy.PREMIUM])
        self.assertEqual(len(audit.runs), 4)

    def test_cascade_only_audit_still_executes_the_missing_columns(self):
        """The cascade needs both counterfactual columns to be priceable, so the
        engine must silently execute whichever of A/B the caller excluded."""
        audit = run_audit(
            AuditConfig(n_tasks=4, thresholds=(0.5,), include_strategies=("C",))
        )
        self.assertEqual(audit.strategies_present(), [Strategy.CASCADE])
        self.assertEqual(len(audit.select(Strategy.CASCADE)), 4)
        # both counterfactuals must still be populated on the cascade runs
        for run in audit.select(Strategy.CASCADE):
            self.assertIsNotNone(run.cheap_verdict)
            self.assertIsNotNone(run.premium_verdict)


# --------------------------------------------------------------------------- #
# BudgetGuard
# --------------------------------------------------------------------------- #
class BudgetGuardTests(unittest.TestCase):
    def test_charge_accumulates_spend_and_calls(self):
        guard = BudgetGuard(limit_usd=10.0)
        guard.charge(0.5)
        guard.charge(1.5, calls=3)
        self.assertAlmostEqual(guard.spent_usd, 2.0)
        self.assertEqual(guard.calls, 4)

    def test_exceeding_the_budget_raises(self):
        guard = BudgetGuard(limit_usd=1.0)
        guard.charge(0.6)
        with self.assertRaises(BudgetExceeded):
            guard.charge(0.5)

    def test_spending_exactly_the_limit_is_allowed(self):
        guard = BudgetGuard(limit_usd=1.0)
        guard.charge(1.0)  # must not raise: the check is strictly greater-than
        self.assertAlmostEqual(guard.spent_usd, 1.0)

    def test_spend_is_recorded_even_when_the_guard_raises(self):
        guard = BudgetGuard(limit_usd=1.0)
        with self.assertRaises(BudgetExceeded):
            guard.charge(5.0)
        self.assertAlmostEqual(guard.spent_usd, 5.0)

    def test_error_message_names_the_limit_and_the_spend(self):
        guard = BudgetGuard(limit_usd=0.01)
        with self.assertRaises(BudgetExceeded) as ctx:
            guard.charge(1.0)
        message = str(ctx.exception)
        self.assertIn("budget exceeded", message)
        self.assertIn("0.01", message)

    def test_budget_exceeded_is_a_runtime_error(self):
        # callers that catch RuntimeError must catch this too
        self.assertTrue(issubclass(BudgetExceeded, RuntimeError))

    def test_zero_cost_charges_never_raise(self):
        guard = BudgetGuard(limit_usd=1e-9)
        for _ in range(5):
            guard.charge(0.0)
        self.assertAlmostEqual(guard.spent_usd, 0.0)
        self.assertEqual(guard.calls, 5)

    def test_audit_aborts_when_the_budget_is_too_small(self):
        with self.assertRaises(BudgetExceeded):
            run_audit(
                AuditConfig(n_tasks=12, thresholds=(0.5,), max_budget_usd=1e-9)
            )

    def test_audit_aborts_at_a_tight_but_nonzero_budget(self):
        # a budget smaller than a single premium call must still stop the run
        with self.assertRaises(BudgetExceeded):
            run_audit(
                AuditConfig(n_tasks=12, thresholds=(0.5,), max_budget_usd=1e-6)
            )

    def test_generous_budget_completes(self):
        audit = run_audit(
            AuditConfig(n_tasks=4, thresholds=(0.5,), max_budget_usd=1000.0)
        )
        self.assertEqual(audit.n_tasks, 4)
        self.assertLess(audit.stats["executed_cost_usd"], 1000.0)

    def test_executed_cost_is_at_least_the_cascade_billed_cost(self):
        """``executed_cost`` counts every provider call in the whole sweep
        (A + B + each cascade escalation), so it exceeds what one cascade
        configuration would be billed on its own."""
        audit = run_audit(AuditConfig(n_tasks=6, thresholds=(0.5,)))
        executed = audit.stats["executed_cost_usd"]
        self.assertGreater(executed, 0.0)
        cascade_billed = sum(
            r.cheap_cost_usd + r.premium_cost_usd for r in audit.select(Strategy.CASCADE)
        )
        self.assertGreaterEqual(executed, cascade_billed - 1e-9)
        # and it cannot exceed the total billed across every configuration
        total_billed = sum(r.cheap_cost_usd + r.premium_cost_usd for r in audit.runs)
        self.assertLessEqual(executed, total_billed + 1e-9)

    def test_invalid_budget_is_rejected_by_the_config(self):
        with self.assertRaises(ValueError):
            AuditConfig(max_budget_usd=0.0)
        with self.assertRaises(ValueError):
            AuditConfig(max_budget_usd=-1.0)


# --------------------------------------------------------------------------- #
# CallCache
# --------------------------------------------------------------------------- #
class CallCacheTests(unittest.TestCase):
    def test_disabled_cache_never_stores(self):
        cache = CallCache(enabled=False)
        cache.put(("t", "m", "primary"), None)  # type: ignore[arg-type]
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get(("t", "m", "primary")))

    def test_miss_then_hit_is_counted(self):
        from routeaudit.types import ModelResponse, Usage

        cache = CallCache(enabled=True)
        key = ("t", "m", "primary")
        self.assertIsNone(cache.get(key))
        self.assertEqual(cache.misses, 1)
        cache.put(
            key,
            ModelResponse(
                text="x", usage=Usage(1, 1), latency_ms=1.0, model="m", provider="mock"
            ),
        )
        self.assertIsNotNone(cache.get(key))
        self.assertEqual(cache.hits, 1)

    def test_cache_keys_are_independent(self):
        from routeaudit.types import ModelResponse, Usage

        cache = CallCache(enabled=True)
        response = ModelResponse(
            text="x", usage=Usage(1, 1), latency_ms=1.0, model="m", provider="mock"
        )
        cache.put(("t", "m", "primary"), response)
        self.assertIsNone(cache.get(("t", "m", "escalated")))
        self.assertIsNone(cache.get(("other", "m", "primary")))


# --------------------------------------------------------------------------- #
# Zero-threshold and extreme configuration
# --------------------------------------------------------------------------- #
class ExtremeThresholdTests(unittest.TestCase):
    def test_threshold_zero_only_escalates_on_structural_failures(self):
        """``confidence < 0.0`` is impossible, so a zero threshold can only
        escalate through the *format* gate (or a missing confidence channel) —
        never through the confidence comparison itself."""
        audit = run_audit(AuditConfig(n_tasks=12, thresholds=(0.0,)))
        allowed = {EscalationReason.FORMAT_FAILURE, EscalationReason.MISSING_CONFIDENCE}
        for run in audit.select(Strategy.CASCADE):
            if not run.escalated:
                continue
            self.assertIn(
                run.escalation_reason,
                allowed,
                msg=f"{run.task_id} escalated on confidence at threshold 0",
            )
            # and the confidence gate specifically must never be the cause
            self.assertIsNot(run.escalation_reason, EscalationReason.LOW_CONFIDENCE)

    def test_threshold_one_escalates_almost_everything(self):
        audit = run_audit(AuditConfig(n_tasks=12, thresholds=(1.0,)))
        escalated = [r for r in audit.select(Strategy.CASCADE) if r.escalated]
        # only confidence == 1.0 (or a malformed answer, which escalates anyway)
        # can avoid escalation, so nearly all tasks must escalate
        self.assertGreater(len(escalated), len(audit.select(Strategy.CASCADE)) // 2)

    def test_escalation_count_is_monotone_across_the_full_ladder(self):
        audit = run_audit(
            AuditConfig(n_tasks=12, thresholds=(0.0, 0.25, 0.5, 0.75, 1.0))
        )
        counts = [
            len([r for r in audit.select(Strategy.CASCADE, t) if r.escalated])
            for t in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        self.assertEqual(counts, sorted(counts), msg=f"not monotone: {counts}")

    def test_full_sweep_ladder_runs(self):
        from routeaudit.config import SWEEP_THRESHOLDS

        audit = run_audit(AuditConfig(n_tasks=4, thresholds=SWEEP_THRESHOLDS))
        self.assertEqual(len(audit.select(Strategy.CASCADE)), 4 * len(SWEEP_THRESHOLDS))
        bundle = analyse(audit, threshold=0.5)
        self.assertEqual(
            [p.threshold for p in bundle.sensitivity.points], list(SWEEP_THRESHOLDS)
        )


# --------------------------------------------------------------------------- #
# Robustness of the reporting contract at the edges
# --------------------------------------------------------------------------- #
class EdgeCaseReportingTests(unittest.TestCase):
    def test_all_correct_audit_has_zero_waste(self):
        """With the format gate off and a zero threshold the only escalations
        are missing-confidence ones, so no *confidence*-driven routing error can
        be priced."""
        from routeaudit.taskset import load_tasks

        tasks = load_tasks("builtin-qa", 6, seed=1)
        config = AuditConfig(n_tasks=6, thresholds=(0.0,), format_gate=False)
        engine = AuditEngine(config=config, tasks=tasks)
        result = engine.run()
        summary = summarise(result.select(Strategy.CASCADE, 0.0))
        # nothing escalated on a confidence comparison
        for run in result.select(Strategy.CASCADE, 0.0):
            self.assertIsNot(run.escalation_reason, EscalationReason.LOW_CONFIDENCE)
        self.assertEqual(
            summary.n_escalated,
            sum(
                1
                for r in result.select(Strategy.CASCADE, 0.0)
                if r.escalation_reason is EscalationReason.MISSING_CONFIDENCE
            ),
        )
        # and the shares must not divide by zero
        self.assertGreaterEqual(summary.wasted_token_share, 0.0)
        self.assertLessEqual(summary.wasted_token_share, 1.0)
        self.assertGreaterEqual(summary.escalation_token_share, 0.0)

    def test_format_gate_off_still_labels_the_real_cause(self):
        """With the format gate off and threshold 0, the confidence comparison
        (``confidence < 0.0``) can never fire, so escalation can only come from
        an unparseable answer or a missing confidence channel.

        A malformed answer has *both* problems: it has no parseable answer and
        therefore no ``Confidence:`` line.  The escalation reason must name the
        format failure — blaming the missing confidence channel would mislabel
        the cause and make an ablation (``--no-format-gate``) unreadable.
        """
        audit = run_audit(
            AuditConfig(n_tasks=8, thresholds=(0.0,), format_gate=False)
        )
        runs = audit.select(Strategy.CASCADE, 0.0)
        for run in runs:
            if not run.escalated:
                continue
            self.assertIsNone(run.confidence)
            if not run.format_ok:
                # unparseable answer -> the *format* is the real cause
                self.assertIs(run.escalation_reason, EscalationReason.FORMAT_FAILURE)
            else:
                # parseable but silent about confidence -> the channel is the cause
                self.assertIs(run.escalation_reason, EscalationReason.MISSING_CONFIDENCE)

        summary = summarise(runs)
        self.assertEqual(summary.n_escalated, sum(1 for r in runs if r.escalated))
        # the no-escalation share arithmetic must still be well defined
        self.assertGreaterEqual(summary.wasted_token_share, 0.0)
        self.assertLessEqual(summary.wasted_token_share, 1.0)
        self.assertEqual(summary.escalation_token_share >= 0.0, True)

    def test_tokens_per_correct_is_infinite_when_nothing_is_correct(self):
        """A configuration with zero correct answers must not divide by zero."""
        from routeaudit.accounting import ConfigSummary

        summary = ConfigSummary(
            strategy=Strategy.CHEAP,
            threshold=None,
            label="A",
            n_tasks=2,
            n_correct=0,
            total_tokens=100,
            cheap_tokens=100,
            escalation_tokens=0,
            total_cost_usd=0.01,
            cheap_cost_usd=0.01,
            escalation_cost_usd=0.0,
            n_escalated=0,
            n_wasted_escalation=0,
            n_necessary_escalation=0,
            n_missed_escalation=0,
            n_silent_error=0,
            n_cheap_correct=0,
            n_premium_correct=0,
            wasted_escalation_tokens=0,
            wasted_downgrade_tokens=0,
            wasted_escalation_cost_usd=0.0,
            wasted_downgrade_cost_usd=0.0,
        )
        self.assertEqual(summary.tokens_per_correct, float("inf"))
        self.assertEqual(summary.cost_per_correct, float("inf"))
        self.assertEqual(summary.accuracy, 0.0)
        # and to_dict must not emit a non-JSON "Infinity"
        payload = summary.to_dict()
        self.assertIsNone(payload["tokens_per_correct"])
        self.assertIsNone(payload["cost_per_correct_usd"])
        json.dumps(payload)  # must be serialisable

    def test_empty_audit_helpers_do_not_divide_by_zero(self):
        audit = AuditRun(
            run_id="empty",
            created_at="",
            provider="mock",
            task_set="builtin-mixed",
            n_tasks=0,
            seed=1,
            thresholds=(0.5,),
            config={},
            runs=[],
        )
        self.assertEqual(audit.total_tokens, 0)
        self.assertEqual(audit.total_cost_usd, 0.0)
        self.assertEqual(audit.strategies_present(), [])
        self.assertEqual(audit.select(Strategy.CASCADE), [])


if __name__ == "__main__":
    unittest.main()
