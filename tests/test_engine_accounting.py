"""Tests for ``routeaudit.engine`` + ``routeaudit.accounting``.

These run a *real* audit end to end with the deterministic offline simulator,
using a small task set (``n_tasks=12``) so the suite stays fast.

Focus areas (task-2 item 5):
  * every task has one A run and one B run; C has one run per threshold
  * counterfactual-matrix consistency: C's ``cheap_verdict``/``premium_verdict``
    equal A's / B's verdict for the same task
  * escalated runs satisfy ``tokens.total == primary + escalated``
  * non-escalated runs have ``escalated_tokens == 0``
  * ``summarise`` agrees with a hand count
  * determinism: the same seed reproduces the same run JSON
"""

from __future__ import annotations

import json
import os
import shutil
import unittest
import uuid
from pathlib import Path

from routeaudit.accounting import attribute, attribute_all, summarise, summarise_audit
from routeaudit.config import AuditConfig
from routeaudit.engine import AuditEngine, BudgetExceeded, build_matrix, run_audit
from routeaudit.types import AuditRun, Strategy, Verdict

# --------------------------------------------------------------------------- #
# Sandbox-safe scratch directory
#
# NOTE: ``tempfile.TemporaryDirectory`` cannot be used here.  Under the DSH
# workspace-write sandbox it creates the directory with mode 0o700, and the
# confined process token is then unable to write inside it (PermissionError),
# leaving an undeletable directory behind.  ``os.makedirs`` with the default
# mode works, so the tests manage their own scratch dir under ``tests/_tmp``.
# --------------------------------------------------------------------------- #
SCRATCH_ROOT = Path(__file__).resolve().parent / "_tmp"


def make_scratch_dir() -> Path:
    os.makedirs(SCRATCH_ROOT, exist_ok=True)
    path = SCRATCH_ROOT / uuid.uuid4().hex[:12]
    os.makedirs(path, exist_ok=True)
    return path


def remove_scratch_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


# A small audit used by most tests in this module.  Module-level so the
# (cheap) simulation runs once rather than once per test.
N_TASKS = 12
THRESHOLDS = (0.9, 0.7, 0.5)


def build_audit() -> AuditRun:
    config = AuditConfig(n_tasks=N_TASKS, thresholds=THRESHOLDS, seed=20260101)
    return run_audit(config)


_AUDIT: AuditRun | None = None


def get_audit() -> AuditRun:
    global _AUDIT
    if _AUDIT is None:
        _AUDIT = build_audit()
    return _AUDIT


# --------------------------------------------------------------------------- #
# Run structure
# --------------------------------------------------------------------------- #
class AuditStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = get_audit()

    def test_n_tasks_matches_the_config(self):
        self.assertEqual(self.audit.n_tasks, N_TASKS)

    def test_thresholds_are_persisted_in_order(self):
        self.assertEqual(self.audit.thresholds, THRESHOLDS)

    def test_every_task_has_exactly_one_cheap_run(self):
        cheap = self.audit.select(Strategy.CHEAP)
        self.assertEqual(len(cheap), N_TASKS)
        self.assertEqual(len({r.task_id for r in cheap}), N_TASKS)

    def test_every_task_has_exactly_one_premium_run(self):
        premium = self.audit.select(Strategy.PREMIUM)
        self.assertEqual(len(premium), N_TASKS)
        self.assertEqual(len({r.task_id for r in premium}), N_TASKS)

    def test_cascade_has_one_run_per_threshold_per_task(self):
        for threshold in THRESHOLDS:
            runs = self.audit.select(Strategy.CASCADE, threshold)
            self.assertEqual(len(runs), N_TASKS, msg=f"threshold {threshold}")
            self.assertEqual(len({r.task_id for r in runs}), N_TASKS)

    def test_cascade_run_count_is_tasks_times_thresholds(self):
        self.assertEqual(
            len(self.audit.select(Strategy.CASCADE)), N_TASKS * len(THRESHOLDS)
        )

    def test_total_run_count(self):
        # A + B + C x thresholds
        self.assertEqual(len(self.audit.runs), N_TASKS * 2 + N_TASKS * len(THRESHOLDS))

    def test_strategy_and_threshold_are_consistent_on_each_run(self):
        for run in self.audit.runs:
            if run.strategy is Strategy.CASCADE:
                self.assertIsNotNone(run.threshold)
                self.assertIn(run.threshold, THRESHOLDS)
            else:
                self.assertIsNone(run.threshold)

    def test_a_and_b_runs_never_escalate(self):
        for run in self.audit.select(Strategy.CHEAP) + self.audit.select(Strategy.PREMIUM):
            self.assertFalse(run.escalated)
            self.assertEqual(run.tokens.escalated_tokens, 0)

    def test_a_runs_report_only_a_cheap_verdict(self):
        for run in self.audit.select(Strategy.CHEAP):
            self.assertIsNotNone(run.cheap_verdict)
            self.assertIsNone(run.premium_verdict)

    def test_b_runs_report_only_a_premium_verdict(self):
        for run in self.audit.select(Strategy.PREMIUM):
            self.assertIsNotNone(run.premium_verdict)
            self.assertIsNone(run.cheap_verdict)

    def test_runs_are_deterministically_ordered(self):
        # A block, then B, then C ordered by descending threshold
        order = [r.strategy for r in self.audit.runs]
        self.assertEqual(order, sorted(order, key=lambda s: {"A": 0, "B": 1, "C": 2}[s.value]))

    def test_executed_calls_are_fewer_than_billed_cascade_runs(self):
        # the whole point of the counterfactual matrix: the cheap stage is shared
        billed_calls = sum(len(r.attempts) for r in self.audit.runs)
        self.assertLess(self.audit.stats["executed_calls"], billed_calls)

    def test_stats_record_the_models_used(self):
        stats = self.audit.stats
        self.assertEqual(stats["cheap_model"]["name"], "gpt-4o-mini")
        self.assertEqual(stats["premium_model"]["name"], "gpt-4o")
        self.assertGreater(stats["executed_calls"], 0)


# --------------------------------------------------------------------------- #
# Counterfactual matrix consistency
# --------------------------------------------------------------------------- #
class CounterfactualMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = get_audit()
        cls.cheap = cls.audit.by_task(Strategy.CHEAP)
        cls.premium = cls.audit.by_task(Strategy.PREMIUM)

    def test_cascade_cheap_verdict_equals_strategy_a_verdict(self):
        for threshold in THRESHOLDS:
            for run in self.audit.select(Strategy.CASCADE, threshold):
                self.assertEqual(
                    run.cheap_verdict,
                    self.cheap[run.task_id].final_verdict,
                    msg=f"{run.task_id} @ {threshold}",
                )

    def test_cascade_premium_verdict_equals_strategy_b_verdict(self):
        for threshold in THRESHOLDS:
            for run in self.audit.select(Strategy.CASCADE, threshold):
                self.assertEqual(
                    run.premium_verdict,
                    self.premium[run.task_id].final_verdict,
                    msg=f"{run.task_id} @ {threshold}",
                )

    def test_counterfactual_verdicts_are_threshold_independent(self):
        # replaying the same matrix must not change the recorded counterfactuals
        by_task: dict[str, set] = {}
        for run in self.audit.select(Strategy.CASCADE):
            by_task.setdefault(run.task_id, set()).add(
                (run.cheap_verdict, run.premium_verdict)
            )
        for task_id, seen in by_task.items():
            self.assertEqual(len(seen), 1, msg=f"{task_id} has unstable counterfactuals: {seen}")

    def test_build_matrix_agrees_with_runs(self):
        matrix = build_matrix(self.audit)
        self.assertEqual(len(matrix), N_TASKS)
        for task_id, (cheap_verdict, premium_verdict) in matrix.items():
            self.assertEqual(cheap_verdict, self.cheap[task_id].final_verdict)
            self.assertEqual(premium_verdict, self.premium[task_id].final_verdict)

    def test_non_escalated_cascade_ships_the_cheap_answer(self):
        for run in self.audit.select(Strategy.CASCADE):
            if run.escalated:
                continue
            self.assertEqual(run.final_verdict, run.cheap_verdict)
            self.assertEqual(run.final_tier, "cheap")

    def test_escalated_cascade_ships_the_premium_attempt(self):
        for run in self.audit.select(Strategy.CASCADE):
            if not run.escalated:
                continue
            self.assertEqual(run.final_tier, "premium")
            self.assertEqual(run.final_verdict, run.escalated_attempt.verdict)
            self.assertEqual(len(run.attempts), 2)


# --------------------------------------------------------------------------- #
# Token accounting invariants
# --------------------------------------------------------------------------- #
class TokenAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = get_audit()

    def test_escalated_runs_split_tokens_across_both_stages(self):
        escalated = [r for r in self.audit.select(Strategy.CASCADE) if r.escalated]
        self.assertTrue(escalated, "fixture should contain at least one escalation")
        for run in escalated:
            self.assertEqual(
                run.tokens.total_tokens,
                run.tokens.primary_tokens + run.tokens.escalated_tokens,
                msg=run.task_id,
            )
            self.assertGreater(run.tokens.primary_tokens, 0)
            self.assertGreater(run.tokens.escalated_tokens, 0)

    def test_escalated_run_tokens_equal_the_sum_of_its_attempts(self):
        for run in self.audit.select(Strategy.CASCADE):
            if not run.escalated:
                continue
            expected = sum(attempt.tokens for attempt in run.attempts)
            self.assertEqual(run.tokens.total_tokens, expected, msg=run.task_id)

    def test_non_escalated_runs_have_zero_escalated_tokens(self):
        for run in self.audit.select(Strategy.CASCADE):
            if run.escalated:
                continue
            self.assertEqual(run.tokens.escalated_tokens, 0, msg=run.task_id)
            self.assertEqual(run.tokens.total_tokens, run.tokens.primary_tokens)

    def test_non_escalated_runs_pay_no_premium_cost(self):
        for run in self.audit.select(Strategy.CASCADE):
            if run.escalated:
                continue
            self.assertEqual(run.premium_cost_usd, 0.0)
            self.assertAlmostEqual(run.cost_usd, run.cheap_cost_usd)

    def test_escalated_runs_pay_both_costs(self):
        for run in self.audit.select(Strategy.CASCADE):
            if not run.escalated:
                continue
            self.assertGreater(run.cheap_cost_usd, 0.0)
            self.assertGreater(run.premium_cost_usd, 0.0)
            self.assertAlmostEqual(
                run.cost_usd, run.cheap_cost_usd + run.premium_cost_usd
            )

    def test_a_run_tokens_all_land_in_the_primary_bucket(self):
        for run in self.audit.select(Strategy.CHEAP):
            self.assertEqual(run.tokens.escalated_tokens, 0)
            self.assertEqual(run.tokens.total_tokens, run.tokens.primary_tokens)

    def test_all_tokens_are_positive(self):
        for run in self.audit.runs:
            self.assertGreater(run.total_tokens, 0, msg=run.task_id)

    def test_premium_tier_costs_more_tokens_on_average(self):
        """The premium model's token band is wider, but per-call jitter
        (``MockProvider`` draws ``uniform(0.72, 1.35)``) means an individual
        task can go either way, so the invariant is the aggregate."""
        cheap = self.audit.select(Strategy.CHEAP)
        premium = self.audit.select(Strategy.PREMIUM)
        cheap_total = sum(r.total_tokens for r in cheap)
        premium_total = sum(r.total_tokens for r in premium)
        self.assertGreater(premium_total, cheap_total)


# --------------------------------------------------------------------------- #
# Accounting: attribute() and summarise()
# --------------------------------------------------------------------------- #
class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = get_audit()

    def test_attribute_mirrors_the_run(self):
        for run in self.audit.select(Strategy.CASCADE):
            ledger = attribute(run)
            self.assertEqual(ledger.task_id, run.task_id)
            self.assertEqual(ledger.strategy, run.strategy)
            self.assertEqual(ledger.threshold, run.threshold)
            self.assertEqual(ledger.total_tokens, run.tokens.total_tokens)
            self.assertEqual(ledger.escalated, run.escalated)
            self.assertEqual(ledger.final_correct, run.final_verdict.is_correct)

    def test_attribute_splits_cheap_and_escalation_tokens(self):
        for run in self.audit.select(Strategy.CASCADE):
            ledger = attribute(run)
            self.assertEqual(ledger.cheap_tokens, run.tokens.primary_tokens)
            self.assertEqual(ledger.escalation_tokens, run.tokens.escalated_tokens)

    def test_wasted_escalation_tokens_only_count_over_escalation(self):
        for ledger in attribute_all(self.audit.select(Strategy.CASCADE)):
            if ledger.escalated and ledger.cheap_correct is True:
                self.assertEqual(ledger.wasted_escalation_tokens, ledger.escalation_tokens)
            else:
                self.assertEqual(ledger.wasted_escalation_tokens, 0)

    def test_wasted_downgrade_tokens_only_count_necessary_escalation(self):
        for ledger in attribute_all(self.audit.select(Strategy.CASCADE)):
            if ledger.escalated and ledger.cheap_correct is False:
                self.assertEqual(ledger.wasted_downgrade_tokens, ledger.cheap_tokens)
            else:
                self.assertEqual(ledger.wasted_downgrade_tokens, 0)

    def test_silent_error_tokens_only_count_missed_escalation(self):
        for ledger in attribute_all(self.audit.select(Strategy.CASCADE)):
            if ledger.missed_escalation:
                self.assertEqual(ledger.silent_error_tokens, ledger.total_tokens)
            else:
                self.assertEqual(ledger.silent_error_tokens, 0)

    def test_attribute_all_preserves_order_and_length(self):
        runs = self.audit.select(Strategy.CASCADE)
        ledgers = attribute_all(runs)
        self.assertEqual(len(ledgers), len(runs))
        self.assertEqual([l.task_id for l in ledgers], [r.task_id for r in runs])


class SummariseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = get_audit()

    def test_n_correct_matches_a_manual_count(self):
        for threshold in THRESHOLDS:
            runs = self.audit.select(Strategy.CASCADE, threshold)
            summary = summarise(runs)
            manual = sum(1 for r in runs if r.final_verdict.is_correct)
            self.assertEqual(summary.n_correct, manual, msg=f"threshold {threshold}")
            self.assertEqual(summary.n_tasks, len(runs))

    def test_n_correct_matches_for_a_and_b(self):
        for strategy in (Strategy.CHEAP, Strategy.PREMIUM):
            runs = self.audit.select(strategy)
            summary = summarise(runs)
            self.assertEqual(summary.n_correct, sum(1 for r in runs if r.correct))

    def test_accuracy_is_n_correct_over_n_tasks(self):
        summary = summarise(self.audit.select(Strategy.CHEAP))
        self.assertAlmostEqual(summary.accuracy, summary.n_correct / summary.n_tasks)

    def test_token_totals_match_the_runs(self):
        runs = self.audit.select(Strategy.CASCADE, 0.7)
        summary = summarise(runs)
        self.assertEqual(summary.total_tokens, sum(r.total_tokens for r in runs))
        self.assertEqual(summary.cheap_tokens, sum(r.tokens.primary_tokens for r in runs))
        self.assertEqual(
            summary.escalation_tokens, sum(r.tokens.escalated_tokens for r in runs)
        )
        self.assertEqual(
            summary.total_tokens, summary.cheap_tokens + summary.escalation_tokens
        )

    def test_cost_totals_match_the_runs(self):
        runs = self.audit.select(Strategy.CASCADE, 0.7)
        summary = summarise(runs)
        self.assertAlmostEqual(
            summary.total_cost_usd, sum(r.cost_usd for r in runs)
        )

    def test_escalation_counts_match_the_runs(self):
        runs = self.audit.select(Strategy.CASCADE, 0.7)
        summary = summarise(runs)
        self.assertEqual(summary.n_escalated, sum(1 for r in runs if r.escalated))
        self.assertEqual(
            summary.n_wasted_escalation, sum(1 for r in runs if r.wasted_escalation)
        )
        self.assertEqual(
            summary.n_necessary_escalation,
            sum(1 for r in runs if r.necessary_escalation),
        )
        self.assertEqual(
            summary.n_missed_escalation, sum(1 for r in runs if r.silent_error)
        )

    def test_wasted_token_share_is_a_fraction(self):
        for threshold in THRESHOLDS:
            summary = summarise(self.audit.select(Strategy.CASCADE, threshold))
            self.assertGreaterEqual(summary.wasted_token_share, 0.0)
            self.assertLessEqual(summary.wasted_token_share, 1.0)

    def test_escalation_reasons_are_counted(self):
        runs = self.audit.select(Strategy.CASCADE, 0.9)
        summary = summarise(runs)
        self.assertEqual(sum(summary.escalation_reasons.values()), summary.n_escalated)

    def test_summarise_requires_at_least_one_run(self):
        with self.assertRaises(ValueError):
            summarise([])

    def test_summarise_audit_returns_every_configuration(self):
        summaries = summarise_audit(self.audit)
        self.assertIn("A", summaries)
        self.assertIn("B", summaries)
        for threshold in THRESHOLDS:
            self.assertIn(f"C@{threshold:.1f}", summaries)

    def test_summarise_audit_labels_are_stable(self):
        summaries = summarise_audit(self.audit)
        self.assertEqual(summaries["A"].label, "A · cheap-only")
        self.assertEqual(summaries["C@0.7"].label, "C · cascade @ 0.7")


# --------------------------------------------------------------------------- #
# Determinism and persistence
# --------------------------------------------------------------------------- #
class DeterminismTests(unittest.TestCase):
    def test_same_seed_reproduces_identical_run_json(self):
        config = dict(n_tasks=8, thresholds=(0.7, 0.5), seed=4242)
        first = run_audit(AuditConfig(**config))
        second = run_audit(AuditConfig(**config))

        self.assertEqual(
            json.dumps(first.to_dict()["runs"], sort_keys=True),
            json.dumps(second.to_dict()["runs"], sort_keys=True),
        )
        self.assertEqual(first.stats["executed_calls"], second.stats["executed_calls"])
        self.assertEqual(first.stats["executed_cost_usd"], second.stats["executed_cost_usd"])

    def test_run_id_differs_between_runs(self):
        # run_id/created_at are run metadata, not part of the deterministic payload
        config = dict(n_tasks=4, thresholds=(0.5,), seed=1)
        self.assertNotEqual(
            run_audit(AuditConfig(**config)).run_id,
            run_audit(AuditConfig(**config)).run_id,
        )

    def test_different_seed_changes_the_run(self):
        first = run_audit(AuditConfig(n_tasks=8, thresholds=(0.5,), seed=1))
        second = run_audit(AuditConfig(n_tasks=8, thresholds=(0.5,), seed=2))
        self.assertNotEqual(
            json.dumps(first.to_dict()["runs"], sort_keys=True),
            json.dumps(second.to_dict()["runs"], sort_keys=True),
        )

    def test_audit_survives_a_save_load_round_trip(self):
        audit = get_audit()
        scratch = make_scratch_dir()
        try:
            path = scratch / "audit.json"
            audit.save(str(path))
            self.assertTrue(path.exists())
            loaded = AuditRun.load(str(path))
            self.assertEqual(loaded.to_dict(), audit.to_dict())
            self.assertEqual(
                [r.to_dict() for r in loaded.runs], [r.to_dict() for r in audit.runs]
            )
        finally:
            remove_scratch_dir(scratch)

    def test_saved_file_is_utf8_and_reloadable_as_json(self):
        audit = get_audit()
        scratch = make_scratch_dir()
        try:
            path = scratch / "audit.json"
            audit.save(str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "route-audit/1")
            self.assertEqual(payload["n_tasks"], N_TASKS)
        finally:
            remove_scratch_dir(scratch)


# --------------------------------------------------------------------------- #
# Engine plumbing
# --------------------------------------------------------------------------- #
class EnginePlumbingTests(unittest.TestCase):
    def test_custom_task_list_is_used_and_n_tasks_is_reconciled(self):
        from routeaudit.taskset import load_tasks

        tasks = load_tasks("builtin-mixed", 5, 7)
        engine = AuditEngine(config=AuditConfig(n_tasks=99, thresholds=(0.5,)), tasks=tasks)
        audit = engine.run()
        self.assertEqual(audit.n_tasks, 5)
        self.assertEqual(len(audit.select(Strategy.CHEAP)), 5)

    def test_progress_callback_is_invoked(self):
        calls: list[tuple[str, int, int]] = []
        config = AuditConfig(n_tasks=4, thresholds=(0.5,))
        run_audit(config, progress=lambda phase, done, total: calls.append((phase, done, total)))
        self.assertTrue(calls)
        self.assertTrue(any(phase == "matrix" for phase, _, _ in calls))
        self.assertTrue(any(phase.startswith("cascade@") for phase, _, _ in calls))

    def test_broken_progress_callback_does_not_break_the_audit(self):
        def explode(phase: str, done: int, total: int) -> None:
            raise RuntimeError("progress renderer is broken")

        audit = run_audit(AuditConfig(n_tasks=4, thresholds=(0.5,)), progress=explode)
        self.assertEqual(audit.n_tasks, 4)

    def test_empty_task_list_is_rejected(self):
        engine = AuditEngine(config=AuditConfig(n_tasks=1, thresholds=(0.5,)), tasks=[])
        with self.assertRaises(ValueError):
            engine.run()

    def test_cache_is_used_across_cascade_thresholds(self):
        audit = run_audit(AuditConfig(n_tasks=8, thresholds=(0.9, 0.7, 0.5), enable_cache=True))
        # cheap attempts are shared, so hits must be recorded
        self.assertGreater(audit.stats["cache_hits"], 0)

    def test_disabling_the_cache_increases_executed_calls(self):
        cached = run_audit(AuditConfig(n_tasks=8, thresholds=(0.9, 0.7, 0.5), enable_cache=True))
        uncached = run_audit(
            AuditConfig(n_tasks=8, thresholds=(0.9, 0.7, 0.5), enable_cache=False)
        )
        self.assertEqual(uncached.stats["cache_hits"], 0)
        self.assertGreater(
            uncached.stats["executed_calls"], cached.stats["executed_calls"]
        )


if __name__ == "__main__":
    unittest.main()
