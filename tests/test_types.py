"""Tests for ``routeaudit.types``: the shared data contracts.

Focus areas (task-2 item 1):
  * ``Usage`` addition / ``total_tokens`` / zero element
  * ``TokenBreakdown`` primary/escalated split and ``from_attempts``
  * ``TaskRun.wasted_escalation`` / ``necessary_escalation`` / ``silent_error``
  * JSON round-trip for every contract (``to_dict`` / ``from_dict``)
"""

from __future__ import annotations

import unittest

from routeaudit.types import (
    Attempt,
    AuditRun,
    EscalationReason,
    ModelResponse,
    Strategy,
    Task,
    TaskRun,
    TokenBreakdown,
    Usage,
    Verdict,
    group_by_threshold,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_response(
    text: str = "Answer: 5\nConfidence: 0.80",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    error: str | None = None,
) -> ModelResponse:
    return ModelResponse(
        text=text,
        usage=Usage(prompt_tokens, completion_tokens),
        latency_ms=12.5,
        model="gpt-4o-mini",
        provider="mock",
        error=error,
    )


def make_attempt(
    stage: str = "primary",
    tier: str = "cheap",
    model: str = "gpt-4o-mini",
    verdict: Verdict = Verdict.CORRECT,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    confidence: float | None = 0.80,
    reason: EscalationReason = EscalationReason.NONE,
    index: int = 0,
) -> Attempt:
    return Attempt(
        index=index,
        stage=stage,
        model=model,
        tier=tier,
        response=make_response(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
        verdict=verdict,
        confidence=confidence,
        format_ok=True,
        escalation_reason=reason,
    )


def make_run(
    *,
    strategy: Strategy = Strategy.CASCADE,
    escalated: bool = False,
    cheap_verdict: Verdict | None = Verdict.CORRECT,
    premium_verdict: Verdict | None = Verdict.CORRECT,
    final_verdict: Verdict | None = None,
    attempts: list[Attempt] | None = None,
) -> TaskRun:
    if attempts is None:
        attempts = [make_attempt()]
        if escalated:
            attempts.append(
                make_attempt(
                    stage="escalated",
                    tier="premium",
                    model="gpt-4o",
                    prompt_tokens=20,
                    completion_tokens=8,
                    index=1,
                )
            )
    if final_verdict is None:
        final_verdict = attempts[-1].verdict
    return TaskRun(
        task_id="task-1",
        strategy=strategy,
        threshold=0.5 if strategy is Strategy.CASCADE else None,
        attempts=attempts,
        final_answer=attempts[-1].response.text,
        final_model=attempts[-1].model,
        final_tier=attempts[-1].tier,
        final_verdict=final_verdict,
        escalated=escalated,
        escalation_reason=(
            EscalationReason.LOW_CONFIDENCE if escalated else EscalationReason.NONE
        ),
        cheap_verdict=cheap_verdict,
        premium_verdict=premium_verdict,
        tokens=TokenBreakdown.from_attempts(attempts),
        cheap_cost_usd=0.001,
        premium_cost_usd=0.02 if escalated else 0.0,
    )


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #
class UsageTests(unittest.TestCase):
    def test_total_tokens_sums_prompt_and_completion(self):
        usage = Usage(prompt_tokens=120, completion_tokens=30)
        self.assertEqual(usage.total_tokens, 150)

    def test_addition_is_componentwise_and_returns_new_object(self):
        left = Usage(10, 2)
        right = Usage(5, 7)
        summed = left + right
        self.assertEqual(summed, Usage(15, 9))
        # addition must not mutate the operands (frozen value semantics)
        self.assertEqual(left, Usage(10, 2))
        self.assertEqual(right, Usage(5, 7))
        self.assertIsNot(summed, left)

    def test_addition_is_associative(self):
        a, b, c = Usage(1, 2), Usage(3, 4), Usage(5, 6)
        self.assertEqual((a + b) + c, a + (b + c))

    def test_zero_is_additive_identity(self):
        usage = Usage(42, 8)
        self.assertEqual(usage + Usage.zero(), usage)
        self.assertEqual(Usage.zero() + usage, usage)
        self.assertEqual(Usage.zero().total_tokens, 0)

    def test_negative_usage_is_rejected(self):
        with self.assertRaises(ValueError):
            Usage(prompt_tokens=-1, completion_tokens=0)
        with self.assertRaises(ValueError):
            Usage(prompt_tokens=0, completion_tokens=-5)

    def test_round_trip(self):
        usage = Usage(11, 22)
        self.assertEqual(Usage.from_dict(usage.to_dict()), usage)
        self.assertEqual(usage.to_dict(), {"prompt_tokens": 11, "completion_tokens": 22})


# --------------------------------------------------------------------------- #
# TokenBreakdown
# --------------------------------------------------------------------------- #
class TokenBreakdownTests(unittest.TestCase):
    def test_primary_and_escalated_are_separate_sums(self):
        breakdown = TokenBreakdown(
            primary_prompt_tokens=100,
            primary_completion_tokens=20,
            escalated_prompt_tokens=30,
            escalated_completion_tokens=10,
        )
        self.assertEqual(breakdown.primary_tokens, 120)
        self.assertEqual(breakdown.escalated_tokens, 40)
        self.assertEqual(breakdown.total_tokens, 160)
        # total must be the sum of the two buckets, not an independent field
        self.assertEqual(
            breakdown.total_tokens, breakdown.primary_tokens + breakdown.escalated_tokens
        )

    def test_from_attempts_routes_tokens_by_stage(self):
        primary = make_attempt(
            stage="primary", tier="cheap", prompt_tokens=100, completion_tokens=20
        )
        escalated = make_attempt(
            stage="escalated",
            tier="premium",
            model="gpt-4o",
            prompt_tokens=300,
            completion_tokens=50,
            index=1,
        )
        breakdown = TokenBreakdown.from_attempts([primary, escalated])
        self.assertEqual(breakdown.primary_prompt_tokens, 100)
        self.assertEqual(breakdown.primary_completion_tokens, 20)
        self.assertEqual(breakdown.escalated_prompt_tokens, 300)
        self.assertEqual(breakdown.escalated_completion_tokens, 50)
        self.assertEqual(breakdown.total_tokens, 470)

    def test_from_attempts_with_only_primary_leaves_escalated_zero(self):
        breakdown = TokenBreakdown.from_attempts([make_attempt(prompt_tokens=7, completion_tokens=3)])
        self.assertEqual(breakdown.primary_tokens, 10)
        self.assertEqual(breakdown.escalated_tokens, 0)

    def test_from_attempts_on_empty_iterable_is_zero(self):
        self.assertEqual(TokenBreakdown.from_attempts([]).total_tokens, 0)

    def test_addition_sums_each_field_independently(self):
        a = TokenBreakdown(1, 2, 3, 4)
        b = TokenBreakdown(10, 20, 30, 40)
        self.assertEqual(a + b, TokenBreakdown(11, 22, 33, 44))

    def test_round_trip(self):
        breakdown = TokenBreakdown(1, 2, 3, 4)
        self.assertEqual(TokenBreakdown.from_dict(breakdown.to_dict()), breakdown)

    def test_from_dict_defaults_missing_fields_to_zero(self):
        self.assertEqual(TokenBreakdown.from_dict({}), TokenBreakdown())


# --------------------------------------------------------------------------- #
# TaskRun escalation taxonomy
# --------------------------------------------------------------------------- #
class TaskRunEscalationTests(unittest.TestCase):
    def test_wasted_escalation_when_cheap_was_already_correct(self):
        run = make_run(
            escalated=True,
            cheap_verdict=Verdict.CORRECT,
            premium_verdict=Verdict.CORRECT,
        )
        self.assertTrue(run.escalated)
        self.assertTrue(run.wasted_escalation)
        self.assertFalse(run.necessary_escalation)
        self.assertFalse(run.silent_error)

    def test_necessary_escalation_when_cheap_was_wrong(self):
        run = make_run(
            escalated=True,
            cheap_verdict=Verdict.INCORRECT,
            premium_verdict=Verdict.CORRECT,
        )
        self.assertTrue(run.necessary_escalation)
        self.assertFalse(run.wasted_escalation)
        self.assertFalse(run.silent_error)

    def test_silent_error_when_not_escalated_and_cheap_wrong(self):
        run = make_run(
            escalated=False,
            cheap_verdict=Verdict.INCORRECT,
            final_verdict=Verdict.INCORRECT,
        )
        self.assertTrue(run.silent_error)
        self.assertFalse(run.wasted_escalation)
        self.assertFalse(run.necessary_escalation)

    def test_three_predicates_are_mutually_exclusive(self):
        scenarios = [
            make_run(escalated=True, cheap_verdict=Verdict.CORRECT),
            make_run(escalated=True, cheap_verdict=Verdict.INCORRECT),
            make_run(escalated=False, cheap_verdict=Verdict.INCORRECT),
            make_run(escalated=False, cheap_verdict=Verdict.CORRECT),
        ]
        for run in scenarios:
            flags = [run.wasted_escalation, run.necessary_escalation, run.silent_error]
            self.assertLessEqual(sum(flags), 1, msg=f"overlapping predicates on {run!r}")

    def test_no_escalation_means_no_escalation_cost(self):
        run = make_run(escalated=False)
        self.assertFalse(run.escalated)
        self.assertEqual(run.premium_cost_usd, 0.0)
        self.assertEqual(run.tokens.escalated_tokens, 0)
        self.assertIsNone(run.escalated_attempt)

    def test_correct_property_tracks_final_verdict(self):
        self.assertTrue(make_run(final_verdict=Verdict.CORRECT).correct)
        self.assertFalse(make_run(final_verdict=Verdict.INCORRECT).correct)
        self.assertFalse(make_run(final_verdict=Verdict.UNGRADED).correct)

    def test_cost_usd_sums_both_stages(self):
        run = make_run(escalated=True)
        self.assertAlmostEqual(run.cost_usd, run.cheap_cost_usd + run.premium_cost_usd)
        self.assertAlmostEqual(run.cost_usd, 0.021)

    def test_attempt_lookup_helpers(self):
        run = make_run(escalated=True)
        self.assertEqual(run.primary.stage, "primary")
        self.assertEqual(run.escalated_attempt.stage, "escalated")
        self.assertEqual(run.total_tokens, run.tokens.total_tokens)

    def test_cheap_and_premium_was_correct_handle_missing_verdict(self):
        run = make_run(cheap_verdict=None, premium_verdict=None)
        self.assertIsNone(run.cheap_was_correct)
        self.assertIsNone(run.premium_was_correct)

    def test_attempt_rejects_bad_stage_and_tier(self):
        response = make_response()
        with self.assertRaises(ValueError):
            Attempt(
                index=0,
                stage="bogus",
                model="m",
                tier="cheap",
                response=response,
                verdict=Verdict.CORRECT,
                confidence=0.5,
                format_ok=True,
            )
        with self.assertRaises(ValueError):
            Attempt(
                index=0,
                stage="primary",
                model="m",
                tier="bogus",
                response=response,
                verdict=Verdict.CORRECT,
                confidence=0.5,
                format_ok=True,
            )


# --------------------------------------------------------------------------- #
# TaskRun / AuditRun JSON round-trip
# --------------------------------------------------------------------------- #
class JsonRoundTripTests(unittest.TestCase):
    def test_task_round_trip(self):
        task = Task(
            task_id="math-001",
            prompt="Compute 2 * 3.",
            reference="6",
            domain="math",
            difficulty=0.25,
            metadata={"answer_kind": "number", "source": "unit-test"},
        )
        self.assertEqual(Task.from_dict(task.to_dict()), task)

    def test_task_rejects_invalid_fields(self):
        with self.assertRaises(ValueError):
            Task(task_id="", prompt="p", reference="r")
        with self.assertRaises(ValueError):
            Task(task_id="t", prompt="p", reference="r", difficulty=1.5)
        with self.assertRaises(ValueError):
            Task(task_id="t", prompt="p", reference="r", difficulty=-0.1)

    def test_taskrun_round_trip_preserves_everything(self):
        original = make_run(
            escalated=True,
            cheap_verdict=Verdict.INCORRECT,
            premium_verdict=Verdict.CORRECT,
            final_verdict=Verdict.CORRECT,
        )
        payload = original.to_dict()
        restored = TaskRun.from_dict(payload)
        self.assertEqual(restored.to_dict(), payload)
        self.assertEqual(restored.escalated, original.escalated)
        self.assertEqual(restored.tokens, original.tokens)
        self.assertEqual(restored.escalation_reason, original.escalation_reason)

    def test_taskrun_round_trip_preserves_none_verdicts(self):
        original = make_run(
            strategy=Strategy.CHEAP,
            escalated=False,
            cheap_verdict=Verdict.CORRECT,
            premium_verdict=None,
        )
        restored = TaskRun.from_dict(original.to_dict())
        self.assertIsNone(restored.premium_verdict)
        self.assertEqual(restored.cheap_verdict, Verdict.CORRECT)

    def test_attempt_round_trip(self):
        attempt = make_attempt(
            stage="escalated",
            tier="premium",
            model="gpt-4o",
            reason=EscalationReason.LOW_CONFIDENCE,
            index=1,
        )
        self.assertEqual(Attempt.from_dict(attempt.to_dict()), attempt)

    def test_model_response_round_trip_keeps_error(self):
        response = make_response(error="boom")
        restored = ModelResponse.from_dict(response.to_dict())
        self.assertEqual(restored, response)
        self.assertFalse(restored.ok)
        self.assertFalse(make_response(error="x").ok)
        self.assertTrue(make_response().ok)

    def test_auditrun_round_trip(self):
        run = make_run()
        audit = AuditRun(
            run_id="audit-abc123",
            created_at="2026-01-01T00:00:00+00:00",
            provider="mock",
            task_set="builtin-mixed",
            n_tasks=1,
            seed=7,
            thresholds=(0.5, 0.7),
            config={"n_tasks": 1},
            runs=[run],
            stats={"executed_calls": 3},
            notes={"method": "unit test"},
        )
        restored = AuditRun.from_dict(audit.to_dict())
        self.assertEqual(restored.to_dict(), audit.to_dict())
        self.assertEqual(restored.thresholds, (0.5, 0.7))
        self.assertEqual(restored.runs[0].task_id, run.task_id)

    def test_auditrun_total_helpers_sum_runs(self):
        audit = AuditRun(
            run_id="r",
            created_at="",
            provider="mock",
            task_set="s",
            n_tasks=2,
            seed=1,
            thresholds=(0.5,),
            config={},
            runs=[make_run(), make_run(escalated=True)],
        )
        self.assertEqual(
            audit.total_tokens, sum(r.total_tokens for r in audit.runs)
        )
        self.assertAlmostEqual(
            audit.total_cost_usd, sum(r.cost_usd for r in audit.runs)
        )


# --------------------------------------------------------------------------- #
# Selection helpers
# --------------------------------------------------------------------------- #
class SelectionHelperTests(unittest.TestCase):
    def _audit(self) -> AuditRun:
        runs = [
            make_run(strategy=Strategy.CHEAP, cheap_verdict=Verdict.CORRECT, premium_verdict=None),
            make_run(strategy=Strategy.PREMIUM, cheap_verdict=None, premium_verdict=Verdict.CORRECT),
        ]
        for threshold in (0.3, 0.5):
            cascade = make_run(escalated=False)
            cascade.threshold = threshold
            runs.append(cascade)
        return AuditRun(
            run_id="r",
            created_at="",
            provider="mock",
            task_set="s",
            n_tasks=1,
            seed=1,
            thresholds=(0.3, 0.5),
            config={},
            runs=runs,
        )

    def test_select_filters_by_strategy(self):
        audit = self._audit()
        self.assertEqual(len(audit.select(Strategy.CHEAP)), 1)
        self.assertEqual(len(audit.select(Strategy.PREMIUM)), 1)

    def test_select_cascade_filters_by_threshold(self):
        audit = self._audit()
        self.assertEqual(len(audit.select(Strategy.CASCADE)), 2)
        self.assertEqual(len(audit.select(Strategy.CASCADE, 0.3)), 1)
        self.assertEqual(audit.select(Strategy.CASCADE, 0.3)[0].threshold, 0.3)

    def test_strategies_present_is_stable_ordered(self):
        self.assertEqual(
            self._audit().strategies_present(),
            [Strategy.CHEAP, Strategy.PREMIUM, Strategy.CASCADE],
        )

    def test_group_by_threshold_puts_a_and_b_under_none(self):
        buckets = group_by_threshold(self._audit().runs)
        self.assertIn(None, buckets)
        self.assertEqual(len(buckets[None]), 2)
        self.assertEqual(sorted(k for k in buckets if k is not None), [0.3, 0.5])


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class EnumTests(unittest.TestCase):
    def test_strategy_values_are_the_policy_letters(self):
        self.assertEqual(Strategy.CHEAP.value, "A")
        self.assertEqual(Strategy.PREMIUM.value, "B")
        self.assertEqual(Strategy.CASCADE.value, "C")

    def test_strategy_has_human_titles(self):
        for strategy in Strategy:
            self.assertTrue(strategy.title)
            self.assertTrue(strategy.zh)

    def test_verdict_is_correct_only_for_correct(self):
        self.assertTrue(Verdict.CORRECT.is_correct)
        self.assertFalse(Verdict.INCORRECT.is_correct)
        self.assertFalse(Verdict.UNGRADED.is_correct)

    def test_escalation_reason_none_means_no_escalation(self):
        self.assertEqual(EscalationReason.NONE.value, "none")
        for reason in EscalationReason:
            self.assertTrue(reason.zh)


if __name__ == "__main__":
    unittest.main()
