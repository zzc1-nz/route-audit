"""Tests for ``routeaudit.config``: model catalogue and audit configuration.

Covers the validation rules the CLI depends on, the derived properties used for
budget pre-flight, and the price/scale plumbing shared with ``pricing``.
"""

from __future__ import annotations

import unittest

from routeaudit.config import (
    CATALOGUE,
    DEFAULT_CHEAP,
    DEFAULT_PREMIUM,
    DEFAULT_THRESHOLDS,
    SWEEP_THRESHOLDS,
    AuditConfig,
    ModelSpec,
    resolve_model,
    thresholds_from_csv,
)


class ModelSpecTests(unittest.TestCase):
    def test_p_correct_is_monotone_decreasing_in_difficulty(self):
        spec = CATALOGUE["gpt-4o-mini"]
        values = [spec.p_correct(d) for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_p_correct_stays_within_the_unit_interval(self):
        for spec in CATALOGUE.values():
            for difficulty in (0.0, 0.5, 1.0):
                self.assertGreaterEqual(spec.p_correct(difficulty), 0.0)
                self.assertLessEqual(spec.p_correct(difficulty), 1.0)

    def test_p_correct_clamps_out_of_range_difficulty(self):
        spec = CATALOGUE["gpt-4o"]
        self.assertAlmostEqual(spec.p_correct(-5.0), spec.p_correct(0.0))
        self.assertAlmostEqual(spec.p_correct(5.0), spec.p_correct(1.0))

    def test_premium_floor_is_above_cheap_floor(self):
        self.assertGreater(
            CATALOGUE["gpt-4o"].floor_quality, CATALOGUE["gpt-4o-mini"].floor_quality
        )

    def test_bad_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelSpec(name="m", provider="p", tier="gold", input_price_per_mtok=1.0,
                      output_price_per_mtok=1.0)

    def test_negative_price_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelSpec(name="m", provider="p", tier="cheap", input_price_per_mtok=-1.0,
                      output_price_per_mtok=1.0)

    def test_floor_above_quality_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelSpec(name="m", provider="p", tier="cheap", input_price_per_mtok=1.0,
                      output_price_per_mtok=1.0, quality=0.5, floor_quality=0.9)

    def test_round_trip(self):
        spec = CATALOGUE["claude-sonnet"]
        self.assertEqual(ModelSpec.from_dict(spec.to_dict()), spec)


class ResolveModelTests(unittest.TestCase):
    def test_resolves_a_catalogue_name(self):
        self.assertIs(resolve_model("gpt-4o"), CATALOGUE["gpt-4o"])

    def test_passes_a_spec_through(self):
        spec = CATALOGUE["deepseek-chat"]
        self.assertIs(resolve_model(spec), spec)

    def test_unknown_name_raises_with_a_helpful_message(self):
        with self.assertRaises(KeyError) as ctx:
            resolve_model("no-such-model")
        self.assertIn("no-such-model", str(ctx.exception))
        self.assertIn("gpt-4o", str(ctx.exception))


class AuditConfigValidationTests(unittest.TestCase):
    def test_defaults_match_the_documented_brief(self):
        config = AuditConfig()
        self.assertEqual(config.n_tasks, 200)
        self.assertEqual(config.thresholds, DEFAULT_THRESHOLDS)
        self.assertEqual(config.cheap_model, DEFAULT_CHEAP)
        self.assertEqual(config.premium_model, DEFAULT_PREMIUM)
        self.assertEqual(config.include_strategies, ("A", "B", "C"))

    def test_non_positive_n_tasks_is_rejected(self):
        with self.assertRaises(ValueError):
            AuditConfig(n_tasks=0)
        with self.assertRaises(ValueError):
            AuditConfig(n_tasks=-5)

    def test_empty_thresholds_is_rejected(self):
        with self.assertRaises(ValueError):
            AuditConfig(thresholds=())

    def test_thresholds_outside_unit_interval_are_rejected(self):
        with self.assertRaises(ValueError):
            AuditConfig(thresholds=(1.5,))
        with self.assertRaises(ValueError):
            AuditConfig(thresholds=(-0.1,))

    def test_non_positive_price_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            AuditConfig(price_scale=0)
        with self.assertRaises(ValueError):
            AuditConfig(price_scale=-1.0)

    def test_non_positive_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            AuditConfig(max_budget_usd=0)

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            AuditConfig(include_strategies=("A", "D"))

    def test_thresholds_are_coerced_to_floats(self):
        config = AuditConfig(thresholds=(0, 1))  # type: ignore[arg-type]
        self.assertEqual(config.thresholds, (0.0, 1.0))
        self.assertIsInstance(config.thresholds[0], float)


class AuditConfigDerivedTests(unittest.TestCase):
    def test_cheap_and_premium_resolve_to_catalogue_specs(self):
        config = AuditConfig()
        self.assertIs(config.cheap, CATALOGUE[DEFAULT_CHEAP])
        self.assertIs(config.premium, CATALOGUE[DEFAULT_PREMIUM])

    def test_primary_threshold_prefers_half(self):
        self.assertAlmostEqual(AuditConfig(thresholds=(0.3, 0.7)).primary_threshold, 0.3)
        self.assertAlmostEqual(AuditConfig(thresholds=(0.3, 0.5, 0.7)).primary_threshold, 0.5)

    def test_planned_calls_counts_one_call_per_task_for_a_and_b(self):
        config = AuditConfig(n_tasks=10, thresholds=(0.5,), include_strategies=("A", "B"))
        self.assertEqual(config.planned_calls, 20)

    def test_planned_calls_counts_two_per_threshold_for_cascade(self):
        config = AuditConfig(n_tasks=10, thresholds=(0.5, 0.7), include_strategies=("C",))
        # worst case: cheap + premium for each of the 2 thresholds
        self.assertEqual(config.planned_calls, 10 * 2 * 2)

    def test_planned_calls_full_matrix(self):
        config = AuditConfig(n_tasks=5, thresholds=(0.5, 0.7, 0.3))
        self.assertEqual(config.planned_calls, 5 * (1 + 1 + 2 * 3))


class AuditConfigPersistenceTests(unittest.TestCase):
    def test_round_trip(self):
        config = AuditConfig(
            n_tasks=17,
            seed=99,
            thresholds=(0.25, 0.75),
            price_scale=1.5,
            format_gate=False,
            include_strategies=("A", "C"),
        )
        self.assertEqual(AuditConfig.from_dict(config.to_dict()), config)

    def test_from_dict_applies_documented_defaults(self):
        config = AuditConfig.from_dict({})
        self.assertEqual(config.n_tasks, 200)
        self.assertEqual(config.thresholds, DEFAULT_THRESHOLDS)
        self.assertTrue(config.format_gate)


class ThresholdParsingTests(unittest.TestCase):
    def test_csv_is_sorted_and_deduplicated(self):
        self.assertEqual(thresholds_from_csv("0.7, 0.3, 0.5"), (0.3, 0.5, 0.7))
        self.assertEqual(thresholds_from_csv("0.5,0.5,0.5"), (0.5,))

    def test_whitespace_is_tolerated(self):
        self.assertEqual(thresholds_from_csv(" 0.3 , 0.5 "), (0.3, 0.5))

    def test_accepts_an_iterable(self):
        self.assertEqual(thresholds_from_csv([0.5, 0.1]), (0.1, 0.5))

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            thresholds_from_csv("")
        with self.assertRaises(ValueError):
            thresholds_from_csv(" , , ")

    def test_sweep_ladder_spans_the_unit_interval(self):
        self.assertEqual(SWEEP_THRESHOLDS, tuple(sorted(SWEEP_THRESHOLDS)))
        self.assertEqual(len(set(SWEEP_THRESHOLDS)), len(SWEEP_THRESHOLDS))
        self.assertGreater(SWEEP_THRESHOLDS[0], 0.0)
        self.assertLess(SWEEP_THRESHOLDS[-1], 1.0)
        self.assertIn(0.5, SWEEP_THRESHOLDS)


if __name__ == "__main__":
    unittest.main()
