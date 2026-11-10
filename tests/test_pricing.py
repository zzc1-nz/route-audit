"""Tests for ``routeaudit.pricing``: token -> USD conversion.

Focus areas (task-2 item 4):
  * ``cost_usd`` arithmetic against the published per-Mtok prices
  * ``scale`` multiplies cost linearly
  * input vs output prices are applied to the right token bucket
  * unknown models / invalid scales fail loudly
"""

from __future__ import annotations

import unittest

from routeaudit.config import CATALOGUE, ModelSpec
from routeaudit.pricing import PriceTable, default_price_table, usage_from_dict
from routeaudit.types import Usage


def make_spec(
    name: str = "unit-model",
    tier: str = "cheap",
    input_price: float = 2.0,
    output_price: float = 8.0,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        provider="unit",
        tier=tier,
        input_price_per_mtok=input_price,
        output_price_per_mtok=output_price,
    )


class CostUsdTests(unittest.TestCase):
    def test_cost_is_prompt_and_completion_priced_separately(self):
        # 1M prompt tokens at $2/Mtok + 1M completion at $8/Mtok = $10
        table = PriceTable({"m": make_spec(input_price=2.0, output_price=8.0)})
        self.assertAlmostEqual(table.cost_usd("m", Usage(1_000_000, 1_000_000)), 10.0)

    def test_zero_usage_costs_nothing(self):
        table = PriceTable({"m": make_spec()})
        self.assertEqual(table.cost_usd("m", Usage.zero()), 0.0)

    def test_half_million_prompt_tokens_is_half_the_price(self):
        table = PriceTable({"m": make_spec(input_price=2.0, output_price=8.0)})
        self.assertAlmostEqual(table.cost_usd("m", Usage(500_000, 0)), 1.0)

    def test_completion_tokens_are_charged_at_the_output_rate(self):
        table = PriceTable({"m": make_spec(input_price=2.0, output_price=8.0)})
        self.assertAlmostEqual(table.cost_usd("m", Usage(0, 500_000)), 4.0)

    def test_cost_is_linear_in_token_count(self):
        table = PriceTable({"m": make_spec(input_price=2.0, output_price=8.0)})
        single = table.cost_usd("m", Usage(1000, 500))
        triple = table.cost_usd("m", Usage(3000, 1500))
        self.assertAlmostEqual(triple, single * 3)

    def test_published_catalogue_price_is_used(self):
        # gpt-4o-mini is $0.15 in / $0.60 out per Mtok
        table = default_price_table()
        cost = table.cost_usd("gpt-4o-mini", Usage(1_000_000, 0))
        self.assertAlmostEqual(cost, 0.15)
        cost = table.cost_usd("gpt-4o-mini", Usage(0, 1_000_000))
        self.assertAlmostEqual(cost, 0.60)

    def test_premium_costs_more_than_cheap_for_the_same_usage(self):
        table = default_price_table()
        usage = Usage(100_000, 50_000)
        self.assertGreater(
            table.cost_usd("gpt-4o", usage), table.cost_usd("gpt-4o-mini", usage)
        )

    def test_accepts_a_model_spec_directly(self):
        table = PriceTable({})
        spec = make_spec(input_price=2.0, output_price=8.0)
        self.assertAlmostEqual(table.cost_usd(spec, Usage(1_000_000, 0)), 2.0)

    def test_unknown_model_raises_key_error(self):
        table = default_price_table()
        with self.assertRaises(KeyError):
            table.cost_usd("no-such-model", Usage(1, 1))


class ScaleTests(unittest.TestCase):
    def test_scale_multiplies_cost(self):
        spec = make_spec(input_price=2.0, output_price=8.0)
        usage = Usage(1_000_000, 0)
        base = PriceTable({"m": spec}, scale=1.0).cost_usd("m", usage)
        doubled = PriceTable({"m": spec}, scale=2.0).cost_usd("m", usage)
        self.assertAlmostEqual(base, 2.0)
        self.assertAlmostEqual(doubled, 4.0)

    def test_scale_below_one_is_a_discount(self):
        spec = make_spec(input_price=2.0, output_price=8.0)
        usage = Usage(1_000_000, 0)
        discounted = PriceTable({"m": spec}, scale=0.5).cost_usd("m", usage)
        self.assertAlmostEqual(discounted, 1.0)

    def test_default_price_table_honours_scale(self):
        usage = Usage(1_000_000, 0)
        self.assertAlmostEqual(
            default_price_table(scale=3.0).cost_usd("gpt-4o-mini", usage), 0.45
        )

    def test_with_scale_returns_a_new_table(self):
        spec = make_spec(input_price=2.0, output_price=8.0)
        table = PriceTable({"m": spec}, scale=1.0)
        scaled = table.with_scale(4.0)
        self.assertEqual(table.scale, 1.0)  # original untouched
        self.assertEqual(scaled.scale, 4.0)
        self.assertAlmostEqual(
            scaled.cost_usd("m", Usage(1_000_000, 0)), 8.0
        )

    def test_non_positive_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            PriceTable({}, scale=0.0)
        with self.assertRaises(ValueError):
            PriceTable({}, scale=-1.0)


class PerTokenAndBlendedTests(unittest.TestCase):
    def test_per_token_matches_cost_of_one_token(self):
        table = PriceTable({"m": make_spec(input_price=2.0, output_price=8.0)})
        input_rate, output_rate = table.per_token("m")
        self.assertAlmostEqual(input_rate, 2.0 / 1_000_000)
        self.assertAlmostEqual(output_rate, 8.0 / 1_000_000)
        self.assertAlmostEqual(table.cost_usd("m", Usage(1, 0)), input_rate)
        self.assertAlmostEqual(table.cost_usd("m", Usage(0, 1)), output_rate)

    def test_per_token_applies_scale(self):
        table = PriceTable({"m": make_spec(input_price=2.0, output_price=8.0)}, scale=10.0)
        input_rate, _ = table.per_token("m")
        self.assertAlmostEqual(input_rate, 20.0 / 1_000_000)

    def test_blended_price_weights_input_and_output(self):
        table = PriceTable({"m": make_spec(input_price=2.0, output_price=8.0)})
        # 70% input, 30% output by default
        self.assertAlmostEqual(table.blended_usd_per_mtok("m", 0.3), 2.0 * 0.7 + 8.0 * 0.3)

    def test_blended_ratio_is_clamped_to_unit_interval(self):
        table = PriceTable({"m": make_spec(input_price=2.0, output_price=8.0)})
        self.assertAlmostEqual(table.blended_usd_per_mtok("m", -5.0), 2.0)
        self.assertAlmostEqual(table.blended_usd_per_mtok("m", 5.0), 8.0)


class CatalogueIntegrityTests(unittest.TestCase):
    def test_every_catalogue_model_is_priceable(self):
        table = default_price_table()
        for name in CATALOGUE:
            self.assertGreaterEqual(table.cost_usd(name, Usage(1000, 1000)), 0.0)

    def test_catalogue_tiers_are_only_cheap_or_premium(self):
        for name, spec in CATALOGUE.items():
            self.assertIn(spec.tier, ("cheap", "premium"), msg=name)

    def test_premium_models_are_priced_above_cheap_models(self):
        cheap = [s for s in CATALOGUE.values() if s.tier == "cheap"]
        premium = [s for s in CATALOGUE.values() if s.tier == "premium"]
        self.assertTrue(cheap and premium)
        self.assertGreater(
            min(s.input_price_per_mtok for s in premium),
            max(s.input_price_per_mtok for s in cheap),
        )

    def test_usage_from_dict(self):
        self.assertEqual(usage_from_dict({"prompt_tokens": 3, "completion_tokens": 4}), Usage(3, 4))
        self.assertEqual(usage_from_dict({}), Usage.zero())


if __name__ == "__main__":
    unittest.main()
