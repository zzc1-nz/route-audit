"""Token -> USD conversion.

Kept separate from the accounting layer so that a reviewer can swap in a
different price list (or a negotiated discount) without touching the audit
methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import ModelSpec, resolve_model
from .types import Usage


@dataclass(frozen=True)
class PriceTable:
    """Maps a model name to a price, with an optional global multiplier."""

    models: Mapping[str, ModelSpec]
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def spec(self, model: str | ModelSpec) -> ModelSpec:
        if isinstance(model, ModelSpec):
            return model
        try:
            return self.models[model]
        except KeyError as exc:
            raise KeyError(f"model {model!r} not in price table") from exc

    def cost_usd(self, model: str | ModelSpec, usage: Usage) -> float:
        spec = self.spec(model)
        return (
            usage.prompt_tokens * spec.input_price_per_mtok
            + usage.completion_tokens * spec.output_price_per_mtok
        ) / 1_000_000.0 * self.scale

    def per_token(self, model: str | ModelSpec) -> tuple[float, float]:
        """(input, output) USD per token — handy for regret arithmetic."""
        spec = self.spec(model)
        return (
            spec.input_price_per_mtok / 1_000_000.0 * self.scale,
            spec.output_price_per_mtok / 1_000_000.0 * self.scale,
        )

    def blended_usd_per_mtok(self, model: str | ModelSpec, output_ratio: float = 0.3) -> float:
        """A single headline price for tables: weight inputs and outputs."""
        spec = self.spec(model)
        ratio = min(max(output_ratio, 0.0), 1.0)
        return (
            spec.input_price_per_mtok * (1.0 - ratio) + spec.output_price_per_mtok * ratio
        ) * self.scale

    def with_scale(self, scale: float) -> "PriceTable":
        return PriceTable(self.models, scale)


def default_price_table(scale: float = 1.0) -> PriceTable:
    from .config import CATALOGUE

    return PriceTable(CATALOGUE, scale)


def usage_from_dict(data: Mapping[str, int]) -> Usage:
    return Usage(int(data.get("prompt_tokens", 0)), int(data.get("completion_tokens", 0)))
