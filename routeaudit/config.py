"""Model catalogue, price table and audit configuration.

Prices are USD per 1M tokens and are *deliberately data, not constants*: the
audit's conclusions must be re-derivable for any price list, since the whole
point is to price routing mistakes rather than to hardcode one vendor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """One routable model.

    ``tier`` is what the router reasons about.  The accuracy-shape parameters
    (``quality``, ``floor_quality``, ``difficulty_gamma``) and
    ``overconfidence`` only matter for the offline simulator: they describe how
    the model's accuracy decays with task difficulty and how well its
    self-reported confidence tracks that accuracy.

    The accuracy curve is deliberately *interpretable* rather than a magic
    constant, because the audit's conclusions must be attributable:

        p_correct(difficulty) = floor_quality
                              + (quality - floor_quality)
                                * (1 - difficulty) ** difficulty_gamma

    * ``quality``          accuracy ceiling on near-trivial tasks
    * ``floor_quality``    accuracy floor on the hardest tasks
    * ``difficulty_gamma`` how fast it decays from one to the other
      (>1 decays faster early, <1 holds up longer)
    """

    name: str
    provider: str
    tier: str
    input_price_per_mtok: float
    output_price_per_mtok: float
    quality: float = 0.93
    floor_quality: float = 0.25
    difficulty_gamma: float = 1.1
    overconfidence: float = 0.10
    latency_ms_per_call: float = 400.0
    supports_confidence: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.tier not in ("cheap", "premium"):
            raise ValueError(f"bad tier for {self.name!r}: {self.tier!r}")
        if self.input_price_per_mtok < 0 or self.output_price_per_mtok < 0:
            raise ValueError(f"negative price for {self.name!r}")
        if not 0.0 <= self.floor_quality <= self.quality <= 1.0:
            raise ValueError(
                f"need 0 <= floor_quality <= quality <= 1 for {self.name!r}: "
                f"{self.floor_quality!r} / {self.quality!r}"
            )
        if self.difficulty_gamma <= 0:
            raise ValueError(f"difficulty_gamma must be positive for {self.name!r}")
        if not 0.0 <= self.overconfidence <= 1.0:
            raise ValueError(f"overconfidence out of [0,1] for {self.name!r}")

    def p_correct(self, difficulty: float) -> float:
        """Latent probability of a correct answer at a given difficulty."""
        d = min(max(float(difficulty), 0.0), 1.0)
        return self.floor_quality + (self.quality - self.floor_quality) * (1.0 - d) ** (
            self.difficulty_gamma
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "tier": self.tier,
            "input_price_per_mtok": self.input_price_per_mtok,
            "output_price_per_mtok": self.output_price_per_mtok,
            "quality": self.quality,
            "floor_quality": self.floor_quality,
            "difficulty_gamma": self.difficulty_gamma,
            "overconfidence": self.overconfidence,
            "latency_ms_per_call": self.latency_ms_per_call,
            "supports_confidence": self.supports_confidence,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSpec":
        return cls(
            name=str(data["name"]),
            provider=str(data.get("provider", "mock")),
            tier=str(data["tier"]),
            input_price_per_mtok=float(data["input_price_per_mtok"]),
            output_price_per_mtok=float(data["output_price_per_mtok"]),
            quality=float(data.get("quality", 0.93)),
            floor_quality=float(data.get("floor_quality", 0.25)),
            difficulty_gamma=float(data.get("difficulty_gamma", 1.1)),
            overconfidence=float(data.get("overconfidence", 0.10)),
            latency_ms_per_call=float(data.get("latency_ms_per_call", 400.0)),
            supports_confidence=bool(data.get("supports_confidence", True)),
            notes=str(data.get("notes", "")),
        )


# Public list prices, rounded, as of the model cards this sandbox was written
# against.  The audit never trusts them implicitly: `--price-scale` and
# `ModelSpec` overrides exist so a reader can re-run with their own contract.
#
# The accuracy-shape parameters are *calibration assumptions*, not published
# facts.  They are chosen to reproduce three behaviours that matter for routing
# economics and are stated in the README so they can be challenged:
#   * the cheap model is strong on easy traffic and decays fast with difficulty
#   * the premium model has a much higher floor and decays more slowly
#   * the cheap model is systematically overconfident (+0.10), the premium one
#     nearly calibrated (+0.02) — this is what makes the gate imperfect
CATALOGUE: dict[str, ModelSpec] = {
    # --- cheap tier ------------------------------------------------------- #
    "deepseek-chat": ModelSpec(
        name="deepseek-chat",
        provider="deepseek",
        tier="cheap",
        input_price_per_mtok=0.27,
        output_price_per_mtok=1.10,
        quality=0.91,
        floor_quality=0.22,
        difficulty_gamma=1.15,
        overconfidence=0.11,
        latency_ms_per_call=650.0,
        notes="cheap workhorse; strong on code, weaker on multi-step math",
    ),
    "gpt-4o-mini": ModelSpec(
        name="gpt-4o-mini",
        provider="openai",
        tier="cheap",
        input_price_per_mtok=0.15,
        output_price_per_mtok=0.60,
        quality=0.93,
        floor_quality=0.25,
        difficulty_gamma=1.10,
        overconfidence=0.10,
        latency_ms_per_call=520.0,
        notes="reference cheap model in the CostGuard write-up",
    ),
    # --- premium tier ----------------------------------------------------- #
    "gpt-4o": ModelSpec(
        name="gpt-4o",
        provider="openai",
        tier="premium",
        input_price_per_mtok=2.50,
        output_price_per_mtok=10.00,
        quality=0.985,
        floor_quality=0.62,
        difficulty_gamma=0.90,
        overconfidence=0.02,
        latency_ms_per_call=1100.0,
        notes="premium baseline",
    ),
    "claude-sonnet": ModelSpec(
        name="claude-sonnet",
        provider="anthropic",
        tier="premium",
        input_price_per_mtok=3.00,
        output_price_per_mtok=15.00,
        quality=0.99,
        floor_quality=0.66,
        difficulty_gamma=0.85,
        overconfidence=0.015,
        latency_ms_per_call=1250.0,
        notes="premium baseline with a slightly better calibration",
    ),
}

DEFAULT_CHEAP = "gpt-4o-mini"
DEFAULT_PREMIUM = "gpt-4o"

#: Threshold ladder used by the sensitivity sweep.  ``0.5`` is the operating
#: point; ``0.7`` and ``0.3`` are the two comparison points required by the
#: brief, and ``0.9``/``0.1`` are included as the endpoints of the curve.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.5, 0.7, 0.3)
SWEEP_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def resolve_model(name_or_spec: str | ModelSpec) -> ModelSpec:
    """Look a model up in the catalogue, or pass a custom spec straight through."""
    if isinstance(name_or_spec, ModelSpec):
        return name_or_spec
    try:
        return CATALOGUE[name_or_spec]
    except KeyError as exc:
        known = ", ".join(sorted(CATALOGUE))
        raise KeyError(f"unknown model {name_or_spec!r}; known: {known}") from exc


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AuditConfig:
    """Everything that defines one audit run.

    The defaults match the brief: 200 tasks x 3 strategies x 3 thresholds.
    """

    task_set: str = "builtin-mixed"
    n_tasks: int = 200
    seed: int = 20260101
    cheap_model: str = DEFAULT_CHEAP
    premium_model: str = DEFAULT_PREMIUM
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    max_workers: int = 8
    price_scale: float = 1.0
    #: Hard cap on *simulated* spend; the audit aborts rather than silently
    #: running a 100x larger sweep than intended (CostGuard-style guardrail).
    max_budget_usd: float = 25.0
    #: Escalate when the cheap answer fails structural checks even if confidence
    #: is high (format gate).  Disable to isolate the confidence gate.
    format_gate: bool = True
    #: Cache identical (task, model, prompt) calls within a run so replaying a
    #: cascade does not re-pay for the same cheap completion.
    enable_cache: bool = True
    #: Persist per-attempt token ledgers (off by default? no: it is the point).
    keep_attempts: bool = True
    include_strategies: tuple[str, ...] = ("A", "B", "C")

    def __post_init__(self) -> None:
        if self.n_tasks <= 0:
            raise ValueError("n_tasks must be positive")
        if not self.thresholds:
            raise ValueError("at least one cascade threshold is required")
        for threshold in self.thresholds:
            if not 0.0 <= float(threshold) <= 1.0:
                raise ValueError(f"threshold out of [0,1]: {threshold!r}")
        if self.price_scale <= 0:
            raise ValueError("price_scale must be positive")
        if self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive")
        unknown = set(self.include_strategies) - {"A", "B", "C"}
        if unknown:
            raise ValueError(f"unknown strategies: {sorted(unknown)}")
        self.thresholds = tuple(float(t) for t in self.thresholds)
        self.include_strategies = tuple(self.include_strategies)

    @property
    def cheap(self) -> ModelSpec:
        return resolve_model(self.cheap_model)

    @property
    def premium(self) -> ModelSpec:
        return resolve_model(self.premium_model)

    @property
    def primary_threshold(self) -> float:
        """The operating point used for headline numbers (0.5 by default)."""
        return 0.5 if 0.5 in self.thresholds else self.thresholds[0]

    @property
    def planned_calls(self) -> int:
        """Rough upper bound on provider calls, used for budget pre-flight."""
        per_task = 0
        if "A" in self.include_strategies:
            per_task += 1
        if "B" in self.include_strategies:
            per_task += 1
        if "C" in self.include_strategies:
            per_task += 2 * len(self.thresholds)  # worst case: always escalates
        return self.n_tasks * per_task

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_set": self.task_set,
            "n_tasks": self.n_tasks,
            "seed": self.seed,
            "cheap_model": self.cheap_model,
            "premium_model": self.premium_model,
            "thresholds": list(self.thresholds),
            "max_workers": self.max_workers,
            "price_scale": self.price_scale,
            "max_budget_usd": self.max_budget_usd,
            "format_gate": self.format_gate,
            "enable_cache": self.enable_cache,
            "keep_attempts": self.keep_attempts,
            "include_strategies": list(self.include_strategies),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditConfig":
        return cls(
            task_set=str(data.get("task_set", "builtin-mixed")),
            n_tasks=int(data.get("n_tasks", 200)),
            seed=int(data.get("seed", 20260101)),
            cheap_model=str(data.get("cheap_model", DEFAULT_CHEAP)),
            premium_model=str(data.get("premium_model", DEFAULT_PREMIUM)),
            thresholds=tuple(float(t) for t in data.get("thresholds", DEFAULT_THRESHOLDS)),
            max_workers=int(data.get("max_workers", 8)),
            price_scale=float(data.get("price_scale", 1.0)),
            max_budget_usd=float(data.get("max_budget_usd", 25.0)),
            format_gate=bool(data.get("format_gate", True)),
            enable_cache=bool(data.get("enable_cache", True)),
            keep_attempts=bool(data.get("keep_attempts", True)),
            include_strategies=tuple(data.get("include_strategies", ("A", "B", "C"))),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "AuditConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def thresholds_from_csv(raw: str | Iterable[float]) -> tuple[float, ...]:
    """Parse ``"0.3,0.5,0.7"`` into a validated threshold tuple."""
    if isinstance(raw, str):
        parts = [p for p in raw.replace(" ", "").split(",") if p]
    else:
        parts = list(raw)
    values = tuple(sorted({round(float(p), 4) for p in parts}))
    if not values:
        raise ValueError("no thresholds given")
    return values
