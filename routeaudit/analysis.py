"""Routing-error cost analysis.

The report layer is a *renderer*; every number it prints is produced here.  The
analysis answers four questions:

1. **What did each configuration cost, in tokens and dollars, and what did it
   buy in accuracy?**                          -> :func:`compare`
2. **How much of the cascade's spend was routing error?**  -> :func:`routing_error_analysis`
   Split into over-escalation (premium tokens spent on tasks the cheap model
   already solved) and over-downgrade (cheap tokens spent on tasks the cheap
   model could not solve, i.e. tokens that bought nothing before the necessary
   upgrade).
3. **How does the threshold move cost and accuracy?**      -> :func:`threshold_sensitivity`
4. **How far from the best possible routing is the cascade?** -> :func:`regret_analysis`
   The oracle knows the labels and pays the cheap model only when it would have
   been right; the gap between oracle cost and cascade cost is the irreducible
   price of not having labels online — and the ceiling on what better routing
   could ever recover.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from .accounting import ConfigSummary, attribute_all, cascade_key as cascade_key_for, summarise, summarise_audit
from .types import AuditRun, EscalationReason, Strategy, TaskRun, Verdict


# --------------------------------------------------------------------------- #
# 1. Configuration comparison
# --------------------------------------------------------------------------- #
@dataclass
class Comparison:
    """All configurations of one audit, plus the relative deltas that matter."""

    summaries: dict[str, ConfigSummary]
    cheap_key: str
    premium_key: str
    cascade_key: str

    @property
    def cheap(self) -> ConfigSummary:
        return self.summaries[self.cheap_key]

    @property
    def premium(self) -> ConfigSummary:
        return self.summaries[self.premium_key]

    @property
    def cascade(self) -> ConfigSummary:
        return self.summaries[self.cascade_key]

    def ordered(self) -> list[ConfigSummary]:
        """Stable display order: A, B, then cascade thresholds ascending."""
        out = []
        for key in (self.cheap_key, self.premium_key):
            if key in self.summaries:
                out.append(self.summaries[key])
        for key in sorted(
            (k for k in self.summaries if k.startswith("C@")),
            key=lambda k: float(k.split("@")[1]),
        ):
            out.append(self.summaries[key])
        return out

    # -- headline ratios ---------------------------------------------------- #
    @property
    def cascade_savings_vs_premium(self) -> float:
        """Fraction of premium spend the cascade avoids (the "85% savings" claim)."""
        premium = self.premium.total_cost_usd
        if premium <= 0:
            return 0.0
        return 1.0 - self.cascade.total_cost_usd / premium

    @property
    def cascade_token_savings_vs_premium(self) -> float:
        premium = self.premium.total_tokens
        if premium <= 0:
            return 0.0
        return 1.0 - self.cascade.total_tokens / premium

    @property
    def cascade_accuracy_loss_vs_premium(self) -> float:
        return self.premium.accuracy - self.cascade.accuracy

    @property
    def cascade_cost_multiple_vs_cheap(self) -> float:
        cheap = self.cheap.total_cost_usd
        return self.cascade.total_cost_usd / cheap if cheap > 0 else float("inf")

    @property
    def cascade_accuracy_gain_vs_cheap(self) -> float:
        return self.cascade.accuracy - self.cheap.accuracy

    def to_dict(self) -> dict[str, Any]:
        return {
            "configurations": [s.to_dict() for s in self.ordered()],
            "cascade_savings_vs_premium": round(self.cascade_savings_vs_premium, 6),
            "cascade_token_savings_vs_premium": round(self.cascade_token_savings_vs_premium, 6),
            "cascade_accuracy_loss_vs_premium": round(self.cascade_accuracy_loss_vs_premium, 6),
            "cascade_cost_multiple_vs_cheap": round(self.cascade_cost_multiple_vs_cheap, 4),
            "cascade_accuracy_gain_vs_cheap": round(self.cascade_accuracy_gain_vs_cheap, 6),
        }


def compare(audit: AuditRun, *, threshold: float | None = None) -> Comparison:
    """Build the configuration comparison for one audit.

    ``threshold`` selects which cascade configuration is the "headline" cascade;
    it defaults to 0.5 when present, else the lowest configured threshold.
    """
    summaries = summarise_audit(audit)
    if not summaries:
        raise ValueError("audit has no runs to compare")
    cheap_key = Strategy.CHEAP.value
    premium_key = Strategy.PREMIUM.value
    for key in (cheap_key, premium_key):
        if key not in summaries:
            raise ValueError(f"audit is missing strategy {key}; run with --strategies A,B,C")
    cascade_keys = sorted(
        (k for k in summaries if k.startswith("C@")),
        key=lambda k: float(k.split("@")[1]),
    )
    if not cascade_keys:
        raise ValueError("audit has no cascade configuration to compare")
    if threshold is None:
        threshold = 0.5 if any(abs(float(k.split("@")[1]) - 0.5) < 1e-9 for k in cascade_keys) else float(
            cascade_keys[0].split("@")[1]
        )
    cascade_key = cascade_key_for(threshold)
    if cascade_key not in summaries:
        raise ValueError(
            f"no cascade configuration at threshold {threshold}; available: "
            + ", ".join(k.split('@')[1] for k in cascade_keys)
        )
    return Comparison(
        summaries=summaries,
        cheap_key=cheap_key,
        premium_key=premium_key,
        cascade_key=cascade_key,
    )


# --------------------------------------------------------------------------- #
# 2. Routing-error attribution
# --------------------------------------------------------------------------- #
@dataclass
class ErrorBucket:
    """One direction of routing error, priced in tokens and dollars."""

    name: str
    n_tasks: int
    tokens: int
    cost_usd: float
    token_share: float
    cost_share: float
    task_share: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_tasks": self.n_tasks,
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 8),
            "token_share": round(self.token_share, 6),
            "cost_share": round(self.cost_share, 6),
            "task_share": round(self.task_share, 6),
        }


@dataclass
class RoutingErrorAnalysis:
    """Item 3 of the brief: what did routing mistakes cost?

    Definitions (stated explicitly because the two "rates" in the brief are
    easy to conflate):

    over-escalation
        ``cheap_correct and escalated``.  The premium call was pure waste; the
        cheap answer was already right.  Reported both as a share of
        *escalations* (how often the router escalates needlessly) and as a share
        of *tasks*.
    over-downgrade
        ``cheap_incorrect and escalated``.  The router started too low, paid for
        a cheap attempt that produced nothing usable, then paid again.  This is
        the "double token spend" case: the cheap tokens in these tasks are pure
        overhead on top of the premium call.
    missed escalation
        ``cheap_incorrect and not escalated``.  The router trusted a wrong
        answer.  Costs no extra tokens but ships a wrong result — the failure
        the "savings" framing hides.
    """

    threshold: float
    n_tasks: int
    total_tokens: int
    total_cost_usd: float
    over_escalation: ErrorBucket
    over_downgrade: ErrorBucket
    missed_escalation: ErrorBucket
    n_escalated: int
    n_escalation_reasons: Mapping[str, int]
    cheap_tokens: int
    escalation_tokens: int
    wasted_escalation_tokens: int
    wasted_downgrade_tokens: int
    wasted_escalation_cost_usd: float
    wasted_downgrade_cost_usd: float
    escalation_cost_ratio: float
    wasted_escalation_tasks: list[str] = field(default_factory=list)
    wasted_downgrade_tasks: list[str] = field(default_factory=list)
    missed_escalation_tasks: list[str] = field(default_factory=list)
    #: Escalated, cheap was right, but the premium answer was wrong anyway.
    n_harmful_escalations: int = 0
    harmful_escalation_tasks: list[str] = field(default_factory=list)

    # -- headline shares ---------------------------------------------------- #
    @property
    def wasted_token_share(self) -> float:
        if not self.total_tokens:
            return 0.0
        return (self.wasted_escalation_tokens + self.wasted_downgrade_tokens) / self.total_tokens

    @property
    def wasted_cost_share(self) -> float:
        if not self.total_cost_usd:
            return 0.0
        return (
            self.wasted_escalation_cost_usd + self.wasted_downgrade_cost_usd
        ) / self.total_cost_usd

    @property
    def over_escalation_rate(self) -> float:
        return self.over_escalation.n_tasks / self.n_escalated if self.n_escalated else 0.0

    @property
    def over_downgrade_rate(self) -> float:
        return self.over_downgrade.n_tasks / self.n_tasks if self.n_tasks else 0.0

    @property
    def missed_escalation_rate(self) -> float:
        return self.missed_escalation.n_tasks / self.n_tasks if self.n_tasks else 0.0

    @property
    def escalation_precision(self) -> float:
        """Share of escalations that were actually necessary."""
        if not self.n_escalated:
            return 0.0
        return (self.n_escalated - self.over_escalation.n_tasks) / self.n_escalated

    @property
    def wasted_cost_usd(self) -> float:
        return self.wasted_escalation_cost_usd + self.wasted_downgrade_cost_usd

    @property
    def useful_tokens(self) -> int:
        return self.total_tokens - self.wasted_escalation_tokens - self.wasted_downgrade_tokens

    @property
    def harmful_escalation_rate(self) -> float:
        """Share of tasks where escalating destroyed a correct cheap answer.

        The fourth failure mode, and the one every cost metric is blind to:
        escalation is usually framed as "buy accuracy with money", but when the
        premium answer is wrong on a task the cheap model got right, the router
        pays *and* loses accuracy.  A cascade's accuracy is therefore not
        monotone in the threshold.
        """
        return self.n_harmful_escalations / self.n_tasks if self.n_tasks else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "n_tasks": self.n_tasks,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "cheap_tokens": self.cheap_tokens,
            "escalation_tokens": self.escalation_tokens,
            "n_escalated": self.n_escalated,
            "escalation_cost_ratio": round(self.escalation_cost_ratio, 4),
            "over_escalation": self.over_escalation.to_dict(),
            "over_downgrade": self.over_downgrade.to_dict(),
            "missed_escalation": self.missed_escalation.to_dict(),
            "n_harmful_escalations": self.n_harmful_escalations,
            "harmful_escalation_rate": round(self.harmful_escalation_rate, 6),
            "over_escalation_rate": round(self.over_escalation_rate, 6),
            "over_downgrade_rate": round(self.over_downgrade_rate, 6),
            "missed_escalation_rate": round(self.missed_escalation_rate, 6),
            "escalation_precision": round(self.escalation_precision, 6),
            "wasted_escalation_tokens": self.wasted_escalation_tokens,
            "wasted_downgrade_tokens": self.wasted_downgrade_tokens,
            "wasted_token_share": round(self.wasted_token_share, 6),
            "wasted_cost_usd": round(self.wasted_cost_usd, 8),
            "wasted_cost_share": round(self.wasted_cost_share, 6),
            "useful_tokens": self.useful_tokens,
            "escalation_reasons": dict(self.n_escalation_reasons),
            "wasted_escalation_task_ids": self.wasted_escalation_tasks[:50],
            "wasted_downgrade_task_ids": self.wasted_downgrade_tasks[:50],
            "missed_escalation_task_ids": self.missed_escalation_tasks[:50],
        }


def routing_error_analysis(
    audit: AuditRun,
    threshold: float | None = None,
) -> RoutingErrorAnalysis:
    """Attribute one cascade configuration's spend to routing mistakes."""
    comparison = compare(audit, threshold=threshold)
    cascade = comparison.cascade
    runs = audit.select(Strategy.CASCADE, cascade.threshold)
    if not runs:
        raise ValueError("no cascade runs at the requested threshold")

    ledgers = attribute_all(runs)
    total_tokens = cascade.total_tokens
    total_cost = cascade.total_cost_usd
    n_tasks = cascade.n_tasks

    wasted_escalation_tokens = sum(l.wasted_escalation_tokens for l in ledgers)
    wasted_downgrade_tokens = sum(l.wasted_downgrade_tokens for l in ledgers)
    wasted_escalation_cost = sum(l.wasted_escalation_cost_usd for l in ledgers)
    wasted_downgrade_cost = sum(l.wasted_downgrade_cost_usd for l in ledgers)

    def bucket(name: str, n: int, tokens: int, cost: float) -> ErrorBucket:
        return ErrorBucket(
            name=name,
            n_tasks=n,
            tokens=tokens,
            cost_usd=cost,
            token_share=(tokens / total_tokens) if total_tokens else 0.0,
            cost_share=(cost / total_cost) if total_cost else 0.0,
            task_share=(n / n_tasks) if n_tasks else 0.0,
        )

    # Premium price / cheap price, blended — how many cheap calls one escalation
    # is "worth".  Quoted in the report so the reader can sanity-check the
    # economics rather than take the percentages on faith.
    cheap_spec = audit.stats.get("cheap_model") or {}
    premium_spec = audit.stats.get("premium_model") or {}
    escalation_ratio = _price_ratio(cheap_spec, premium_spec)

    reasons: dict[str, int] = {}
    for run in runs:
        if run.escalated:
            reasons[run.escalation_reason.value] = reasons.get(run.escalation_reason.value, 0) + 1

    return RoutingErrorAnalysis(
        threshold=cascade.threshold,  # type: ignore[arg-type]
        n_tasks=n_tasks,
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        over_escalation=bucket(
            "over_escalation",
            cascade.n_wasted_escalation,
            wasted_escalation_tokens,
            wasted_escalation_cost,
        ),
        over_downgrade=bucket(
            "over_downgrade",
            cascade.n_necessary_escalation,
            wasted_downgrade_tokens,
            wasted_downgrade_cost,
        ),
        missed_escalation=bucket(
            "missed_escalation",
            cascade.n_missed_escalation,
            sum(l.silent_error_tokens for l in ledgers),
            0.0,
        ),
        n_escalated=cascade.n_escalated,
        n_escalation_reasons=dict(sorted(reasons.items())),
        cheap_tokens=cascade.cheap_tokens,
        escalation_tokens=cascade.escalation_tokens,
        wasted_escalation_tokens=wasted_escalation_tokens,
        wasted_downgrade_tokens=wasted_downgrade_tokens,
        wasted_escalation_cost_usd=wasted_escalation_cost,
        wasted_downgrade_cost_usd=wasted_downgrade_cost,
        escalation_cost_ratio=escalation_ratio,
        wasted_escalation_tasks=[l.task_id for l in ledgers if l.wasted_escalation_tokens],
        wasted_downgrade_tasks=[l.task_id for l in ledgers if l.wasted_downgrade_tokens],
        missed_escalation_tasks=[l.task_id for l in ledgers if l.missed_escalation],
        n_harmful_escalations=sum(1 for l in ledgers if l.harmful_escalation),
        harmful_escalation_tasks=[l.task_id for l in ledgers if l.harmful_escalation],
    )


def _price_ratio(cheap: Mapping[str, Any], premium: Mapping[str, Any]) -> float:
    """Blended premium/cheap price ratio (assumes a 70/30 input/output mix)."""
    def blended(spec: Mapping[str, Any]) -> float:
        return float(spec.get("input_price_per_mtok", 0.0)) * 0.7 + float(
            spec.get("output_price_per_mtok", 0.0)
        ) * 0.3

    cheap_price = blended(cheap)
    premium_price = blended(premium)
    if cheap_price <= 0:
        return float("inf")
    return premium_price / cheap_price


# --------------------------------------------------------------------------- #
# 3. Threshold sensitivity
# --------------------------------------------------------------------------- #
@dataclass
class ThresholdPoint:
    """One threshold on the sensitivity curve."""

    threshold: float
    accuracy: float
    total_tokens: int
    total_cost_usd: float
    escalation_rate: float
    over_escalation_rate: float
    over_downgrade_rate: float
    missed_escalation_rate: float
    wasted_token_share: float
    wasted_cost_share: float
    tokens_per_correct: float
    cost_per_correct_usd: float
    n_escalated: int
    escalation_precision: float
    #: Marginal cost of the next accuracy point vs the previous threshold.
    marginal_cost_per_point: float | None = None
    marginal_tokens_per_point: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "accuracy": round(self.accuracy, 6),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "escalation_rate": round(self.escalation_rate, 6),
            "over_escalation_rate": round(self.over_escalation_rate, 6),
            "over_downgrade_rate": round(self.over_downgrade_rate, 6),
            "missed_escalation_rate": round(self.missed_escalation_rate, 6),
            "wasted_token_share": round(self.wasted_token_share, 6),
            "wasted_cost_share": round(self.wasted_cost_share, 6),
            "tokens_per_correct": round(self.tokens_per_correct, 2),
            "cost_per_correct_usd": round(self.cost_per_correct_usd, 8),
            "n_escalated": self.n_escalated,
            "escalation_precision": round(self.escalation_precision, 6),
            "marginal_cost_per_point": None
            if self.marginal_cost_per_point is None
            else round(self.marginal_cost_per_point, 8),
            "marginal_tokens_per_point": None
            if self.marginal_tokens_per_point is None
            else round(self.marginal_tokens_per_point, 2),
        }


@dataclass
class ThresholdSensitivity:
    """Item 3's second half: how cost and accuracy move with the threshold."""

    points: list[ThresholdPoint]
    cheap_accuracy: float
    cheap_cost_usd: float
    cheap_tokens: int
    premium_accuracy: float
    premium_cost_usd: float
    premium_tokens: int
    operating_threshold: float

    @property
    def best_accuracy_point(self) -> ThresholdPoint:
        return max(self.points, key=lambda p: (p.accuracy, -p.total_cost_usd))

    @property
    def cheapest_point(self) -> ThresholdPoint:
        return min(self.points, key=lambda p: (p.total_cost_usd, -p.accuracy))

    def point(self, threshold: float) -> ThresholdPoint:
        for p in self.points:
            if abs(p.threshold - threshold) < 1e-9:
                return p
        raise KeyError(f"threshold {threshold} not in sweep; have {[p.threshold for p in self.points]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "cheap_accuracy": round(self.cheap_accuracy, 6),
            "cheap_cost_usd": round(self.cheap_cost_usd, 8),
            "cheap_tokens": self.cheap_tokens,
            "premium_accuracy": round(self.premium_accuracy, 6),
            "premium_cost_usd": round(self.premium_cost_usd, 8),
            "premium_tokens": self.premium_tokens,
            "operating_threshold": self.operating_threshold,
            "best_accuracy_threshold": self.best_accuracy_point.threshold,
            "cheapest_threshold": self.cheapest_point.threshold,
        }


def threshold_sensitivity(audit: AuditRun) -> ThresholdSensitivity:
    """Sweep every cascade threshold in the audit and compute marginal costs."""
    summaries = summarise_audit(audit)
    cheap = summaries[Strategy.CHEAP.value]
    premium = summaries[Strategy.PREMIUM.value]

    thresholds = sorted(
        float(k.split("@")[1]) for k in summaries if k.startswith("C@")
    )
    if not thresholds:
        raise ValueError("audit has no cascade thresholds to sweep")

    points: list[ThresholdPoint] = []
    for threshold in thresholds:
        summary = summaries[cascade_key_for(threshold)]
        points.append(
            ThresholdPoint(
                threshold=threshold,
                accuracy=summary.accuracy,
                total_tokens=summary.total_tokens,
                total_cost_usd=summary.total_cost_usd,
                escalation_rate=summary.escalation_rate,
                over_escalation_rate=summary.over_escalation_rate,
                over_downgrade_rate=summary.over_downgrade_rate,
                missed_escalation_rate=summary.missed_escalation_rate,
                wasted_token_share=summary.wasted_token_share,
                wasted_cost_share=summary.wasted_cost_share,
                tokens_per_correct=summary.tokens_per_correct,
                cost_per_correct_usd=summary.cost_per_correct,
                n_escalated=summary.n_escalated,
                escalation_precision=(
                    (summary.n_escalated - summary.n_wasted_escalation) / summary.n_escalated
                    if summary.n_escalated
                    else 0.0
                ),
            )
        )

    # Marginal economics: what does the next accuracy point cost?
    for prev, cur in zip(points, points[1:]):
        d_acc = cur.accuracy - prev.accuracy
        if abs(d_acc) < 1e-12:
            cur.marginal_cost_per_point = None
            cur.marginal_tokens_per_point = None
        else:
            cur.marginal_cost_per_point = (cur.total_cost_usd - prev.total_cost_usd) / d_acc
            cur.marginal_tokens_per_point = (cur.total_tokens - prev.total_tokens) / d_acc

    operating = 0.5 if any(abs(t - 0.5) < 1e-9 for t in thresholds) else thresholds[0]
    return ThresholdSensitivity(
        points=points,
        cheap_accuracy=cheap.accuracy,
        cheap_cost_usd=cheap.total_cost_usd,
        cheap_tokens=cheap.total_tokens,
        premium_accuracy=premium.accuracy,
        premium_cost_usd=premium.total_cost_usd,
        premium_tokens=premium.total_tokens,
        operating_threshold=operating,
    )


# --------------------------------------------------------------------------- #
# 4. Regret vs the label-aware oracle
# --------------------------------------------------------------------------- #
@dataclass
class RegretAnalysis:
    """How close is the cascade to the best routing a label-aware oracle could do?

    The oracle must be *accuracy-matched*, otherwise the comparison is
    meaningless: a router that always picks the cheap model has zero cost and
    terrible accuracy, and a router that always escalates every wrong cheap
    answer has great accuracy and premium cost.  Neither is a fair benchmark.

    So the oracle answers a well-posed question:

        *What is the minimum cost of reaching the cascade's own accuracy, given
        that you may route each task individually with full knowledge of labels?*

    Construction: start from "cheap for everyone", then escalate the tasks whose
    premium answer fixes a wrong cheap answer, cheapest-first, until the target
    accuracy is met.  Because each escalation buys at most one correct task, the
    cheapest-first greedy is exactly optimal for this integer problem.

    **The target is not always reachable.**  Escalation can only convert a
    *wrong cheap* answer into a *correct premium* answer.  If the cascade's
    accuracy exceeds what that set can deliver — e.g. the cascade happened to be
    right on tasks where both models are wrong, or the cascade beat premium on a
    task — then no amount of label-aware routing reaches ``target_accuracy``.
    In that case the oracle returns the cheapest plan at its **attainable
    ceiling**, and ``oracle_accuracy_matched`` is ``False``.  Reporting the gap
    is the honest thing to do: it means the regret figure compares the cascade
    against a *slightly weaker* oracle, which biases regret **downward**, so a
    positive regret remains a safe claim.  Callers that need a strict
    apples-to-apples number should check the flag before quoting the ratio.

    ``regret`` is then the amount the cascade overspends *for the accuracy it
    actually delivered* — the money that a better escalation signal (not a
    better model) could have saved.  It is non-negative by construction, and it
    is the honest upper bound on what routing improvements can recover.
    """

    target_accuracy: float
    oracle_cost_usd: float
    oracle_tokens: int
    oracle_accuracy: float
    oracle_n_escalations: int
    cascade_cost_usd: float
    cascade_tokens: int
    cascade_accuracy: float
    cascade_n_escalations: int
    cheap_cost_usd: float
    cheap_tokens: int
    premium_cost_usd: float
    premium_tokens: int
    n_tasks: int
    #: How many tasks the oracle could possibly have gotten right, given that
    #: escalation only fixes wrong-cheap/correct-premium tasks.
    oracle_max_correct: int = 0

    @property
    def oracle_accuracy_matched(self) -> bool:
        """True when the oracle actually reached the cascade's accuracy."""
        return abs(self.oracle_accuracy - self.cascade_accuracy) < 1e-9

    @property
    def oracle_accuracy_gap(self) -> float:
        """``cascade_accuracy - oracle_accuracy``; positive = oracle fell short."""
        return self.cascade_accuracy - self.oracle_accuracy

    @property
    def regret_cost_usd(self) -> float:
        return self.cascade_cost_usd - self.oracle_cost_usd

    @property
    def regret_cost_share(self) -> float:
        if self.cascade_cost_usd <= 0:
            return 0.0
        return self.regret_cost_usd / self.cascade_cost_usd

    @property
    def regret_tokens(self) -> int:
        return self.cascade_tokens - self.oracle_tokens

    @property
    def regret_token_share(self) -> float:
        if self.cascade_tokens <= 0:
            return 0.0
        return self.regret_tokens / self.cascade_tokens

    @property
    def routing_efficiency(self) -> float:
        """Share of the achievable savings (vs always-premium) actually captured.

        ``1.0`` means the cascade is as cheap as the label-aware oracle at the
        same accuracy; ``0.0`` means it saved nothing over always-premium.
        """
        achievable = self.premium_cost_usd - self.oracle_cost_usd
        if achievable <= 0:
            return 1.0
        captured = self.premium_cost_usd - self.cascade_cost_usd
        return max(0.0, min(1.0, captured / achievable))

    @property
    def cost_ratio_vs_oracle(self) -> float:
        if self.oracle_cost_usd <= 0:
            return float("inf")
        return self.cascade_cost_usd / self.oracle_cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_accuracy": round(self.target_accuracy, 6),
            "oracle_cost_usd": round(self.oracle_cost_usd, 8),
            "oracle_tokens": self.oracle_tokens,
            "oracle_accuracy": round(self.oracle_accuracy, 6),
            "oracle_n_escalations": self.oracle_n_escalations,
            "oracle_max_correct": self.oracle_max_correct,
            "oracle_accuracy_matched": self.oracle_accuracy_matched,
            "oracle_accuracy_gap": round(self.oracle_accuracy_gap, 6),
            "cascade_cost_usd": round(self.cascade_cost_usd, 8),
            "cascade_tokens": self.cascade_tokens,
            "cascade_accuracy": round(self.cascade_accuracy, 6),
            "cascade_n_escalations": self.cascade_n_escalations,
            "cheap_cost_usd": round(self.cheap_cost_usd, 8),
            "cheap_tokens": self.cheap_tokens,
            "premium_cost_usd": round(self.premium_cost_usd, 8),
            "premium_tokens": self.premium_tokens,
            "regret_cost_usd": round(self.regret_cost_usd, 8),
            "regret_cost_share": round(self.regret_cost_share, 6),
            "regret_tokens": self.regret_tokens,
            "regret_token_share": round(self.regret_token_share, 6),
            "routing_efficiency": round(self.routing_efficiency, 6),
            "cost_ratio_vs_oracle": round(self.cost_ratio_vs_oracle, 6),
        }


class OraclePlan(NamedTuple):
    """The cheapest label-aware routing plan reaching a target accuracy.

    A named tuple (rather than a bare tuple) so that call sites read as
    ``plan.oracle_max_correct`` instead of ``result[4]``; it still unpacks
    positionally for convenience.
    """

    cost_usd: float
    tokens: int
    n_correct: int
    n_escalations: int
    #: Ceiling the oracle can reach: cheap-correct tasks plus tasks where
    #: escalation converts a wrong cheap answer into a correct premium one.
    max_correct: int


def oracle_plan(
    cheap_runs: Mapping[str, TaskRun],
    premium_runs: Mapping[str, TaskRun],
    target_n_correct: int,
) -> OraclePlan:
    """Cheapest routing that reaches ``target_n_correct`` with label knowledge.

    ``max_correct`` is the ceiling the oracle can reach: every task where the
    cheap answer is already right, plus every task where escalation converts a
    wrong cheap answer into a correct premium one.  Tasks where both models are
    wrong are unattainable.  When ``target_n_correct > max_correct`` the oracle
    returns its best attainable plan and the caller reports the shortfall.
    """
    task_ids = sorted(set(cheap_runs) & set(premium_runs))
    # Everyone gets the cheap answer first; that is the baseline.
    cost = sum(cheap_runs[t].cost_usd for t in task_ids)
    tokens = sum(cheap_runs[t].total_tokens for t in task_ids)
    n_correct = sum(1 for t in task_ids if cheap_runs[t].final_verdict.is_correct)

    # Only tasks where escalating actually fixes the answer can buy accuracy.
    candidates = []
    for task_id in task_ids:
        cheap = cheap_runs[task_id]
        premium = premium_runs[task_id]
        if cheap.final_verdict.is_correct:
            continue
        if not premium.final_verdict.is_correct:
            continue
        delta_cost = premium.cost_usd - cheap.cost_usd
        delta_tokens = premium.total_tokens - cheap.total_tokens
        candidates.append((delta_cost, delta_tokens, task_id))

    max_correct = n_correct + len(candidates)
    if n_correct >= target_n_correct:
        return OraclePlan(cost, tokens, n_correct, 0, max_correct)

    candidates.sort(key=lambda item: (item[0], item[1]))
    n_escalations = 0
    for delta_cost, delta_tokens, _task_id in candidates:
        if n_correct >= target_n_correct:
            break
        cost += delta_cost
        tokens += delta_tokens
        n_correct += 1
        n_escalations += 1
    return OraclePlan(cost, tokens, n_correct, n_escalations, max_correct)


def regret_analysis(audit: AuditRun, threshold: float | None = None) -> RegretAnalysis:
    """Compare the cascade against an accuracy-matched label-aware oracle."""
    comparison = compare(audit, threshold=threshold)
    cascade = comparison.cascade
    cheap_runs = {r.task_id: r for r in audit.select(Strategy.CHEAP)}
    premium_runs = {r.task_id: r for r in audit.select(Strategy.PREMIUM)}
    cascade_runs = audit.select(Strategy.CASCADE, cascade.threshold)
    n_tasks = len(cascade_runs) or 1

    target_n_correct = cascade.n_correct
    plan = oracle_plan(cheap_runs, premium_runs, target_n_correct)

    return RegretAnalysis(
        target_accuracy=cascade.accuracy,
        oracle_cost_usd=plan.cost_usd,
        oracle_tokens=plan.tokens,
        oracle_accuracy=plan.n_correct / n_tasks,
        oracle_n_escalations=plan.n_escalations,
        oracle_max_correct=plan.max_correct,
        cascade_cost_usd=cascade.total_cost_usd,
        cascade_tokens=cascade.total_tokens,
        cascade_accuracy=cascade.accuracy,
        cascade_n_escalations=cascade.n_escalated,
        cheap_cost_usd=comparison.cheap.total_cost_usd,
        cheap_tokens=comparison.cheap.total_tokens,
        premium_cost_usd=comparison.premium.total_cost_usd,
        premium_tokens=comparison.premium.total_tokens,
        n_tasks=n_tasks,
    )


# --------------------------------------------------------------------------- #
# 5. Domain breakdown
# --------------------------------------------------------------------------- #
@dataclass
class DomainSlice:
    domain: str
    n_tasks: int
    cheap_accuracy: float
    premium_accuracy: float
    cascade_accuracy: float
    cascade_escalation_rate: float
    cascade_tokens: int
    cascade_cost_usd: float
    wasted_token_share: float
    #: Escalations that turned a correct cheap answer into an incorrect final
    #: answer.  The "upgrade made it worse" case, which pure cost accounting
    #: cannot see at all.
    n_harmful_escalations: int = 0

    @property
    def harmful_escalation_rate(self) -> float:
        return self.n_harmful_escalations / self.n_tasks if self.n_tasks else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "n_tasks": self.n_tasks,
            "cheap_accuracy": round(self.cheap_accuracy, 6),
            "premium_accuracy": round(self.premium_accuracy, 6),
            "cascade_accuracy": round(self.cascade_accuracy, 6),
            "cascade_escalation_rate": round(self.cascade_escalation_rate, 6),
            "cascade_tokens": self.cascade_tokens,
            "cascade_cost_usd": round(self.cascade_cost_usd, 8),
            "wasted_token_share": round(self.wasted_token_share, 6),
            "n_harmful_escalations": self.n_harmful_escalations,
            "harmful_escalation_rate": round(self.harmful_escalation_rate, 6),
        }


def domain_breakdown(audit: AuditRun, threshold: float | None = None) -> list[DomainSlice]:
    """Per-domain accuracy/cost, so the report can say *where* routing fails."""
    comparison = compare(audit, threshold=threshold)
    cascade_runs = audit.select(Strategy.CASCADE, comparison.cascade.threshold)
    cheap_runs = {r.task_id: r for r in audit.select(Strategy.CHEAP)}
    premium_runs = {r.task_id: r for r in audit.select(Strategy.PREMIUM)}

    domains: dict[str, list[TaskRun]] = {}
    for run in cascade_runs:
        domain = str(run.metadata.get("domain") or _domain_of(audit, run.task_id))
        domains.setdefault(domain, []).append(run)

    slices: list[DomainSlice] = []
    for domain, runs in sorted(domains.items()):
        ledgers = attribute_all(runs)
        tokens = sum(l.total_tokens for l in ledgers)
        wasted = sum(l.wasted_escalation_tokens + l.wasted_downgrade_tokens for l in ledgers)
        cheap_acc = _accuracy([cheap_runs[r.task_id] for r in runs if r.task_id in cheap_runs])
        premium_acc = _accuracy([premium_runs[r.task_id] for r in runs if r.task_id in premium_runs])
        slices.append(
            DomainSlice(
                domain=domain,
                n_tasks=len(runs),
                cheap_accuracy=cheap_acc,
                premium_accuracy=premium_acc,
                cascade_accuracy=_accuracy(runs),
                cascade_escalation_rate=sum(1 for r in runs if r.escalated) / len(runs),
                cascade_tokens=tokens,
                cascade_cost_usd=sum(r.cost_usd for r in runs),
                wasted_token_share=(wasted / tokens) if tokens else 0.0,
                n_harmful_escalations=sum(
                    1
                    for r in runs
                    if r.escalated
                    and r.cheap_was_correct is True
                    and not r.final_verdict.is_correct
                ),
            )
        )
    return slices


def _domain_of(audit: AuditRun, task_id: str) -> str:
    prefix = task_id.split("-")[0]
    return {"qa": "qa", "math": "math", "code": "code"}.get(prefix, "other")


def _accuracy(runs: Sequence[TaskRun]) -> float:
    if not runs:
        return 0.0
    return sum(1 for r in runs if r.final_verdict.is_correct) / len(runs)


# --------------------------------------------------------------------------- #
# 6. Convenience: the whole analysis bundle
# --------------------------------------------------------------------------- #
@dataclass
class AuditAnalysis:
    """Everything the report needs, computed once."""

    audit: AuditRun
    comparison: Comparison
    routing_errors: RoutingErrorAnalysis
    sensitivity: ThresholdSensitivity
    regret: RegretAnalysis
    domains: list[DomainSlice]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.audit.run_id,
            "provider": self.audit.provider,
            "task_set": self.audit.task_set,
            "n_tasks": self.audit.n_tasks,
            "seed": self.audit.seed,
            "thresholds": list(self.audit.thresholds),
            "comparison": self.comparison.to_dict(),
            "routing_errors": self.routing_errors.to_dict(),
            "sensitivity": self.sensitivity.to_dict(),
            "regret": self.regret.to_dict(),
            "domains": [d.to_dict() for d in self.domains],
        }


def analyse(audit: AuditRun, threshold: float | None = None) -> AuditAnalysis:
    """Run the full analysis bundle for one audit run."""
    return AuditAnalysis(
        audit=audit,
        comparison=compare(audit, threshold=threshold),
        routing_errors=routing_error_analysis(audit, threshold=threshold),
        sensitivity=threshold_sensitivity(audit),
        regret=regret_analysis(audit, threshold=threshold),
        domains=domain_breakdown(audit, threshold=threshold),
    )
