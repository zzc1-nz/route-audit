"""Per-step token attribution and per-configuration aggregation.

Two things live here:

``TaskAttribution``
    The per-task ledger: cheap-attempt tokens, escalation tokens, total, and
    *who paid for what*.  This is the object the brief asks for in item 2.

``ConfigSummary``
    The per-(strategy, threshold) aggregate: accuracy, tokens, cost, escalation
    counts, over-escalation rate, over-downgrade rate, and the wasted-token
    share.  This is the object the report renders.

Design rule: **no metric is computed in the report layer.**  Everything the
README quotes is produced here and persisted in the JSON run, so a reviewer can
recompute the headline numbers from ``audit.json`` alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .types import (
    AuditRun,
    EscalationReason,
    Strategy,
    TaskRun,
    TokenBreakdown,
    Verdict,
)


# --------------------------------------------------------------------------- #
# Per-task ledger
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaskAttribution:
    """Token attribution for one task run.

    ``wasted_tokens`` is the money the router burned by making a *wrong* routing
    decision, split into the two directions:

    * ``wasted_escalation_tokens`` — premium tokens spent although the cheap
      answer was already correct (over-escalation).
    * ``wasted_downgrade_tokens`` — cheap tokens spent on a task the cheap model
      could not do, i.e. tokens that bought nothing before the necessary
      escalation (over-downgrade).

    Note the asymmetry the audit exposes: ``wasted_escalation`` is usually
    expensive *per token* (premium price) but rare, while ``wasted_downgrade``
    is cheap *per token* but happens on every genuinely hard task.
    """

    task_id: str
    strategy: Strategy
    threshold: float | None
    cheap_tokens: int
    escalation_tokens: int
    total_tokens: int
    cheap_cost_usd: float
    escalation_cost_usd: float
    escalated: bool
    escalation_reason: EscalationReason
    final_correct: bool
    cheap_correct: bool | None
    premium_correct: bool | None
    confidence: float | None
    format_ok: bool

    @property
    def total_cost_usd(self) -> float:
        return self.cheap_cost_usd + self.escalation_cost_usd

    @property
    def wasted_escalation_tokens(self) -> int:
        return self.escalation_tokens if self.escalated and self.cheap_correct is True else 0

    @property
    def wasted_escalation_cost_usd(self) -> float:
        return self.escalation_cost_usd if self.escalated and self.cheap_correct is True else 0.0

    @property
    def wasted_downgrade_tokens(self) -> int:
        return self.cheap_tokens if self.escalated and self.cheap_correct is False else 0

    @property
    def wasted_downgrade_cost_usd(self) -> float:
        return self.cheap_cost_usd if self.escalated and self.cheap_correct is False else 0.0

    @property
    def missed_escalation(self) -> bool:
        """Cheap was wrong, router stayed cheap, answer shipped wrong."""
        return (not self.escalated) and self.cheap_correct is False

    @property
    def harmful_escalation(self) -> bool:
        """Escalated from a *correct* cheap answer into an *incorrect* one.

        The failure mode that no cost metric can see: the router paid the
        premium price and still lost accuracy.  This is why cascade accuracy is
        not monotone in the threshold.
        """
        return self.escalated and self.cheap_correct is True and not self.final_correct

    @property
    def silent_error_tokens(self) -> int:
        """Tokens spent producing a confidently wrong answer."""
        return self.total_tokens if self.missed_escalation else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy.value,
            "threshold": self.threshold,
            "cheap_tokens": self.cheap_tokens,
            "escalation_tokens": self.escalation_tokens,
            "total_tokens": self.total_tokens,
            "cheap_cost_usd": round(self.cheap_cost_usd, 8),
            "escalation_cost_usd": round(self.escalation_cost_usd, 8),
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason.value,
            "final_correct": self.final_correct,
            "cheap_correct": self.cheap_correct,
            "premium_correct": self.premium_correct,
            "confidence": self.confidence,
            "format_ok": self.format_ok,
            "wasted_escalation_tokens": self.wasted_escalation_tokens,
            "wasted_downgrade_tokens": self.wasted_downgrade_tokens,
            "missed_escalation": self.missed_escalation,
        }


def attribute(run: TaskRun) -> TaskAttribution:
    """Build the ledger for one task run, straight from its attempts."""
    primary = run.primary
    escalated = run.escalated_attempt
    return TaskAttribution(
        task_id=run.task_id,
        strategy=run.strategy,
        threshold=run.threshold,
        cheap_tokens=primary.tokens if primary is not None else 0,
        escalation_tokens=escalated.tokens if escalated is not None else 0,
        total_tokens=run.total_tokens,
        cheap_cost_usd=run.cheap_cost_usd,
        escalation_cost_usd=run.premium_cost_usd,
        escalated=run.escalated,
        escalation_reason=run.escalation_reason,
        final_correct=run.final_verdict.is_correct,
        cheap_correct=run.cheap_was_correct,
        premium_correct=run.premium_was_correct,
        confidence=run.confidence,
        format_ok=run.format_ok,
    )


def attribute_all(runs: Iterable[TaskRun]) -> list[TaskAttribution]:
    return [attribute(run) for run in runs]


# --------------------------------------------------------------------------- #
# Per-configuration aggregate
# --------------------------------------------------------------------------- #
@dataclass
class ConfigSummary:
    """Everything the report says about one (strategy, threshold) configuration."""

    strategy: Strategy
    threshold: float | None
    label: str
    n_tasks: int
    n_correct: int
    total_tokens: int
    #: Tokens of the first stage.  For cascade configurations this is the cheap
    #: attempt; **for strategy B it is the premium call**, since there is no
    #: cheaper stage.  Kept for backwards compatibility — prefer the
    #: ``primary_tokens`` property when the intent is "first stage".
    cheap_tokens: int
    #: Tokens of the second (escalated) stage; always 0 for A and B.
    escalation_tokens: int
    total_cost_usd: float
    cheap_cost_usd: float
    escalation_cost_usd: float
    n_escalated: int
    n_wasted_escalation: int
    n_necessary_escalation: int
    n_missed_escalation: int
    n_silent_error: int
    n_cheap_correct: int
    n_premium_correct: int
    wasted_escalation_tokens: int
    wasted_downgrade_tokens: int
    wasted_escalation_cost_usd: float
    wasted_downgrade_cost_usd: float
    escalation_reasons: Mapping[str, int] = field(default_factory=dict)
    mean_confidence: float | None = None
    latency_ms: float = 0.0

    @property
    def primary_tokens(self) -> int:
        """Tokens of the *first* stage.

        For cascade configurations this is the cheap attempt; for strategy B it
        is the premium call (there is no cheaper stage).  Use this — not
        ``cheap_tokens`` — whenever the intent is "first stage" rather than
        "billed at the cheap tier", because strategy B's whole spend lands here.
        """
        return self.cheap_tokens

    @property
    def escalation_tokens_actual(self) -> int:
        """Tokens of the *second* stage; always zero for A and B."""
        return self.escalation_tokens

    # -- headline rates ----------------------------------------------------- #
    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_tasks if self.n_tasks else 0.0

    @property
    def escalation_rate(self) -> float:
        return self.n_escalated / self.n_tasks if self.n_tasks else 0.0

    @property
    def over_escalation_rate(self) -> float:
        """Share of *escalations* that were unnecessary (cheap was right)."""
        return self.n_wasted_escalation / self.n_escalated if self.n_escalated else 0.0

    @property
    def over_escalation_rate_of_tasks(self) -> float:
        return self.n_wasted_escalation / self.n_tasks if self.n_tasks else 0.0

    @property
    def over_downgrade_rate(self) -> float:
        """Share of *tasks* where the cheap attempt failed and forced an upgrade.

        This is the brief's definition: "廉价模型答错、触发升级，产生了双倍 token
        开销".  Every one of these tasks paid for a cheap attempt that bought
        nothing plus a premium attempt that did the real work.
        """
        return self.n_necessary_escalation / self.n_tasks if self.n_tasks else 0.0

    @property
    def missed_escalation_rate(self) -> float:
        """Share of tasks that shipped a wrong answer without escalating.

        The *other* routing failure: the router trusted a wrong cheap answer.
        """
        return self.n_missed_escalation / self.n_tasks if self.n_tasks else 0.0

    @property
    def wasted_token_share(self) -> float:
        """Wasted tokens (both directions) as a share of total tokens."""
        if not self.total_tokens:
            return 0.0
        return (self.wasted_escalation_tokens + self.wasted_downgrade_tokens) / self.total_tokens

    @property
    def wasted_downgrade_token_share(self) -> float:
        """The brief's item 3: extra tokens from over-downgrade / total tokens."""
        if not self.total_tokens:
            return 0.0
        return self.wasted_downgrade_tokens / self.total_tokens

    @property
    def wasted_escalation_token_share(self) -> float:
        if not self.total_tokens:
            return 0.0
        return self.wasted_escalation_tokens / self.total_tokens

    @property
    def wasted_cost_share(self) -> float:
        if not self.total_cost_usd:
            return 0.0
        return (self.wasted_escalation_cost_usd + self.wasted_downgrade_cost_usd) / self.total_cost_usd

    @property
    def tokens_per_correct(self) -> float:
        return self.total_tokens / self.n_correct if self.n_correct else float("inf")

    @property
    def cost_per_correct(self) -> float:
        return self.total_cost_usd / self.n_correct if self.n_correct else float("inf")

    @property
    def escalation_token_share(self) -> float:
        if not self.total_tokens:
            return 0.0
        return self.escalation_tokens / self.total_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "threshold": self.threshold,
            "label": self.label,
            "n_tasks": self.n_tasks,
            "n_correct": self.n_correct,
            "accuracy": round(self.accuracy, 6),
            "total_tokens": self.total_tokens,
            "cheap_tokens": self.cheap_tokens,
            "escalation_tokens": self.escalation_tokens,
            "escalation_token_share": round(self.escalation_token_share, 6),
            "total_cost_usd": round(self.total_cost_usd, 8),
            "cheap_cost_usd": round(self.cheap_cost_usd, 8),
            "escalation_cost_usd": round(self.escalation_cost_usd, 8),
            "n_escalated": self.n_escalated,
            "escalation_rate": round(self.escalation_rate, 6),
            "n_wasted_escalation": self.n_wasted_escalation,
            "over_escalation_rate": round(self.over_escalation_rate, 6),
            "over_escalation_rate_of_tasks": round(self.over_escalation_rate_of_tasks, 6),
            "n_necessary_escalation": self.n_necessary_escalation,
            "over_downgrade_rate": round(self.over_downgrade_rate, 6),
            "n_missed_escalation": self.n_missed_escalation,
            "missed_escalation_rate": round(self.missed_escalation_rate, 6),
            "wasted_escalation_tokens": self.wasted_escalation_tokens,
            "wasted_downgrade_tokens": self.wasted_downgrade_tokens,
            "wasted_token_share": round(self.wasted_token_share, 6),
            "wasted_downgrade_token_share": round(self.wasted_downgrade_token_share, 6),
            "wasted_escalation_token_share": round(self.wasted_escalation_token_share, 6),
            "wasted_cost_usd": round(
                self.wasted_escalation_cost_usd + self.wasted_downgrade_cost_usd, 8
            ),
            "wasted_cost_share": round(self.wasted_cost_share, 6),
            "tokens_per_correct": None
            if self.tokens_per_correct == float("inf")
            else round(self.tokens_per_correct, 2),
            "cost_per_correct_usd": None
            if self.cost_per_correct == float("inf")
            else round(self.cost_per_correct, 8),
            "escalation_reasons": dict(self.escalation_reasons),
            "mean_confidence": None if self.mean_confidence is None else round(self.mean_confidence, 6),
            "latency_ms": round(self.latency_ms, 2),
            "n_cheap_correct": self.n_cheap_correct,
            "n_premium_correct": self.n_premium_correct,
        }


def summarise(
    runs: Sequence[TaskRun],
    *,
    strategy: Strategy | None = None,
    threshold: float | None = None,
    label: str | None = None,
) -> ConfigSummary:
    """Aggregate one configuration's runs into a ``ConfigSummary``."""
    if not runs:
        raise ValueError("summarise() needs at least one run")
    runs = list(runs)
    first = runs[0]
    strategy = strategy or first.strategy
    if threshold is None and strategy is Strategy.CASCADE:
        threshold = first.threshold
    ledgers = attribute_all(runs)

    tokens = TokenBreakdown()
    for run in runs:
        tokens = tokens + run.tokens

    confidences = [run.confidence for run in runs if run.confidence is not None]
    reasons: dict[str, int] = {}
    for run in runs:
        if run.escalated:
            key = run.escalation_reason.value
            reasons[key] = reasons.get(key, 0) + 1

    return ConfigSummary(
        strategy=strategy,
        threshold=threshold,
        label=label or _default_label(strategy, threshold),
        n_tasks=len(runs),
        n_correct=sum(1 for run in runs if run.final_verdict.is_correct),
        total_tokens=tokens.total_tokens,
        cheap_tokens=tokens.primary_tokens,
        escalation_tokens=tokens.escalated_tokens,
        total_cost_usd=sum(run.cost_usd for run in runs),
        cheap_cost_usd=sum(run.cheap_cost_usd for run in runs),
        escalation_cost_usd=sum(run.premium_cost_usd for run in runs),
        n_escalated=sum(1 for run in runs if run.escalated),
        n_wasted_escalation=sum(1 for run in runs if run.wasted_escalation),
        n_necessary_escalation=sum(1 for run in runs if run.necessary_escalation),
        # ``n_silent_error`` is the canonical field; ``n_missed_escalation`` is
        # the same predicate under the name used in the report prose.  They are
        # kept as two fields because the JSON schema is consumed by the report
        # and by external analysis, but they are computed once here.
        n_missed_escalation=sum(1 for run in runs if run.silent_error),
        n_silent_error=sum(1 for run in runs if run.silent_error),
        n_cheap_correct=sum(1 for run in runs if run.cheap_was_correct is True),
        n_premium_correct=sum(1 for run in runs if run.premium_was_correct is True),
        wasted_escalation_tokens=sum(ledger.wasted_escalation_tokens for ledger in ledgers),
        wasted_downgrade_tokens=sum(ledger.wasted_downgrade_tokens for ledger in ledgers),
        wasted_escalation_cost_usd=sum(ledger.wasted_escalation_cost_usd for ledger in ledgers),
        wasted_downgrade_cost_usd=sum(ledger.wasted_downgrade_cost_usd for ledger in ledgers),
        escalation_reasons=dict(sorted(reasons.items())),
        mean_confidence=(sum(confidences) / len(confidences)) if confidences else None,
        latency_ms=sum(run.latency_ms for run in runs) / len(runs),
    )


def _default_label(strategy: Strategy, threshold: float | None) -> str:
    if strategy is Strategy.CASCADE and threshold is not None:
        return f"C · cascade @ {threshold:.1f}"
    return strategy.title


def cascade_key(threshold: float) -> str:
    """Canonical dictionary key for one cascade threshold.

    Uses ``%g`` (shortest round-tripping form) rather than ``%.1f``: the config
    layer accepts any float in [0, 1], and a fixed 1-decimal key silently
    collapsed distinct thresholds (0.21 and 0.24 both became ``C@0.2``), which
    made a custom sweep report one point instead of two.  For the default
    ladders (0.1 .. 0.9, and 0.3/0.5/0.7) the keys are unchanged.
    """
    return f"C@{float(threshold):g}"


def summarise_audit(audit: AuditRun) -> dict[str, ConfigSummary]:
    """All configurations of a run, keyed by a stable label.

    Keys: ``A``, ``B``, ``C@0.3``, ``C@0.5``, ... (see :func:`cascade_key`).
    """
    out: dict[str, ConfigSummary] = {}
    for strategy in audit.strategies_present():
        if strategy is Strategy.CASCADE:
            for threshold in audit.thresholds:
                runs = audit.select(Strategy.CASCADE, threshold)
                if runs:
                    out[cascade_key(threshold)] = summarise(runs, threshold=threshold)
        else:
            runs = audit.select(strategy)
            if runs:
                out[strategy.value] = summarise(runs, strategy=strategy)
    return out
