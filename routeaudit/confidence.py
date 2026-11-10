"""Escalation signal for the cascade.

The cascade must decide "do I trust this cheap answer?" *without* the gold
label.  Two observable signals are combined, in this order:

1. **Format gate** — if no answer could be parsed, escalate regardless of
   self-reported confidence.  This is free and catches truncation/malformed
   output.
2. **Confidence gate** — the model's own ``Confidence:`` value.  The audit
   exists precisely because this signal is *miscalibrated*: a cheap model that
   is wrong 28% of the time still reports 0.7 confidence, which is why a
   threshold tuned on vibes overspends.

The module also provides the calibration view (reliability table + Brier score)
used in the report, because "the router's signal is bad" is an audit finding in
its own right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .types import Attempt, EscalationReason, TaskRun, Verdict


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EscalationDecision:
    """Outcome of the label-free gate for one cheap attempt."""

    escalate: bool
    reason: EscalationReason
    confidence: float | None
    format_ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "escalate": self.escalate,
            "reason": self.reason.value,
            "confidence": self.confidence,
            "format_ok": self.format_ok,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Router:
    """Label-free escalation policy.

    ``threshold`` is the operating point under audit.  ``format_gate`` can be
    switched off to isolate the confidence gate's contribution.
    """

    threshold: float = 0.5
    format_gate: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold out of [0,1]: {self.threshold!r}")

    def decide(
        self,
        *,
        confidence: float | None,
        format_ok: bool,
        provider_error: str | None = None,
    ) -> EscalationDecision:
        if provider_error:
            return EscalationDecision(
                escalate=True,
                reason=EscalationReason.PROVIDER_ERROR,
                confidence=confidence,
                format_ok=format_ok,
                detail=provider_error,
            )
        if self.format_gate and not format_ok:
            return EscalationDecision(
                escalate=True,
                reason=EscalationReason.FORMAT_FAILURE,
                confidence=confidence,
                format_ok=format_ok,
                detail="no parseable answer",
            )
        if confidence is None:
            # No confidence channel: the only safe default for an audit is to
            # treat the answer as untrusted and escalate, then let the report
            # show how expensive that default is.
            #
            # Attribute the *cause* honestly: with the format gate disabled, an
            # unparseable answer has no ``Confidence:`` line either, so blaming
            # the missing confidence channel would mislabel the real failure.
            # The escalation still happens; only the reason differs.
            if not format_ok:
                return EscalationDecision(
                    escalate=True,
                    reason=EscalationReason.FORMAT_FAILURE,
                    confidence=None,
                    format_ok=format_ok,
                    detail="no parseable answer (format gate disabled; escalated anyway)",
                )
            return EscalationDecision(
                escalate=True,
                reason=EscalationReason.MISSING_CONFIDENCE,
                confidence=None,
                format_ok=format_ok,
                detail="model did not report confidence",
            )
        if confidence < self.threshold:
            return EscalationDecision(
                escalate=True,
                reason=EscalationReason.LOW_CONFIDENCE,
                confidence=confidence,
                format_ok=format_ok,
                detail=f"{confidence:.3f} < {self.threshold:.3f}",
            )
        return EscalationDecision(
            escalate=False,
            reason=EscalationReason.NONE,
            confidence=confidence,
            format_ok=format_ok,
            detail=f"{confidence:.3f} >= {self.threshold:.3f}",
        )


# --------------------------------------------------------------------------- #
# Calibration diagnostics
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationBin:
    lower: float
    upper: float
    n: int = 0
    n_correct: int = 0
    confidence_sum: float = 0.0

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.n if self.n else 0.0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def gap(self) -> float:
        """Positive gap = overconfident (says more than it delivers)."""
        return self.mean_confidence - self.accuracy

    def to_dict(self) -> dict[str, float | int]:
        return {
            "lower": round(self.lower, 3),
            "upper": round(self.upper, 3),
            "n": self.n,
            "mean_confidence": round(self.mean_confidence, 4),
            "accuracy": round(self.accuracy, 4),
            "gap": round(self.gap, 4),
        }


@dataclass
class CalibrationReport:
    n: int
    brier: float
    ece: float
    bins: list[CalibrationBin]
    mean_confidence: float
    accuracy: float
    overconfidence: float
    confidence_coverage: float

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "brier": round(self.brier, 4),
            "ece": round(self.ece, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "accuracy": round(self.accuracy, 4),
            "overconfidence": round(self.overconfidence, 4),
            "confidence_coverage": round(self.confidence_coverage, 4),
            "bins": [b.to_dict() for b in self.bins],
        }


def calibration_report(
    pairs: Iterable[tuple[float | None, Verdict]],
    n_bins: int = 10,
) -> CalibrationReport:
    """Reliability table + Brier score for one model's confidence channel.

    ``pairs`` yields ``(confidence, verdict)`` for every *cheap* attempt,
    including ungraded ones (they are skipped from accuracy but counted in
    coverage).
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("calibration_report needs at least one (confidence, verdict) pair")
    total = len(pairs)
    observed = [(c, v) for c, v in pairs if c is not None and v is not Verdict.UNGRADED]
    coverage = len(observed) / total

    bins = [
        CalibrationBin(i / n_bins, (i + 1) / n_bins)
        for i in range(n_bins)
    ]
    brier_terms: list[float] = []
    for confidence, verdict in observed:
        confidence = max(0.0, min(1.0, float(confidence)))
        correct = 1.0 if verdict.is_correct else 0.0
        brier_terms.append((confidence - correct) ** 2)
        idx = min(int(confidence * n_bins), n_bins - 1)
        bin_ = bins[idx]
        bin_.n += 1
        bin_.n_correct += int(correct)
        bin_.confidence_sum += confidence

    n_obs = len(observed)
    brier = sum(brier_terms) / n_obs if n_obs else 0.0
    ece = (
        sum(b.n / n_obs * abs(b.gap) for b in bins if b.n)
        if n_obs
        else 0.0
    )
    mean_conf = sum(c for c, _ in observed) / n_obs if n_obs else 0.0
    accuracy = sum(1 for _, v in observed if v.is_correct) / n_obs if n_obs else 0.0
    return CalibrationReport(
        n=n_obs,
        brier=brier,
        ece=ece,
        bins=bins,
        mean_confidence=mean_conf,
        accuracy=accuracy,
        overconfidence=mean_conf - accuracy,
        confidence_coverage=coverage,
    )


def calibration_from_runs(runs: Sequence[TaskRun]) -> CalibrationReport:
    """Convenience: build the calibration view from strategy-A (cheap-only) runs."""
    pairs = [(run.confidence, run.final_verdict) for run in runs]
    return calibration_report(pairs)


def optimal_threshold(
    pairs: Iterable[tuple[float | None, Verdict]],
    *,
    escalation_cost_ratio: float = 1.0,
) -> float:
    """Threshold that minimises expected cost given a fixed cheap/premium ratio.

    Cost model per task, in units of "one cheap call":
      * no escalation: 1
      * escalation:    1 + ``escalation_cost_ratio``
      * a wrong final answer is charged ``escalation_cost_ratio`` (the price of
        the premium call you should have made) so that accuracy is not free.
    Used only as a cross-check on the sweep; the report always shows the curve.
    """
    pairs = [(c, v) for c, v in pairs if c is not None and v is not Verdict.UNGRADED]
    if not pairs:
        raise ValueError("optimal_threshold needs observed pairs")
    best = (math.inf, 0.5)
    for step in range(1, 100):
        threshold = step / 100.0
        cost = 0.0
        for confidence, verdict in pairs:
            escalate = confidence < threshold
            if escalate:
                cost += 1.0 + escalation_cost_ratio
            elif not verdict.is_correct:
                cost += 1.0 + escalation_cost_ratio
            else:
                cost += 1.0
        if cost < best[0]:
            best = (cost, threshold)
    return best[1]
