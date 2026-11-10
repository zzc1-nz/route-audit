"""Shared data contracts for the route-audit sandbox.

Every object that crosses a module boundary is defined here, so the provider
layer, the policy engine, the accounting layer and the report layer agree on a
single vocabulary.  All objects are JSON round-trippable: an audit run can be
persisted and re-analysed without re-spending a single token.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Strategy(str, Enum):
    """The three routing policies under audit."""

    CHEAP = "A"
    PREMIUM = "B"
    CASCADE = "C"

    @property
    def title(self) -> str:
        return {
            Strategy.CHEAP: "A · cheap-only",
            Strategy.PREMIUM: "B · premium-only",
            Strategy.CASCADE: "C · cascade",
        }[self]

    @property
    def zh(self) -> str:
        return {
            Strategy.CHEAP: "策略 A：全廉价模型",
            Strategy.PREMIUM: "策略 B：全高档模型",
            Strategy.CASCADE: "策略 C：级联路由",
        }[self]


class Verdict(str, Enum):
    """Outcome of grading one attempt (or one final answer) against gold."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNGRADED = "ungraded"

    @property
    def is_correct(self) -> bool:
        return self is Verdict.CORRECT


class EscalationReason(str, Enum):
    """Why the cascade called the premium model for this task."""

    NONE = "none"
    LOW_CONFIDENCE = "low_confidence"
    FORMAT_FAILURE = "format_failure"
    MISSING_CONFIDENCE = "missing_confidence"
    PROVIDER_ERROR = "provider_error"

    @property
    def zh(self) -> str:
        return {
            EscalationReason.NONE: "未升级",
            EscalationReason.LOW_CONFIDENCE: "置信度低于阈值",
            EscalationReason.FORMAT_FAILURE: "格式检查失败",
            EscalationReason.MISSING_CONFIDENCE: "置信度缺失",
            EscalationReason.PROVIDER_ERROR: "廉价调用失败",
        }[self]


# --------------------------------------------------------------------------- #
# Token accounting primitives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Usage:
    """Token usage of exactly one model call."""

    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError(f"negative token usage: {self!r}")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )

    @classmethod
    def zero(cls) -> "Usage":
        return cls(0, 0)

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Usage":
        return cls(int(data["prompt_tokens"]), int(data["completion_tokens"]))


@dataclass(frozen=True)
class TokenBreakdown:
    """The per-step token ledger of one task run.

    This is the object that makes "hidden" cascade cost visible: the cheap
    attempt and the escalation are never collapsed into a single number.
    """

    primary_prompt_tokens: int = 0
    primary_completion_tokens: int = 0
    escalated_prompt_tokens: int = 0
    escalated_completion_tokens: int = 0

    @property
    def primary_tokens(self) -> int:
        return self.primary_prompt_tokens + self.primary_completion_tokens

    @property
    def escalated_tokens(self) -> int:
        return self.escalated_prompt_tokens + self.escalated_completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.primary_tokens + self.escalated_tokens

    def __add__(self, other: "TokenBreakdown") -> "TokenBreakdown":
        return TokenBreakdown(
            self.primary_prompt_tokens + other.primary_prompt_tokens,
            self.primary_completion_tokens + other.primary_completion_tokens,
            self.escalated_prompt_tokens + other.escalated_prompt_tokens,
            self.escalated_completion_tokens + other.escalated_completion_tokens,
        )

    @classmethod
    def from_attempts(cls, attempts: Iterable["Attempt"]) -> "TokenBreakdown":
        p_prompt = p_compl = e_prompt = e_compl = 0
        for attempt in attempts:
            usage = attempt.usage
            if attempt.stage == "escalated":
                e_prompt += usage.prompt_tokens
                e_compl += usage.completion_tokens
            else:
                p_prompt += usage.prompt_tokens
                p_compl += usage.completion_tokens
        return cls(p_prompt, p_compl, e_prompt, e_compl)

    def to_dict(self) -> dict[str, int]:
        return {
            "primary_prompt_tokens": self.primary_prompt_tokens,
            "primary_completion_tokens": self.primary_completion_tokens,
            "escalated_prompt_tokens": self.escalated_prompt_tokens,
            "escalated_completion_tokens": self.escalated_completion_tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TokenBreakdown":
        return cls(
            int(data.get("primary_prompt_tokens", 0)),
            int(data.get("primary_completion_tokens", 0)),
            int(data.get("escalated_prompt_tokens", 0)),
            int(data.get("escalated_completion_tokens", 0)),
        )


# --------------------------------------------------------------------------- #
# Tasks and grading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    """One labelled audit item.

    ``difficulty`` is a *generator hint* used by the offline simulator to draw
    a plausible latent difficulty; routers never see it (that would leak the
    label).  ``reference`` is the gold answer and is only used by the grader.
    """

    task_id: str
    prompt: str
    reference: str
    domain: str = "qa"
    difficulty: float = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not 0.0 <= float(self.difficulty) <= 1.0:
            raise ValueError(f"difficulty out of [0,1]: {self.difficulty!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "reference": self.reference,
            "domain": self.domain,
            "difficulty": float(self.difficulty),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Task":
        return cls(
            task_id=str(data["task_id"]),
            prompt=str(data["prompt"]),
            reference=str(data["reference"]),
            domain=str(data.get("domain", "qa")),
            difficulty=float(data.get("difficulty", 0.5)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class GradeResult:
    """Label-aware grading outcome for one attempt.

    ``format_ok`` is *label-free* (was an answer extracted at all?) and is the
    only part a real router could observe; ``verdict`` needs the gold label and
    is what the offline audit uses to price routing mistakes.
    """

    verdict: Verdict
    extracted_answer: str | None
    format_ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "extracted_answer": self.extracted_answer,
            "format_ok": self.format_ok,
            "detail": self.detail,
        }


# --------------------------------------------------------------------------- #
# Model calls
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelResponse:
    """One provider call result (including failures, which still cost nothing)."""

    text: str
    usage: Usage
    latency_ms: float
    model: str
    provider: str
    error: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": self.usage.to_dict(),
            "latency_ms": float(self.latency_ms),
            "model": self.model,
            "provider": self.provider,
            "error": self.error,
            "raw": dict(self.raw),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelResponse":
        return cls(
            text=str(data.get("text", "")),
            usage=Usage.from_dict(data.get("usage") or {}),
            latency_ms=float(data.get("latency_ms", 0.0)),
            model=str(data.get("model", "")),
            provider=str(data.get("provider", "")),
            error=data.get("error"),
            raw=dict(data.get("raw") or {}),
        )


@dataclass(frozen=True)
class Attempt:
    """One model call inside one task run.

    ``stage`` is ``"primary"`` (the first, cheap attempt) or ``"escalated"``
    (the premium attempt triggered by the router).
    """

    index: int
    stage: str
    model: str
    tier: str
    response: ModelResponse
    verdict: Verdict
    confidence: float | None
    format_ok: bool
    escalation_reason: EscalationReason = EscalationReason.NONE
    detail: str = ""

    def __post_init__(self) -> None:
        if self.stage not in ("primary", "escalated"):
            raise ValueError(f"bad stage: {self.stage!r}")
        if self.tier not in ("cheap", "premium"):
            raise ValueError(f"bad tier: {self.tier!r}")

    @property
    def usage(self) -> Usage:
        return self.response.usage

    @property
    def tokens(self) -> int:
        return self.response.usage.total_tokens

    @property
    def ok(self) -> bool:
        return self.response.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "stage": self.stage,
            "model": self.model,
            "tier": self.tier,
            "response": self.response.to_dict(),
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "format_ok": self.format_ok,
            "escalation_reason": self.escalation_reason.value,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Attempt":
        return cls(
            index=int(data["index"]),
            stage=str(data["stage"]),
            model=str(data["model"]),
            tier=str(data["tier"]),
            response=ModelResponse.from_dict(data.get("response") or {}),
            verdict=Verdict(data["verdict"]),
            confidence=data.get("confidence"),
            format_ok=bool(data.get("format_ok", False)),
            escalation_reason=EscalationReason(data.get("escalation_reason", "none")),
            detail=str(data.get("detail", "")),
        )


# --------------------------------------------------------------------------- #
# Task-level and run-level results
# --------------------------------------------------------------------------- #
@dataclass
class TaskRun:
    """Everything the audit knows about one task under one strategy/config.

    ``cheap_verdict`` / ``premium_verdict`` are the *counterfactual* outcomes
    used to price routing errors.  Grading a cheap answer against the gold label
    costs zero tokens, which is precisely why an offline audit can measure
    "would the escalation have been necessary?" while an online router cannot.
    """

    task_id: str
    strategy: Strategy
    threshold: float | None
    attempts: list[Attempt]
    final_answer: str
    final_model: str
    final_tier: str
    final_verdict: Verdict
    escalated: bool
    escalation_reason: EscalationReason
    cheap_verdict: Verdict | None
    premium_verdict: Verdict | None
    tokens: TokenBreakdown
    cheap_cost_usd: float
    premium_cost_usd: float
    confidence: float | None = None
    format_ok: bool = True
    latency_ms: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        return self.final_verdict.is_correct

    @property
    def cost_usd(self) -> float:
        return self.cheap_cost_usd + self.premium_cost_usd

    @property
    def total_tokens(self) -> int:
        return self.tokens.total_tokens

    @property
    def primary(self) -> Attempt | None:
        for attempt in self.attempts:
            if attempt.stage == "primary":
                return attempt
        return None

    @property
    def escalated_attempt(self) -> Attempt | None:
        for attempt in self.attempts:
            if attempt.stage == "escalated":
                return attempt
        return None

    @property
    def cheap_was_correct(self) -> bool | None:
        """``None`` when the cheap attempt was never graded (should not happen)."""
        return None if self.cheap_verdict is None else self.cheap_verdict.is_correct

    @property
    def premium_was_correct(self) -> bool | None:
        """``None`` when strategy B never graded this task.

        Semantics worth noting for the cascade: this is the *blind* premium
        verdict (strategy B, asked the task directly).  The escalated attempt in
        a cascade is a *verifier* prompt — it sees the cheap draft — so its own
        verdict can differ from ``premium_verdict``.  Both are recorded; the
        counterfactual matrix uses this field, the delivered answer uses
        ``final_verdict``.  Do not assume ``premium_verdict ==
        escalated_attempt.verdict``.
        """
        return None if self.premium_verdict is None else self.premium_verdict.is_correct

    @property
    def wasted_escalation(self) -> bool:
        """Escalated although the cheap attempt was already correct."""
        return self.escalated and self.cheap_was_correct is True

    @property
    def necessary_escalation(self) -> bool:
        """Escalated because the cheap attempt was wrong (double token spend)."""
        return self.escalated and self.cheap_was_correct is False

    @property
    def silent_error(self) -> bool:
        """Not escalated, cheap attempt wrong, final answer wrong."""
        return (not self.escalated) and self.cheap_was_correct is False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy.value,
            "threshold": self.threshold,
            "attempts": [a.to_dict() for a in self.attempts],
            "final_answer": self.final_answer,
            "final_model": self.final_model,
            "final_tier": self.final_tier,
            "final_verdict": self.final_verdict.value,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason.value,
            "cheap_verdict": None if self.cheap_verdict is None else self.cheap_verdict.value,
            "premium_verdict": None if self.premium_verdict is None else self.premium_verdict.value,
            "tokens": self.tokens.to_dict(),
            "cheap_cost_usd": self.cheap_cost_usd,
            "premium_cost_usd": self.premium_cost_usd,
            "confidence": self.confidence,
            "format_ok": self.format_ok,
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskRun":
        return cls(
            task_id=str(data["task_id"]),
            strategy=Strategy(data["strategy"]),
            threshold=data.get("threshold"),
            attempts=[Attempt.from_dict(a) for a in data.get("attempts", [])],
            final_answer=str(data.get("final_answer", "")),
            final_model=str(data.get("final_model", "")),
            final_tier=str(data.get("final_tier", "cheap")),
            final_verdict=Verdict(data.get("final_verdict", "ungraded")),
            escalated=bool(data.get("escalated", False)),
            escalation_reason=EscalationReason(data.get("escalation_reason", "none")),
            cheap_verdict=None
            if data.get("cheap_verdict") is None
            else Verdict(data["cheap_verdict"]),
            premium_verdict=None
            if data.get("premium_verdict") is None
            else Verdict(data["premium_verdict"]),
            tokens=TokenBreakdown.from_dict(data.get("tokens") or {}),
            cheap_cost_usd=float(data.get("cheap_cost_usd", 0.0)),
            premium_cost_usd=float(data.get("premium_cost_usd", 0.0)),
            confidence=data.get("confidence"),
            format_ok=bool(data.get("format_ok", True)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class AuditRun:
    """A complete audit: every task under every strategy / threshold."""

    run_id: str
    created_at: str
    provider: str
    task_set: str
    n_tasks: int
    seed: int
    thresholds: tuple[float, ...]
    config: Mapping[str, Any]
    runs: list[TaskRun]
    stats: Mapping[str, Any] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)

    # -- selection helpers -------------------------------------------------- #
    def select(self, strategy: Strategy, threshold: float | None = None) -> list[TaskRun]:
        """Return runs for one strategy; cascade runs are keyed by threshold."""
        out = []
        for run in self.runs:
            if run.strategy is not strategy:
                continue
            if strategy is Strategy.CASCADE:
                if threshold is None or run.threshold == threshold:
                    out.append(run)
            else:
                out.append(run)
        return out

    def by_task(self, strategy: Strategy, threshold: float | None = None) -> dict[str, TaskRun]:
        return {run.task_id: run for run in self.select(strategy, threshold)}

    def strategies_present(self) -> list[Strategy]:
        present = {run.strategy for run in self.runs}
        return [s for s in (Strategy.CHEAP, Strategy.PREMIUM, Strategy.CASCADE) if s in present]

    @property
    def total_tokens(self) -> int:
        return sum(run.total_tokens for run in self.runs)

    @property
    def total_cost_usd(self) -> float:
        return sum(run.cost_usd for run in self.runs)

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "route-audit/1",
            "run_id": self.run_id,
            "created_at": self.created_at,
            "provider": self.provider,
            "task_set": self.task_set,
            "n_tasks": self.n_tasks,
            "seed": self.seed,
            "thresholds": list(self.thresholds),
            "config": dict(self.config),
            "stats": dict(self.stats),
            "notes": dict(self.notes),
            "runs": [run.to_dict() for run in self.runs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditRun":
        return cls(
            run_id=str(data["run_id"]),
            created_at=str(data.get("created_at", "")),
            provider=str(data.get("provider", "mock")),
            task_set=str(data.get("task_set", "")),
            n_tasks=int(data.get("n_tasks", 0)),
            seed=int(data.get("seed", 0)),
            thresholds=tuple(float(t) for t in data.get("thresholds", [])),
            config=dict(data.get("config") or {}),
            runs=[TaskRun.from_dict(r) for r in data.get("runs", [])],
            stats=dict(data.get("stats") or {}),
            notes=dict(data.get("notes") or {}),
        )

    def save(self, path: str) -> None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=1, default=str)

    @classmethod
    def load(cls, path: str) -> "AuditRun":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def group_by_threshold(runs: Sequence[TaskRun]) -> dict[float | None, list[TaskRun]]:
    """Group runs by threshold; strategies A/B have ``None``."""
    buckets: dict[float | None, list[TaskRun]] = {}
    for run in runs:
        buckets.setdefault(run.threshold, []).append(run)
    return buckets
