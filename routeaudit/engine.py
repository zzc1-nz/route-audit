"""Audit engine: executes the three strategies over a labelled task set.

Methodology in one paragraph
---------------------------
An offline audit can do something an online router cannot: it knows the gold
label, so it can grade *every* attempt, including attempts the router skipped.
That makes the **counterfactual outcome matrix** free to compute:

    for every task:  (cheap_verdict, premium_verdict) in {correct, incorrect}^2

Strategy A gives the cheap column, strategy B gives the premium column, and the
two are executed once for the whole sweep.  Every cascade threshold is then a
*pure replay* over the same matrix: no extra cheap tokens, only the premium
calls the threshold actually triggers.  This is why a 3-threshold sweep costs
less than 3 separate deployments — and it is the structural reason an offline
audit is cheap enough to run on every routing change.

Execution phases
----------------
1. ``matrix``  — A (cheap) + B (premium) for every task, graded.
2. ``cascade`` — for each threshold, replay the cheap attempt, run the
   label-free gate, and call the premium model only when the gate fires.

Both phases run in a thread pool (I/O bound for real APIs, cheap for the
simulator) with a deterministic per-task seed, a thread-safe call cache and a
hard budget guard.
"""

from __future__ import annotations

import datetime as _dt
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .config import AuditConfig
from .confidence import Router
from .grading import Grader, extract_confidence
from .pricing import PriceTable, default_price_table
from .providers import Provider, make_provider
from .taskset import load_tasks, task_set_summary
from .types import (
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
)


# --------------------------------------------------------------------------- #
# Budget guard
# --------------------------------------------------------------------------- #
class BudgetExceeded(RuntimeError):
    """Raised when simulated spend crosses ``AuditConfig.max_budget_usd``."""


class BudgetGuard:
    """Thread-safe running-cost counter, CostGuard-style but audit-side.

    The point is not to *enforce* a budget in production (that is CostGuard's
    job) but to guarantee the audit itself cannot silently scale into a real
    invoice.  A blocked audit is a loud failure, not a quiet overrun.
    """

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = float(limit_usd)
        self._spent = 0.0
        self._calls = 0
        self._lock = threading.Lock()

    def charge(self, cost_usd: float, *, calls: int = 1) -> None:
        with self._lock:
            self._spent += cost_usd
            self._calls += calls
            spent = self._spent
        if spent > self.limit_usd:
            raise BudgetExceeded(
                f"audit budget exceeded: ${spent:.4f} > ${self.limit_usd:.2f} "
                f"after {self._calls} calls — reduce --n-tasks/--sweep or raise --max-budget"
            )

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls


# --------------------------------------------------------------------------- #
# Call cache
# --------------------------------------------------------------------------- #
class CallCache:
    """Memoises provider responses within one audit run.

    Keyed by ``(task_id, model, stage)``.  Sharing the *cheap* attempt across
    cascade thresholds is what keeps a sweep cheap; sharing the *escalated*
    attempt across thresholds is what keeps it honest (the same premium call is
    only paid once even if two thresholds both trigger it).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._data: dict[tuple[str, str, str], ModelResponse] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[str, str, str]) -> ModelResponse | None:
        if not self.enabled:
            return None
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
            return value

    def put(self, key: tuple[str, str, str], value: ModelResponse) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
@dataclass
class AuditEngine:
    """Runs one audit: 3 strategies x N tasks x M thresholds."""

    config: AuditConfig
    tasks: Sequence[Task] | None = None
    provider: Provider | None = None
    grader: Grader | None = None
    price_table: PriceTable | None = None
    progress: Any = None  # optional callable(phase, done, total)

    _cache: CallCache = field(init=False)
    _guard: BudgetGuard = field(init=False)

    def __post_init__(self) -> None:
        if self.tasks is None:
            self.tasks = load_tasks(self.config.task_set, self.config.n_tasks, self.config.seed)
        if len(self.tasks) != self.config.n_tasks:
            # A user-supplied set may be smaller than --n-tasks; audit what exists.
            self.config.n_tasks = len(self.tasks)
        if self.provider is None:
            self.provider = make_provider("mock")
        self.grader = self.grader or Grader()
        self.price_table = self.price_table or default_price_table(self.config.price_scale)
        self._cache = CallCache(self.config.enable_cache)
        self._guard = BudgetGuard(self.config.max_budget_usd)

    # -- low level ---------------------------------------------------------- #
    def _call(self, task: Task, model_name: str, stage: str) -> ModelResponse:
        key = (task.task_id, model_name, stage)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        spec = self.config.cheap if model_name == self.config.cheap_model else self.config.premium
        if model_name not in (self.config.cheap_model, self.config.premium_model):
            spec = self.price_table.spec(model_name)
        draft = None
        if stage == "escalated":
            primary = self._cache.get((task.task_id, self.config.cheap_model, "primary"))
            draft = primary.text if primary is not None else None
        try:
            response = self.provider.complete(task, spec, stage=stage, seed=self.config.seed, draft=draft)  # type: ignore[call-arg]
        except TypeError:
            # Provider without the optional ``draft`` kwarg (e.g. MockProvider).
            response = self.provider.complete(task, spec, stage=stage, seed=self.config.seed)  # type: ignore[call-arg]
        self._cache.put(key, response)
        cost = self.price_table.cost_usd(spec, response.usage)
        self._guard.charge(cost)
        return response

    def _attempt(
        self,
        task: Task,
        index: int,
        stage: str,
        model_name: str,
        response: ModelResponse,
        reason: EscalationReason = EscalationReason.NONE,
    ) -> Attempt:
        tier = "cheap" if model_name == self.config.cheap_model else "premium"
        grade = self.grader.grade(task, response.text)  # type: ignore[union-attr]
        return Attempt(
            index=index,
            stage=stage,
            model=model_name,
            tier=tier,
            response=response,
            verdict=grade.verdict,
            confidence=extract_confidence(response.text),
            format_ok=grade.format_ok,
            escalation_reason=reason,
            detail=grade.detail,
        )

    # -- phase 1: counterfactual matrix ------------------------------------- #
    def _run_single(self, task: Task, strategy: Strategy) -> TaskRun:
        """Strategies A and B: one model for every task."""
        if strategy is Strategy.CHEAP:
            model_name = self.config.cheap_model
        else:
            model_name = self.config.premium_model
        response = self._call(task, model_name, "primary")
        attempt = self._attempt(task, 0, "primary", model_name, response)
        verdict = attempt.verdict
        return TaskRun(
            task_id=task.task_id,
            strategy=strategy,
            threshold=None,
            attempts=[attempt],
            final_answer=attempt.response.text,
            final_model=model_name,
            final_tier=attempt.tier,
            final_verdict=verdict,
            escalated=False,
            escalation_reason=EscalationReason.NONE,
            cheap_verdict=verdict if strategy is Strategy.CHEAP else None,
            premium_verdict=verdict if strategy is Strategy.PREMIUM else None,
            tokens=TokenBreakdown.from_attempts([attempt]),
            cheap_cost_usd=(
                self.price_table.cost_usd(self.config.cheap, response.usage)  # type: ignore[union-attr]
                if strategy is Strategy.CHEAP
                else 0.0
            ),
            premium_cost_usd=(
                self.price_table.cost_usd(self.config.premium, response.usage)  # type: ignore[union-attr]
                if strategy is Strategy.PREMIUM
                else 0.0
            ),
            confidence=attempt.confidence,
            format_ok=attempt.format_ok,
            latency_ms=response.latency_ms,
            metadata={"domain": task.domain, "answer_kind": task.metadata.get("answer_kind", "text")},
        )

    # -- phase 2: cascade replay ------------------------------------------- #
    def _run_cascade(
        self,
        task: Task,
        threshold: float,
        cheap_run: TaskRun,
        premium_run: TaskRun,
    ) -> TaskRun:
        """Strategy C at one threshold, replaying the already-paid cheap call."""
        router = Router(threshold=threshold, format_gate=self.config.format_gate)
        primary = cheap_run.attempts[0]
        decision = router.decide(
            confidence=primary.confidence,
            format_ok=primary.format_ok,
            provider_error=primary.response.error,
        )
        attempts: list[Attempt] = [primary]
        if decision.escalate:
            escalated_response = self._call(task, self.config.premium_model, "escalated")
            escalated = self._attempt(
                task,
                1,
                "escalated",
                self.config.premium_model,
                escalated_response,
                reason=decision.reason,
            )
            attempts.append(escalated)
            final = escalated
        else:
            final = primary
        return TaskRun(
            task_id=task.task_id,
            strategy=Strategy.CASCADE,
            threshold=threshold,
            attempts=attempts,
            final_answer=final.response.text,
            final_model=final.model,
            final_tier=final.tier,
            final_verdict=final.verdict,
            escalated=decision.escalate,
            escalation_reason=decision.reason,
            cheap_verdict=cheap_run.final_verdict,
            premium_verdict=premium_run.final_verdict,
            tokens=TokenBreakdown.from_attempts(attempts),
            cheap_cost_usd=self.price_table.cost_usd(self.config.cheap, primary.usage),  # type: ignore[union-attr]
            premium_cost_usd=(
                self.price_table.cost_usd(self.config.premium, attempts[1].usage)  # type: ignore[union-attr]
                if decision.escalate
                else 0.0
            ),
            confidence=primary.confidence,
            format_ok=primary.format_ok,
            latency_ms=sum(a.response.latency_ms for a in attempts),
            metadata={
                "gate_detail": decision.detail,
                "domain": task.domain,
                "answer_kind": task.metadata.get("answer_kind", "text"),
            },
        )

    # -- orchestration ------------------------------------------------------ #
    def run(self) -> AuditRun:
        tasks = list(self.tasks or [])
        if not tasks:
            raise ValueError("audit needs at least one task")
        thresholds = list(self.config.thresholds)
        runs: list[TaskRun] = []

        # Phase 1 — A and B.  These two passes create the counterfactual matrix.
        phase1: list[tuple[Task, Strategy]] = []
        if "A" in self.config.include_strategies:
            phase1 += [(t, Strategy.CHEAP) for t in tasks]
        if "B" in self.config.include_strategies:
            phase1 += [(t, Strategy.PREMIUM) for t in tasks]

        cheap_runs: dict[str, TaskRun] = {}
        premium_runs: dict[str, TaskRun] = {}
        self._report_progress("matrix", 0, len(phase1))
        with ThreadPoolExecutor(max_workers=max(1, self.config.max_workers)) as pool:
            futures = {
                pool.submit(self._run_single, task, strategy): (task, strategy)
                for task, strategy in phase1
            }
            done = 0
            for future in as_completed(futures):
                task, strategy = futures[future]
                result = future.result()
                if strategy is Strategy.CHEAP:
                    cheap_runs[task.task_id] = result
                else:
                    premium_runs[task.task_id] = result
                done += 1
                self._report_progress("matrix", done, len(phase1))

        # A and B must both exist for the cascade to be priceable; if the user
        # excluded one, fall back to executing the missing column for the
        # cascade only (still graded, still counted).
        if "C" in self.config.include_strategies:
            for task in tasks:
                if task.task_id not in cheap_runs:
                    cheap_runs[task.task_id] = self._run_single(task, Strategy.CHEAP)
                if task.task_id not in premium_runs:
                    premium_runs[task.task_id] = self._run_single(task, Strategy.PREMIUM)

            # Phase 2 — cascade replay, one pass per threshold.
            total = len(tasks) * len(thresholds)
            done = 0
            for threshold in thresholds:
                self._report_progress(f"cascade@{threshold}", 0, total)
                with ThreadPoolExecutor(max_workers=max(1, self.config.max_workers)) as pool:
                    futures = {
                        pool.submit(
                            self._run_cascade,
                            task,
                            threshold,
                            cheap_runs[task.task_id],
                            premium_runs[task.task_id],
                        ): task
                        for task in tasks
                    }
                    for future in as_completed(futures):
                        runs.append(future.result())
                        done += 1
                        self._report_progress(f"cascade@{threshold}", done, total)

        # Deterministic ordering: strategy, then threshold, then task order.
        order = {task.task_id: i for i, task in enumerate(tasks)}
        if "A" in self.config.include_strategies:
            runs.extend(sorted(cheap_runs.values(), key=lambda r: order[r.task_id]))
        if "B" in self.config.include_strategies:
            runs.extend(sorted(premium_runs.values(), key=lambda r: order[r.task_id]))
        runs.sort(
            key=lambda r: (
                {"A": 0, "B": 1, "C": 2}[r.strategy.value],
                -1.0 if r.threshold is None else r.threshold,
                order.get(r.task_id, 0),
            )
        )

        audit = AuditRun(
            run_id=f"audit-{uuid.uuid4().hex[:12]}",
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            provider=getattr(self.provider, "name", "unknown"),
            task_set=self.config.task_set,
            n_tasks=len(tasks),
            seed=self.config.seed,
            thresholds=tuple(thresholds),
            config=self.config.to_dict(),
            runs=runs,
            stats={
                "executed_calls": self._guard.calls,
                "cache_hits": self._cache.hits,
                "cache_misses": self._cache.misses,
                "cache_entries": len(self._cache),
                "executed_cost_usd": round(self._guard.spent_usd, 6),
                "task_set_summary": task_set_summary(tasks),
                "cheap_model": self.config.cheap.to_dict(),
                "premium_model": self.config.premium.to_dict(),
                "format_gate": self.config.format_gate,
                "cache_enabled": self.config.enable_cache,
            },
            notes={
                "method": (
                    "Counterfactual matrix: strategy A grades every cheap attempt, strategy B "
                    "grades every premium attempt; cascade thresholds replay the same cheap "
                    "attempt, so a threshold sweep pays only for the premium calls it triggers."
                ),
                "billed_vs_executed": (
                    "Per-strategy token totals are billed totals (what one deployment at that "
                    "setting would pay). stats.executed_calls is what the sweep actually spent, "
                    "which is lower because the cheap stage is shared across thresholds."
                ),
            },
        )
        return audit

    def _report_progress(self, phase: str, done: int, total: int) -> None:
        if self.progress is not None:
            try:
                self.progress(phase, done, total)
            except Exception:  # pragma: no cover - progress must never break an audit
                pass


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def run_audit(
    config: AuditConfig | None = None,
    *,
    tasks: Sequence[Task] | None = None,
    provider: Provider | None = None,
    progress: Any = None,
) -> AuditRun:
    """One-call entry point used by the CLI and the tests."""
    return AuditEngine(
        config=config or AuditConfig(),
        tasks=tasks,
        provider=provider,
        progress=progress,
    ).run()


def build_matrix(audit: AuditRun) -> dict[str, tuple[Verdict | None, Verdict | None]]:
    """``task_id -> (cheap_verdict, premium_verdict)`` from an audit run."""
    cheap = {r.task_id: r for r in audit.select(Strategy.CHEAP)}
    premium = {r.task_id: r for r in audit.select(Strategy.PREMIUM)}
    matrix: dict[str, tuple[Verdict | None, Verdict | None]] = {}
    for task_id in set(cheap) | set(premium):
        c = cheap.get(task_id)
        p = premium.get(task_id)
        matrix[task_id] = (
            c.final_verdict if c else None,
            p.final_verdict if p else None,
        )
    return matrix
