"""Provider adapters.

Two backends share one tiny interface:

``MockProvider``
    Deterministic, offline, zero-cost.  Latent difficulty comes from the task,
    correctness is drawn from a seeded RNG conditioned on model quality, and the
    self-reported confidence is drawn with a *calibration error* so that the
    audit has something real to measure: an overconfident cheap model is exactly
    what makes cascade routing lose money.  Same seed -> same run, forever.

``APIProvider``
    Real HTTP calls (OpenAI-compatible and Anthropic), used only when the user
    explicitly asks for a live run.  The audit methodology is identical; only
    the answer/usage source changes.

Nothing here knows about routing policy, thresholds or cost — providers just
answer questions and report token usage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .config import ModelSpec
from .types import ModelResponse, Task, Usage


class Provider(Protocol):
    """Minimal surface the audit needs from any model backend.

    ``draft`` is the cheap stage's answer, passed only for ``stage="escalated"``
    so that the premium call can be a *verifier* prompt (the realistic cascade
    shape) rather than a blind re-ask.
    """

    name: str

    def complete(
        self,
        task: Task,
        model: ModelSpec,
        *,
        stage: str,
        seed: int,
        draft: str | None = None,
    ) -> ModelResponse:
        ...


# --------------------------------------------------------------------------- #
# Deterministic helpers
# --------------------------------------------------------------------------- #
def stable_hash(*parts: Any) -> int:
    """Process-independent 64-bit hash (Python's ``hash`` is salted)."""
    digest = hashlib.blake2b(
        "||".join(str(p) for p in parts).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _rng(*parts: Any) -> random.Random:
    return random.Random(stable_hash(*parts))


# --------------------------------------------------------------------------- #
# Offline simulator
# --------------------------------------------------------------------------- #
ANSWER_FORMAT = "Answer: <final answer>\nConfidence: <0-1>"


@dataclass
class MockProvider:
    """Deterministic offline model simulator.

    The simulator is a *calibrated stylised model*, not a random answer
    generator: it reproduces the two properties that make routing economics
    interesting, and it is explicit about both.

    Latent correctness
        ``p_correct = floor + (ceiling - floor) * (1 - difficulty) ** gamma``,
        read straight off the ``ModelSpec``.  The cheap tier has a low floor and
        a gamma above 1 (fast decay); the premium tier has a high floor and
        gamma below 1 (holds up).  The gap between the two curves is the entire
        economic justification for routing, so it is expressed in interpretable
        parameters rather than a magic constant.

    Reported confidence
        ``confidence = p_correct + overconfidence + N(0, sigma)``, clipped.
        ``overconfidence`` (cheap +0.10, premium +0.02) is the calibration error
        that the audit exists to measure: the cheap model's self-report is
        systematically optimistic, so a threshold tuned on it over-escalates.
        ``sigma`` (cheap 0.18, premium 0.12) is the per-call noise that makes the
        signal imperfect even when unbiased.

    Plus: ~6% of cheap answers are structurally malformed (the format gate's
    trigger), and token usage is drawn around a per-model band so that the
    premium/cheap blended price ratio lands near the real ~16x.

    Determinism: every draw is seeded from ``(seed, task_id, model, stage)``, so
    the same seed reproduces the same audit byte for byte.
    """

    name: str = "mock"
    mean_tokens: int = 2000
    malformed_rate: float = 0.06
    latency_scale: float = 0.0  # set >0 to simulate wall-clock latency
    #: Per-call noise on the self-reported confidence, per tier.
    confidence_sigma: Mapping[str, float] = field(
        default_factory=lambda: {"cheap": 0.18, "premium": 0.12}
    )
    #: Floor/ceiling so no model claims certainty it cannot have.
    confidence_floor: Mapping[str, float] = field(
        default_factory=lambda: {"cheap": 0.02, "premium": 0.35}
    )
    confidence_ceiling: float = 0.97

    # -- helpers ------------------------------------------------------------ #
    def p_correct(self, model: ModelSpec, task: Task) -> float:
        return model.p_correct(float(task.difficulty))

    def _confidence(self, model: ModelSpec, p_correct: float, rng: random.Random) -> float:
        sigma = float(self.confidence_sigma.get(model.tier, 0.25))
        floor = float(self.confidence_floor.get(model.tier, 0.03))
        raw = p_correct + model.overconfidence + rng.gauss(0.0, sigma)
        return max(floor, min(self.confidence_ceiling, raw))

    def _usage(self, model: ModelSpec, task: Task, stage: str, rng: random.Random) -> Usage:
        # Cheap models answer tersely; premium models reason more.  The
        # escalated call is larger: it sees the task *and* the cheap draft.
        base = self.mean_tokens
        if model.tier == "premium":
            base = int(base * 1.18)
        if stage == "escalated":
            base = int(base * 1.25)
        jitter = rng.uniform(0.72, 1.35)
        # Longer prompts for longer tasks; deterministic on task length.
        length_factor = 1.0 + min(len(task.prompt) / 4000.0, 0.35)
        total = max(64, int(base * jitter * length_factor))
        completion_ratio = rng.uniform(0.22, 0.42) if model.tier == "cheap" else rng.uniform(0.35, 0.60)
        completion = max(16, int(total * completion_ratio))
        return Usage(prompt_tokens=total - completion, completion_tokens=completion)

    def _text(
        self,
        task: Task,
        model: ModelSpec,
        correct: bool,
        confidence: float,
        malformed: bool,
        rng: random.Random,
    ) -> str:
        if malformed:
            return (
                "Sure! Let me think about this step by step.\n"
                "The reasoning is mostly there but I could not settle on a single final value."
            )
        answer = task.reference if correct else _wrong_answer(task, rng)
        return (
            f"Reasoning: consider the problem statement and work it through.\n"
            f"Answer: {answer}\n"
            f"Confidence: {confidence:.2f}"
        )

    # -- interface ---------------------------------------------------------- #
    def complete(
        self,
        task: Task,
        model: ModelSpec,
        *,
        stage: str,
        seed: int,
        draft: str | None = None,
    ) -> ModelResponse:
        started = time.perf_counter()
        # Correctness AND well-formedness are seeded WITHOUT the stage, on
        # purpose.  The audit prices routing errors against a counterfactual
        # ("would premium have fixed this?"), so the escalated call must be a
        # valid sample from that same counterfactual.  If stage changed either
        # draw, `premium_verdict` would describe a different answer than the
        # escalation actually produced and the attribution would measure noise.
        # Tokens, confidence and latency *are* stage-dependent (the escalated
        # prompt is larger and re-prompted).
        outcome_rng = _rng("mock-outcome", seed, task.task_id, model.name)
        stage_rng = _rng("mock-stage", seed, task.task_id, model.name, stage)
        p_correct = self.p_correct(model, task)
        correct = outcome_rng.random() < p_correct
        confidence = self._confidence(model, p_correct, stage_rng)

        malformed = outcome_rng.random() < (
            self.malformed_rate if model.tier == "cheap" else self.malformed_rate * 0.35
        )
        text = self._text(task, model, correct, confidence, malformed, stage_rng)
        usage = self._usage(model, task, stage, stage_rng)
        latency = model.latency_ms_per_call * stage_rng.uniform(0.6, 1.5) + self.latency_scale
        return ModelResponse(
            text=text,
            usage=usage,
            latency_ms=latency,
            model=model.name,
            provider="mock",
            raw={
                "simulated_correct": correct,
                "simulated_p_correct": round(p_correct, 4),
                "malformed": malformed,
                "has_draft": draft is not None,
            },
        )


def _wrong_answer(task: Task, rng: random.Random) -> str:
    """Produce a plausible-but-wrong answer of the same shape as the gold one."""
    gold = task.reference
    kind = task.metadata.get("answer_kind", "text")
    if kind == "number":
        try:
            value = float(gold)
        except (TypeError, ValueError):
            return f"{gold}0"
        delta = max(1.0, abs(value) * rng.uniform(0.05, 0.6))
        shifted = value + rng.choice((-1, 1)) * delta
        return str(int(shifted)) if float(shifted).is_integer() else f"{shifted:.4f}".rstrip("0").rstrip(".")
    if kind == "code":
        # Keep the signature, break the body: looks like real code, fails tests.
        return gold.replace("return", "return None  # ").replace("pass", "return None")
    if kind == "choice":
        letters = [c for c in "ABCD" if c != gold.strip()[:1]]
        return rng.choice(letters) if letters else "A"
    tokens = gold.split()
    if not tokens:
        return "unknown"
    idx = rng.randrange(len(tokens))
    return " ".join(tokens[:idx] + ["not"] + tokens[idx:])


# --------------------------------------------------------------------------- #
# Live providers (only used on explicit request)
# --------------------------------------------------------------------------- #
class ProviderError(RuntimeError):
    """A live call failed; the audit records it as a zero-token failure."""


def _post_json(url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout: float) -> Mapping[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        raise ProviderError(f"network error: {exc.reason}") from exc


def build_prompt(task: Task) -> str:
    return (
        "Answer the task below. Reply in exactly this format and nothing else:\n"
        f"{ANSWER_FORMAT}\n\n"
        f"Task:\n{task.prompt}\n"
    )


def build_verifier_prompt(task: Task, draft: str) -> str:
    return (
        "You are the escalation stage of a model cascade. A cheaper model attempted the task "
        "below and its draft is included. Produce the best final answer.\n"
        "Reply in exactly this format and nothing else:\n"
        f"{ANSWER_FORMAT}\n\n"
        f"Task:\n{task.prompt}\n\n"
        f"Cheap model draft:\n{draft}\n"
    )


@dataclass
class APIProvider:
    """Live provider for OpenAI-compatible and Anthropic endpoints.

    Deliberately tiny: this is an audit sandbox, not a gateway.  Missing keys
    raise immediately so a live run fails loudly instead of silently auditing
    simulated numbers.
    """

    name: str = "api"
    timeout_s: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 1024
    openai_base_url: str = "https://api.openai.com/v1"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com/v1"

    def _prompt(self, task: Task, stage: str, draft: str | None) -> str:
        if stage == "escalated" and draft:
            return build_verifier_prompt(task, draft)
        return build_prompt(task)

    def complete(
        self,
        task: Task,
        model: ModelSpec,
        *,
        stage: str,
        seed: int,
        draft: str | None = None,
    ) -> ModelResponse:
        started = time.perf_counter()
        prompt = self._prompt(task, stage, draft)
        try:
            if model.provider == "anthropic":
                text, usage = self._call_anthropic(model, prompt)
            elif model.provider == "deepseek":
                text, usage = self._call_openai_compat(
                    model, prompt, base_url=self.deepseek_base_url, key_env="DEEPSEEK_API_KEY"
                )
            else:
                text, usage = self._call_openai_compat(
                    model, prompt, base_url=self.openai_base_url, key_env="OPENAI_API_KEY"
                )
        except ProviderError as exc:
            return ModelResponse(
                text="",
                usage=Usage.zero(),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                model=model.name,
                provider=model.provider,
                error=str(exc),
            )
        return ModelResponse(
            text=text,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model=model.name,
            provider=model.provider,
        )

    # -- concrete calls ----------------------------------------------------- #
    def _call_openai_compat(
        self, model: ModelSpec, prompt: str, *, base_url: str, key_env: str
    ) -> tuple[str, Usage]:  # pragma: no cover - network path
        key = os.environ.get(key_env)
        if not key:
            raise ProviderError(f"{key_env} is not set")
        payload = {
            "model": model.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = _post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            payload,
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            self.timeout_s,
        )
        choices = data.get("choices") or [{}]
        text = ((choices[0].get("message") or {}).get("content")) or ""
        raw_usage = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
        )
        return text, usage

    def _call_anthropic(self, model: ModelSpec, prompt: str) -> tuple[str, Usage]:  # pragma: no cover
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        payload = {
            "model": model.name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = _post_json(
            f"{self.anthropic_base_url.rstrip('/')}/messages",
            payload,
            {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            self.timeout_s,
        )
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, Mapping))
        raw_usage = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(raw_usage.get("input_tokens", 0)),
            completion_tokens=int(raw_usage.get("output_tokens", 0)),
        )
        return text, usage


def make_provider(name: str, **kwargs: Any) -> Provider:
    if name in ("mock", "offline", "sim"):
        return MockProvider(**kwargs)
    if name in ("api", "live", "litellm"):
        return APIProvider(**kwargs)
    raise ValueError(f"unknown provider {name!r}; use 'mock' or 'api'")
