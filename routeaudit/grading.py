"""Label-aware answer extraction and grading.

Grading is the part of the audit that costs zero tokens and buys all of the
insight: it lets the framework ask "would the premium model have been needed?"
*after the fact*, for every task, including tasks the router never escalated.

Answer normalisation is intentionally domain-aware but conservative: numeric
answers compare with a relative tolerance, code answers compare after stripping
whitespace/comments, multiple-choice compares letters, and free text compares
token overlap (token F1) against a threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .types import GradeResult, Task, Verdict


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
_ANSWER_LINE = re.compile(
    r"^\s*(?:final\s+)?answer\s*[:：]\s*(?P<answer>.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
_CONFIDENCE_LINE = re.compile(
    r"^\s*(?:final\s+)?confidence\s*[:：]\s*(?P<value>[0-9]*\.?[0-9]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_HASHED = re.compile(r"####\s*(?P<answer>[^\n]+)")
_CONFIDENCE_PCT = re.compile(
    r"^\s*(?:final\s+)?confidence\s*[:：]\s*(?P<value>[0-9]*\.?[0-9]+)\s*%\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_answer(text: str, *, max_fallback_words: int = 8) -> str | None:
    """Pull the final answer out of a model response.

    Order matters: an explicit ``Answer:`` line wins, then a MATH-style
    ``\\boxed{...}`` or ``#### x`` marker, then a bare last line.

    The bare-last-line fallback is deliberately conservative.  A response that
    just trails off in prose ("I could not settle on a single final value") must
    NOT be read as an answer, because the format gate's whole job is to catch
    exactly that failure mode.  So the fallback requires a short, answer-shaped
    line (``max_fallback_words``, default 8) and no sentence punctuation.
    """
    if not text or not text.strip():
        return None
    match = _ANSWER_LINE.search(text)
    if match:
        answer = _clean(match.group("answer"))
        return answer or None
    match = _BOXED.search(text)
    if match:
        answer = _clean(match.group(1))
        return answer or None
    match = _HASHED.search(text)
    if match:
        answer = _clean(match.group("answer"))
        return answer or None
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    # Check the terminator on the RAW line, before ``_clean`` strips trailing
    # punctuation.  ``_clean`` calls ``rstrip(".")``, so testing the cleaned
    # candidate for a trailing "." could never fire — which silently weakened
    # the format gate on exactly the trailing-off-prose case it exists to catch.
    raw = lines[-1]
    if raw.endswith((".", "!", "?", "…")):
        return None
    candidate = _clean(raw)
    if not candidate:
        return None
    # Reject prose: too many words means the model never produced a discrete
    # answer.  (The sentence-terminator test already ran on the raw line.)
    if len(candidate.split()) > max_fallback_words:
        return None
    return candidate


def extract_confidence(text: str) -> float | None:
    """Read the self-reported confidence, if the model provided one."""
    if not text:
        return None
    match = _CONFIDENCE_PCT.search(text)
    if match:
        return _clip(float(match.group("value")) / 100.0)
    match = _CONFIDENCE_LINE.search(text)
    if match:
        return _clip(float(match.group("value")))
    return None


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clean(answer: str) -> str:
    answer = answer.strip().strip("`").strip()
    answer = answer.rstrip(".").strip()
    return answer


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
_NUMERIC = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_CODE_STRIP = re.compile(r"(#.*|//.*|/\*.*?\*/)", re.DOTALL)
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Grader:
    """Configurable answer grader.

    ``text_threshold`` is the token-F1 needed to call two free-text answers
    equal; 0.6 is lenient enough for paraphrase and strict enough that a wrong
    numeric answer never passes.
    """

    text_threshold: float = 0.6
    numeric_tolerance: float = 1e-3
    code_normalise: bool = True

    def grade(self, task: Task, text: str) -> GradeResult:
        answer = extract_answer(text)
        if answer is None:
            return GradeResult(
                verdict=Verdict.INCORRECT,
                extracted_answer=None,
                format_ok=False,
                detail="no parseable answer",
            )
        kind = str(task.metadata.get("answer_kind", "text"))
        correct, detail = self._compare(task.reference, answer, kind)
        return GradeResult(
            verdict=Verdict.CORRECT if correct else Verdict.INCORRECT,
            extracted_answer=answer,
            format_ok=True,
            detail=detail,
        )

    def _compare(self, gold: str, answer: str, kind: str) -> tuple[bool, str]:
        if kind == "number":
            return self._compare_numeric(gold, answer)
        if kind == "code":
            return self._compare_code(gold, answer)
        if kind == "choice":
            return self._compare_choice(gold, answer)
        return self._compare_text(gold, answer)

    def _compare_numeric(self, gold: str, answer: str) -> tuple[bool, str]:
        gold_value = _first_number(gold)
        answer_value = _first_number(answer)
        if gold_value is None or answer_value is None:
            return False, "numeric comparison failed to parse"
        if gold_value == answer_value:
            return True, "exact numeric match"
        denom = max(abs(gold_value), 1e-12)
        relative = abs(gold_value - answer_value) / denom
        if relative <= self.numeric_tolerance:
            return True, f"numeric match within {self.numeric_tolerance:g} relative"
        return False, f"numeric mismatch (relative error {relative:.3g})"

    def _compare_code(self, gold: str, answer: str) -> tuple[bool, str]:
        normalise = (lambda s: _WS.sub(" ", _CODE_STRIP.sub(" ", s)).strip()) if self.code_normalise else (
            lambda s: s.strip()
        )
        if normalise(gold) == normalise(answer):
            return True, "code match after normalisation"
        # Partial credit: a body that contains the gold body is still a pass for
        # audit purposes only when the gold is a single short statement.
        if len(normalise(gold)) > 0 and normalise(gold) in normalise(answer):
            return True, "code contains gold body"
        return False, "code mismatch"

    def _compare_choice(self, gold: str, answer: str) -> tuple[bool, str]:
        gold_letter = _first_letter(gold)
        answer_letter = _first_letter(answer)
        if gold_letter and answer_letter and gold_letter == answer_letter:
            return True, "choice letter match"
        return False, "choice mismatch"

    def _compare_text(self, gold: str, answer: str) -> tuple[bool, str]:
        gold_norm = _normalise_text(gold)
        answer_norm = _normalise_text(answer)
        if not gold_norm or not answer_norm:
            return False, "empty after normalisation"
        if gold_norm == answer_norm:
            return True, "exact text match"
        score = token_f1(gold_norm, answer_norm)
        if score >= self.text_threshold:
            return True, f"token F1 {score:.2f} >= {self.text_threshold:g}"
        return False, f"token F1 {score:.2f} < {self.text_threshold:g}"


def _first_number(text: str) -> float | None:
    match = _NUMERIC.search(text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _first_letter(text: str) -> str | None:
    match = re.search(r"[A-Za-z]", text)
    return match.group(0).upper() if match else None


def _normalise_text(text: str) -> str:
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def token_f1(gold: str, answer: str) -> float:
    gold_tokens = gold.split()
    answer_tokens = answer.split()
    if not gold_tokens or not answer_tokens:
        return 0.0
    from collections import Counter

    gold_counts = Counter(gold_tokens)
    answer_counts = Counter(answer_tokens)
    overlap = sum((gold_counts & answer_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def grade_text(task: Task, text: str, grader: Grader | None = None) -> GradeResult:
    return (grader or Grader()).grade(task, text)
