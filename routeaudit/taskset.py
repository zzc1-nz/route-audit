"""Labelled task sets.

The audit needs *labels*, so every task carries a gold answer.  Two sources:

``builtin``
    Deterministic synthetic sets in the spirit of HumanEval / MATH / QA.  They
    are generated from a seeded RNG so that the same ``--seed`` always yields
    the same 200 tasks.  Each task has a latent ``difficulty`` that the router
    never sees; only the grader and the simulator use it.

``jsonl``
    Anything the user brings: one JSON object per line with ``task_id``,
    ``prompt``, ``reference`` and optional ``domain``/``difficulty``/``metadata``.
    HumanEval's official JSONL (``prompt``/``canonical_solution``/``test``) is
    accepted through :func:`load_humaneval`.

Task sets are *data*, never policy: swapping in a real benchmark must not
require touching the audit engine.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .types import Task


# --------------------------------------------------------------------------- #
# Built-in generators
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BuiltinSpec:
    name: str
    description: str
    factory: Callable[[int, random.Random], list[Task]]


def _qa_tasks(n: int, rng: random.Random) -> list[Task]:
    facts = [
        ("What is the capital of France?", "Paris", 0.12),
        ("Who wrote the novel 1984?", "George Orwell", 0.18),
        ("What is the chemical symbol for gold?", "Au", 0.10),
        ("Which planet is known as the Red Planet?", "Mars", 0.12),
        ("What is the largest ocean on Earth?", "Pacific Ocean", 0.14),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci", 0.16),
        ("What is the boiling point of water at sea level in Celsius?", "100", 0.20),
        ("In which year did the Berlin Wall fall?", "1989", 0.22),
        ("What is the longest river in the world?", "Nile", 0.28),
        ("Who developed the theory of general relativity?", "Albert Einstein", 0.18),
        ("What is the currency of Japan?", "yen", 0.15),
        ("How many bones are in the adult human body?", "206", 0.24),
        ("What gas do plants absorb during photosynthesis?", "carbon dioxide", 0.20),
        ("Who is the author of Pride and Prejudice?", "Jane Austen", 0.18),
        ("What is the smallest prime number?", "2", 0.10),
        ("Which element has atomic number 1?", "hydrogen", 0.16),
    ]
    distractors = [
        ("Lyon", "Madrid", "Rome"),
        ("Aldous Huxley", "Ray Bradbury", "Ernest Hemingway"),
        ("Ag", "Gd", "Go"),
        ("Venus", "Jupiter", "Mercury"),
        ("Atlantic Ocean", "Indian Ocean", "Arctic Ocean"),
        ("Michelangelo", "Raphael", "Vincent van Gogh"),
        ("212", "90", "273"),
        ("1991", "1987", "1993"),
        ("Amazon", "Yangtze", "Mississippi"),
        ("Isaac Newton", "Niels Bohr", "Max Planck"),
        ("yuan", "won", "baht"),
        ("212", "198", "230"),
        ("oxygen", "nitrogen", "hydrogen"),
        ("Charlotte Bronte", "Emily Bronte", "Mary Shelley"),
        ("1", "3", "0"),
        ("helium", "oxygen", "carbon"),
    ]
    tasks: list[Task] = []
    for i in range(n):
        base_idx = i % len(facts)
        question, gold, difficulty = facts[base_idx]
        options = list(distractors[base_idx])
        rng.shuffle(options)
        choices = [gold] + options[:3]
        rng.shuffle(choices)
        letter = "ABCD"[choices.index(gold)]
        rendered = "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(choices))
        tasks.append(
            Task(
                task_id=f"qa-{i:03d}",
                prompt=f"{question}\n{rendered}\nReply with the letter of the correct option.",
                reference=letter,
                domain="qa",
                difficulty=difficulty,
                metadata={"answer_kind": "choice", "source": "builtin-qa"},
            )
        )
    return tasks


def _math_tasks(n: int, rng: random.Random) -> list[Task]:
    tasks: list[Task] = []
    for i in range(n):
        kind = i % 5
        difficulty = 0.25 + 0.55 * rng.random()
        if kind == 0:
            a, b = rng.randint(12, 98), rng.randint(12, 98)
            prompt = f"Compute {a} * {b}."
            gold = str(a * b)
        elif kind == 1:
            a = rng.randint(2, 12)
            b = rng.randint(2, 5)
            prompt = f"Compute {a}**{b}."
            gold = str(a**b)
        elif kind == 2:
            n_terms = rng.randint(5, 30)
            prompt = f"What is the sum of the first {n_terms} positive integers?"
            gold = str(n_terms * (n_terms + 1) // 2)
        elif kind == 3:
            x, y = rng.randint(2, 9), rng.randint(2, 9)
            prompt = f"Solve for x: {x}x + {y} = {x * 4 + y}."
            gold = "4"
        else:
            p = rng.randint(2, 6)
            q = rng.randint(3, 9)
            prompt = f"A rectangle has width {p} and height {q}. What is its area?"
            gold = str(p * q)
        tasks.append(
            Task(
                task_id=f"math-{i:03d}",
                prompt=prompt,
                reference=gold,
                domain="math",
                difficulty=difficulty,
                metadata={"answer_kind": "number", "source": "builtin-math"},
            )
        )
    return tasks


_CODE_TEMPLATES: list[tuple[str, str, str]] = [
    (
        "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"",
        "    return a + b",
        "add(2, 3) == 5",
    ),
    (
        "def is_even(n):\n    \"\"\"Return True when n is even.\"\"\"",
        "    return n % 2 == 0",
        "is_even(4) is True",
    ),
    (
        "def max_of_three(a, b, c):\n    \"\"\"Return the largest of three numbers.\"\"\"",
        "    return max(a, b, c)",
        "max_of_three(1, 9, 4) == 9",
    ),
    (
        "def reverse_string(s):\n    \"\"\"Return s reversed.\"\"\"",
        "    return s[::-1]",
        "reverse_string('abc') == 'cba'",
    ),
    (
        "def factorial(n):\n    \"\"\"Return n! for n >= 0.\"\"\"",
        "    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
        "factorial(5) == 120",
    ),
    (
        "def count_vowels(s):\n    \"\"\"Return the number of vowels in s.\"\"\"",
        "    return sum(1 for ch in s.lower() if ch in 'aeiou')",
        "count_vowels('hello') == 2",
    ),
    (
        "def fib(n):\n    \"\"\"Return the n-th Fibonacci number, fib(0) == 0.\"\"\"",
        "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "fib(7) == 13",
    ),
    (
        "def unique(items):\n    \"\"\"Return items with duplicates removed, order preserved.\"\"\"",
        "    seen = []\n    for item in items:\n        if item not in seen:\n            seen.append(item)\n    return seen",
        "unique([1, 2, 1]) == [1, 2]",
    ),
]


def _code_tasks(n: int, rng: random.Random) -> list[Task]:
    tasks: list[Task] = []
    for i in range(n):
        signature, body, assertion = _CODE_TEMPLATES[i % len(_CODE_TEMPLATES)]
        difficulty = 0.30 + 0.45 * rng.random()
        tasks.append(
            Task(
                task_id=f"code-{i:03d}",
                prompt=(
                    "Complete the Python function below. Reply with the full function body "
                    "as the final answer.\n\n"
                    f"{signature}\n"
                    f"    # assertion: {assertion}"
                ),
                reference=body.strip(),
                domain="code",
                difficulty=difficulty,
                metadata={"answer_kind": "code", "source": "builtin-code"},
            )
        )
    return tasks


def _mixed_tasks(n: int, rng: random.Random) -> list[Task]:
    """50% QA, 30% math, 20% code — a plausible production traffic mix."""
    n_qa = int(round(n * 0.50))
    n_math = int(round(n * 0.30))
    n_code = n - n_qa - n_math
    tasks = _qa_tasks(n_qa, rng) + _math_tasks(n_math, rng) + _code_tasks(n_code, rng)
    rng.shuffle(tasks)
    # Re-index so ids stay stable per set rather than per generator.
    return [
        Task(
            task_id=f"{task.task_id.split('-')[0]}-{i:03d}",
            prompt=task.prompt,
            reference=task.reference,
            domain=task.domain,
            difficulty=task.difficulty,
            metadata=dict(task.metadata),
        )
        for i, task in enumerate(tasks)
    ]


BUILTIN: dict[str, BuiltinSpec] = {
    "builtin-mixed": BuiltinSpec("builtin-mixed", "50% QA / 30% math / 20% code", _mixed_tasks),
    "builtin-qa": BuiltinSpec("builtin-qa", "multiple-choice QA", _qa_tasks),
    "builtin-math": BuiltinSpec("builtin-math", "arithmetic and algebra", _math_tasks),
    "builtin-code": BuiltinSpec("builtin-code", "HumanEval-style function completion", _code_tasks),
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_builtin(name: str, n_tasks: int, seed: int) -> list[Task]:
    try:
        spec = BUILTIN[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown builtin task set {name!r}; known: {', '.join(sorted(BUILTIN))}"
        ) from exc
    rng = random.Random(f"{seed}:{name}")
    tasks = spec.factory(n_tasks, rng)
    if len(tasks) < n_tasks:  # pragma: no cover - generators always satisfy n
        raise RuntimeError(f"generator {name} produced {len(tasks)} < {n_tasks} tasks")
    return tasks[:n_tasks]


def load_jsonl(path: str | Path, n_tasks: int | None = None) -> list[Task]:
    """Load a user-supplied labelled set (one JSON object per line)."""
    tasks: list[Task] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            payload.setdefault("task_id", f"task-{len(tasks):04d}")
            payload.setdefault("reference", payload.get("answer", ""))
            tasks.append(Task.from_dict(payload))
            if n_tasks is not None and len(tasks) >= n_tasks:
                break
    if not tasks:
        raise ValueError(f"{path}: no tasks found")
    return tasks


def load_humaneval(path: str | Path, n_tasks: int | None = None) -> list[Task]:
    """Adapt HumanEval's official JSONL (``prompt`` + ``canonical_solution``)."""
    tasks: list[Task] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            entry_point = payload.get("entry_point", f"task_{lineno}")
            tasks.append(
                Task(
                    task_id=str(payload.get("task_id", f"HumanEval/{lineno}")),
                    prompt=str(payload.get("prompt", "")),
                    reference=str(payload.get("canonical_solution", "")).strip(),
                    domain="code",
                    difficulty=0.55,
                    metadata={
                        "answer_kind": "code",
                        "source": "humaneval",
                        "entry_point": entry_point,
                        "test": payload.get("test", ""),
                    },
                )
            )
            if n_tasks is not None and len(tasks) >= n_tasks:
                break
    if not tasks:
        raise ValueError(f"{path}: no HumanEval tasks found")
    return tasks


def load_tasks(task_set: str, n_tasks: int, seed: int) -> list[Task]:
    """Resolve ``--task-set``: a builtin name, or a path to JSONL."""
    if task_set in BUILTIN:
        return load_builtin(task_set, n_tasks, seed)
    path = Path(task_set)
    if not path.exists():
        raise FileNotFoundError(f"task set not found: {task_set!r} (not a builtin name, not a file)")
    if "humaneval" in path.name.lower():
        return load_humaneval(path, n_tasks)
    return load_jsonl(path, n_tasks)


def task_set_summary(tasks: Sequence[Task]) -> dict[str, Any]:
    """Domain mix and difficulty profile — quoted in the report header."""
    by_domain: dict[str, int] = {}
    difficulties = []
    for task in tasks:
        by_domain[task.domain] = by_domain.get(task.domain, 0) + 1
        difficulties.append(float(task.difficulty))
    mean_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0.0
    return {
        "n_tasks": len(tasks),
        "domains": dict(sorted(by_domain.items())),
        "mean_difficulty": round(mean_difficulty, 4),
        "min_difficulty": round(min(difficulties), 4) if difficulties else 0.0,
        "max_difficulty": round(max(difficulties), 4) if difficulties else 0.0,
    }
