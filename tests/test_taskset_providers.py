"""Tests for ``routeaudit.taskset`` and ``routeaudit.providers``.

The task set is *data*, and the mock provider is the deterministic engine behind
every offline audit, so both need to be pinned: if either drifts, every number
in the report drifts with it.
"""

from __future__ import annotations

import json
import os
import shutil
import unittest
import uuid
from pathlib import Path

from routeaudit.config import CATALOGUE, ModelSpec
from routeaudit.providers import (
    ANSWER_FORMAT,
    MockProvider,
    build_prompt,
    build_verifier_prompt,
    make_provider,
    stable_hash,
)
from routeaudit.taskset import (
    BUILTIN,
    load_builtin,
    load_humaneval,
    load_jsonl,
    load_tasks,
    task_set_summary,
)
from routeaudit.types import Task, Usage

SCRATCH_ROOT = Path(__file__).resolve().parent / "_tmp"


def make_scratch_dir() -> Path:
    os.makedirs(SCRATCH_ROOT, exist_ok=True)
    path = SCRATCH_ROOT / uuid.uuid4().hex[:12]
    os.makedirs(path, exist_ok=True)
    return path


def remove_scratch_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Builtin task sets
# --------------------------------------------------------------------------- #
class BuiltinTaskSetTests(unittest.TestCase):
    def test_every_builtin_produces_the_requested_count(self):
        for name in BUILTIN:
            tasks = load_builtin(name, 10, seed=1)
            self.assertEqual(len(tasks), 10, msg=name)

    def test_same_seed_reproduces_the_same_tasks(self):
        first = load_builtin("builtin-mixed", 20, seed=123)
        second = load_builtin("builtin-mixed", 20, seed=123)
        self.assertEqual([t.to_dict() for t in first], [t.to_dict() for t in second])

    def test_different_seed_changes_the_tasks(self):
        first = load_builtin("builtin-math", 10, seed=1)
        second = load_builtin("builtin-math", 10, seed=2)
        self.assertNotEqual([t.to_dict() for t in first], [t.to_dict() for t in second])

    def test_task_ids_are_unique(self):
        tasks = load_builtin("builtin-mixed", 40, seed=5)
        self.assertEqual(len({t.task_id for t in tasks}), len(tasks))

    def test_mixed_set_contains_all_three_domains(self):
        tasks = load_builtin("builtin-mixed", 20, seed=5)
        self.assertEqual({t.domain for t in tasks}, {"qa", "math", "code"})

    def test_mixed_set_mix_ratio_is_roughly_as_documented(self):
        tasks = load_builtin("builtin-mixed", 100, seed=5)
        summary = task_set_summary(tasks)["domains"]
        self.assertEqual(summary["qa"], 50)
        self.assertEqual(summary["math"], 30)
        self.assertEqual(summary["code"], 20)

    def test_qa_tasks_are_choice_questions_with_letter_references(self):
        for task in load_builtin("builtin-qa", 8, seed=3):
            self.assertEqual(task.metadata["answer_kind"], "choice")
            self.assertIn(task.reference, list("ABCD"))

    def test_math_tasks_are_numeric(self):
        for task in load_builtin("builtin-math", 10, seed=3):
            self.assertEqual(task.metadata["answer_kind"], "number")
            float(task.reference)  # must parse as a number

    def test_code_tasks_are_code(self):
        for task in load_builtin("builtin-code", 5, seed=3):
            self.assertEqual(task.metadata["answer_kind"], "code")
            self.assertTrue(task.reference.strip())

    def test_all_difficulties_are_in_range(self):
        for name in BUILTIN:
            for task in load_builtin(name, 15, seed=9):
                self.assertGreaterEqual(task.difficulty, 0.0, msg=name)
                self.assertLessEqual(task.difficulty, 1.0, msg=name)

    def test_unknown_builtin_raises(self):
        with self.assertRaises(KeyError):
            load_builtin("nope", 1, seed=1)

    def test_load_tasks_resolves_builtin_names(self):
        tasks = load_tasks("builtin-qa", 5, seed=1)
        self.assertEqual(len(tasks), 5)
        self.assertEqual({t.domain for t in tasks}, {"qa"})

    def test_load_tasks_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_tasks("definitely-not-here.jsonl", 5, seed=1)


class TaskSetSummaryTests(unittest.TestCase):
    def test_summary_counts_domains_and_difficulty_range(self):
        tasks = [
            Task(task_id="a", prompt="p", reference="r", domain="qa", difficulty=0.2),
            Task(task_id="b", prompt="p", reference="r", domain="qa", difficulty=0.4),
            Task(task_id="c", prompt="p", reference="r", domain="math", difficulty=0.9),
        ]
        summary = task_set_summary(tasks)
        self.assertEqual(summary["n_tasks"], 3)
        self.assertEqual(summary["domains"], {"math": 1, "qa": 2})
        self.assertAlmostEqual(summary["mean_difficulty"], 0.5)
        self.assertAlmostEqual(summary["min_difficulty"], 0.2)
        self.assertAlmostEqual(summary["max_difficulty"], 0.9)

    def test_empty_summary_is_zeroed(self):
        summary = task_set_summary([])
        self.assertEqual(summary["n_tasks"], 0)
        self.assertEqual(summary["mean_difficulty"], 0.0)


# --------------------------------------------------------------------------- #
# JSONL / HumanEval loaders
# --------------------------------------------------------------------------- #
class JsonlLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = make_scratch_dir()
        self.path = self.scratch / "tasks.jsonl"

    def tearDown(self) -> None:
        remove_scratch_dir(self.scratch)

    def test_loads_explicit_tasks(self):
        self.path.write_text(
            json.dumps(
                {
                    "task_id": "j-1",
                    "prompt": "What is 2 + 2?",
                    "reference": "4",
                    "domain": "math",
                    "difficulty": 0.1,
                    "metadata": {"answer_kind": "number"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        tasks = load_jsonl(str(self.path))
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.task_id, "j-1")
        self.assertEqual(task.reference, "4")
        self.assertEqual(task.domain, "math")
        self.assertEqual(task.metadata["answer_kind"], "number")

    def test_defaults_are_applied_for_missing_fields(self):
        self.path.write_text(
            json.dumps({"prompt": "Capital of France?", "answer": "Paris"}) + "\n",
            encoding="utf-8",
        )
        task = load_jsonl(str(self.path))[0]
        self.assertEqual(task.reference, "Paris")  # falls back to "answer"
        self.assertEqual(task.domain, "qa")
        self.assertTrue(task.task_id)

    def test_blank_lines_and_comments_are_skipped(self):
        self.path.write_text(
            "// a comment\n\n"
            + json.dumps({"task_id": "j-1", "prompt": "p", "reference": "r"})
            + "\n\n",
            encoding="utf-8",
        )
        self.assertEqual(len(load_jsonl(str(self.path))), 1)

    def test_n_tasks_limits_the_result(self):
        lines = [
            json.dumps({"task_id": f"j-{i}", "prompt": "p", "reference": "r"})
            for i in range(5)
        ]
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertEqual(len(load_jsonl(str(self.path), n_tasks=2)), 2)

    def test_invalid_json_raises_value_error_with_line_number(self):
        self.path.write_text("{not json}\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            load_jsonl(str(self.path))
        self.assertIn("1", str(ctx.exception))

    def test_empty_file_raises(self):
        self.path.write_text("\n\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_jsonl(str(self.path))

    def test_load_tasks_dispatches_to_jsonl_by_path(self):
        self.path.write_text(
            json.dumps({"task_id": "j-1", "prompt": "p", "reference": "r"}) + "\n",
            encoding="utf-8",
        )
        self.assertEqual(len(load_tasks(str(self.path), 5, seed=1)), 1)

    def test_humaneval_adapter_maps_canonical_solution(self):
        path = self.scratch / "HumanEval.jsonl"
        path.write_text(
            json.dumps(
                {
                    "task_id": "HumanEval/0",
                    "prompt": "def add(a, b):\n",
                    "canonical_solution": "    return a + b\n",
                    "entry_point": "add",
                    "test": "assert add(1, 2) == 3",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        tasks = load_humaneval(str(path))
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.task_id, "HumanEval/0")
        self.assertEqual(task.reference, "return a + b")
        self.assertEqual(task.domain, "code")
        self.assertEqual(task.metadata["answer_kind"], "code")
        self.assertEqual(task.metadata["entry_point"], "add")

    def test_load_tasks_dispatches_humaneval_by_filename(self):
        path = self.scratch / "HumanEval.jsonl"
        path.write_text(
            json.dumps({"prompt": "def f():\n", "canonical_solution": "    pass"}) + "\n",
            encoding="utf-8",
        )
        tasks = load_tasks(str(path), 5, seed=1)
        self.assertEqual(tasks[0].domain, "code")


# --------------------------------------------------------------------------- #
# Mock provider determinism and shape
# --------------------------------------------------------------------------- #
class MockProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = MockProvider()
        self.task = Task(
            task_id="qa-000",
            prompt="What is the capital of France?",
            reference="Paris",
            domain="qa",
            difficulty=0.2,
            metadata={"answer_kind": "text"},
        )

    def test_same_inputs_reproduce_the_same_response(self):
        first = self.provider.complete(
            self.task, CATALOGUE["gpt-4o-mini"], stage="primary", seed=7
        )
        second = self.provider.complete(
            self.task, CATALOGUE["gpt-4o-mini"], stage="primary", seed=7
        )
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.usage, second.usage)

    def test_different_seed_changes_the_response(self):
        first = self.provider.complete(
            self.task, CATALOGUE["gpt-4o-mini"], stage="primary", seed=1
        )
        second = self.provider.complete(
            self.task, CATALOGUE["gpt-4o-mini"], stage="primary", seed=2
        )
        self.assertNotEqual(
            (first.text, first.usage), (second.text, second.usage)
        )

    def test_stage_changes_the_draw(self):
        primary = self.provider.complete(
            self.task, CATALOGUE["gpt-4o"], stage="primary", seed=7
        )
        escalated = self.provider.complete(
            self.task, CATALOGUE["gpt-4o"], stage="escalated", seed=7
        )
        self.assertNotEqual(primary.usage, escalated.usage)

    def test_response_is_a_valid_model_response(self):
        response = self.provider.complete(
            self.task, CATALOGUE["gpt-4o-mini"], stage="primary", seed=3
        )
        self.assertEqual(response.model, "gpt-4o-mini")
        self.assertEqual(response.provider, "mock")
        self.assertIsNone(response.error)
        self.assertTrue(response.ok)
        self.assertGreater(response.usage.total_tokens, 0)
        self.assertGreater(response.latency_ms, 0.0)

    def test_well_formed_responses_use_the_answer_format(self):
        task = Task(
            task_id="qa-001",
            prompt="Who wrote 1984?",
            reference="George Orwell",
            difficulty=0.1,
        )
        response = self.provider.complete(
            task, CATALOGUE["gpt-4o-mini"], stage="primary", seed=11
        )
        if response.raw["malformed"]:
            self.skipTest("this draw is one of the ~6% deliberately malformed ones")
        self.assertIn("Answer:", response.text)
        self.assertIn("Confidence:", response.text)

    def test_p_correct_decreases_with_difficulty(self):
        easy = Task(task_id="e", prompt="p", reference="r", difficulty=0.1)
        hard = Task(task_id="h", prompt="p", reference="r", difficulty=0.9)
        spec = CATALOGUE["gpt-4o-mini"]
        self.assertGreater(self.provider.p_correct(spec, easy), self.provider.p_correct(spec, hard))

    def test_confidence_is_clipped_into_the_unit_interval(self):
        for seed in range(30):
            response = self.provider.complete(
                self.task, CATALOGUE["gpt-4o-mini"], stage="primary", seed=seed
            )
            if response.raw["malformed"]:
                continue
            confidence = float(response.text.rsplit("Confidence:", 1)[1])
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_escalated_usage_is_larger_than_primary_on_average(self):
        primary_tokens = []
        escalated_tokens = []
        for index in range(40):
            task = Task(task_id=f"t-{index}", prompt="p", reference="r", difficulty=0.5)
            spec = CATALOGUE["gpt-4o"]
            primary_tokens.append(
                self.provider.complete(task, spec, stage="primary", seed=1).usage.total_tokens
            )
            escalated_tokens.append(
                self.provider.complete(task, spec, stage="escalated", seed=1).usage.total_tokens
            )
        self.assertGreater(sum(escalated_tokens), sum(primary_tokens))

    def test_draft_is_acknowledged_but_not_required(self):
        with_draft = self.provider.complete(
            self.task, CATALOGUE["gpt-4o"], stage="escalated", seed=5, draft="Answer: Lyon"
        )
        self.assertTrue(with_draft.raw["has_draft"])
        without_draft = self.provider.complete(
            self.task, CATALOGUE["gpt-4o"], stage="escalated", seed=5
        )
        self.assertFalse(without_draft.raw["has_draft"])

    def test_malformed_rate_produces_unparseable_text(self):
        provider = MockProvider(malformed_rate=1.0)
        response = provider.complete(
            self.task, CATALOGUE["gpt-4o-mini"], stage="primary", seed=1
        )
        self.assertNotIn("Answer:", response.text)
        self.assertTrue(response.raw["malformed"])

    def test_zero_malformed_rate_always_produces_the_format(self):
        provider = MockProvider(malformed_rate=0.0)
        for seed in range(20):
            response = provider.complete(
                self.task, CATALOGUE["gpt-4o-mini"], stage="primary", seed=seed
            )
            self.assertIn("Answer:", response.text)


class StableHashTests(unittest.TestCase):
    def test_hash_is_stable_across_calls(self):
        self.assertEqual(stable_hash("a", 1), stable_hash("a", 1))

    def test_hash_distinguishes_arguments(self):
        self.assertNotEqual(stable_hash("a", 1), stable_hash("a", 2))

    def test_hash_is_not_python_hash(self):
        # Python's hash() is salted per process; the audit needs a stable value
        self.assertIsInstance(stable_hash("x"), int)
        self.assertGreaterEqual(stable_hash("x"), 0)


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = Task(task_id="t", prompt="What is 2+2?", reference="4")

    def test_prompt_states_the_expected_format(self):
        prompt = build_prompt(self.task)
        self.assertIn(ANSWER_FORMAT, prompt)
        self.assertIn(self.task.prompt, prompt)

    def test_verifier_prompt_includes_the_draft(self):
        prompt = build_verifier_prompt(self.task, "Answer: 5")
        self.assertIn("Answer: 5", prompt)
        self.assertIn(self.task.prompt, prompt)
        self.assertIn("escalation", prompt.lower())


class ProviderFactoryTests(unittest.TestCase):
    def test_mock_aliases_resolve_to_the_mock_provider(self):
        for alias in ("mock", "offline", "sim"):
            self.assertIsInstance(make_provider(alias), MockProvider)

    def test_api_provider_is_constructed_without_network_access(self):
        provider = make_provider("api")
        self.assertEqual(provider.name, "api")
        self.assertFalse(hasattr(provider, "complete") is False)  # has the interface

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            make_provider("gpt-5")


if __name__ == "__main__":
    unittest.main()
