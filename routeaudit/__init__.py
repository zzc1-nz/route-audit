"""route-audit — offline audit sandbox for model-routing economics.

The package answers one question that budget/guardrail routers do not:
*how much money did wrong routing decisions cost?*

Layers
------
``types``        shared data contracts (Task, Attempt, TaskRun, AuditRun, ...)
``config``       model catalogue, prices, audit configuration
``pricing``      token -> USD conversion
``providers``    model adapters: deterministic offline simulator + real APIs
``grading``      label-aware answer extraction and scoring
``taskset``      built-in labelled task sets and JSONL/HumanEval loaders
``confidence``   label-free escalation signal
``policies``     strategies A (cheap), B (premium), C (cascade)
``accounting``   per-step token attribution
``audit``        parallel, reproducible execution plan + budget guard
``analysis``     over-escalation / over-downgrade rates, sweeps, regret
``report``       markdown report and figures
``cli``          ``python -m routeaudit`` entry point
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
