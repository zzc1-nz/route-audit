"""Command line interface: ``python -m routeaudit``.

Subcommands
-----------
``audit``    run the audit and persist the raw JSON (no figures)
``report``   run the audit (or load one) and render the Markdown report + figures
``analyse``  re-analyse a saved ``audit.json`` without spending anything
``compare``  print a compact table to stdout

Design note: ``audit`` and ``report`` are separated so that a reviewer can run
the expensive part once (or point at someone else's run) and still regenerate
the report/figures for free.  Every number in a report is derivable from the
persisted ``audit.json`` — that is the reproducibility claim of this sandbox.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .accounting import summarise_audit
from .analysis import analyse
from .config import (
    CATALOGUE,
    DEFAULT_CHEAP,
    DEFAULT_PREMIUM,
    DEFAULT_THRESHOLDS,
    SWEEP_THRESHOLDS,
    AuditConfig,
    thresholds_from_csv,
)
from .engine import run_audit
from .providers import make_provider
from .taskset import BUILTIN, load_tasks, task_set_summary
from .types import AuditRun, Strategy


def _console_safe() -> None:
    """Make stdout/stderr survive non-UTF-8 Windows consoles.

    The CLI prints ``·``, ``→`` and box characters; a cp936 console would raise
    ``UnicodeEncodeError`` and kill an otherwise successful audit.  Reconfigure
    to UTF-8 when possible, else fall back to replacing unencodable characters.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - very old/odd streams
            pass


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #
class ProgressPrinter:
    """Single-line progress with throttling, so long runs stay informative."""

    def __init__(self, enabled: bool = True, interval_s: float = 0.5) -> None:
        self.enabled = enabled
        self.interval_s = interval_s
        self._last = 0.0
        self._last_phase = ""

    def __call__(self, phase: str, done: int, total: int) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if phase == self._last_phase and done < total and now - self._last < self.interval_s:
            return
        self._last = now
        self._last_phase = phase
        pct = (done / total * 100.0) if total else 100.0
        bar_width = 28
        filled = int(bar_width * done / total) if total else bar_width
        bar = "#" * filled + "." * (bar_width - filled)
        sys.stderr.write(f"\r  [{bar}] {phase:<14s} {done:>5d}/{total:<5d} {pct:5.1f}%")
        if done >= total:
            sys.stderr.write("\n")
        sys.stderr.flush()


# --------------------------------------------------------------------------- #
# Audit construction
# --------------------------------------------------------------------------- #
def build_config(args: argparse.Namespace) -> AuditConfig:
    thresholds = (
        SWEEP_THRESHOLDS if getattr(args, "sweep", False) else thresholds_from_csv(args.thresholds)
    )
    return AuditConfig(
        task_set=args.task_set,
        n_tasks=args.n_tasks,
        seed=args.seed,
        cheap_model=args.cheap_model,
        premium_model=args.premium_model,
        thresholds=thresholds,
        max_workers=args.workers,
        price_scale=args.price_scale,
        max_budget_usd=args.max_budget,
        format_gate=not getattr(args, "no_format_gate", False),
        enable_cache=not getattr(args, "no_cache", False),
        include_strategies=tuple(s.strip().upper() for s in args.strategies.split(",")),
    )


def execute(args: argparse.Namespace, *, progress: bool = True) -> AuditRun:
    """Run the audit described by CLI args and print a one-screen summary."""
    config = build_config(args)
    tasks = load_tasks(config.task_set, config.n_tasks, config.seed)
    provider = make_provider(args.provider)

    summary = task_set_summary(tasks)
    print(f"route-audit {__version__} · provider={provider.name} · {config.n_tasks} tasks")
    print(
        f"  cheap={config.cheap_model} (${config.cheap.input_price_per_mtok}/${config.cheap.output_price_per_mtok} per Mtok)"
        f"  premium={config.premium_model} (${config.premium.input_price_per_mtok}/${config.premium.output_price_per_mtok} per Mtok)"
    )
    print(
        f"  thresholds={', '.join(f'{t:.1f}' for t in config.thresholds)} · "
        f"strategies={','.join(config.include_strategies)} · seed={config.seed}"
    )
    print(f"  domains={summary['domains']} · mean difficulty={summary['mean_difficulty']}")
    print(f"  planned call ceiling={config.planned_calls} · budget cap=${config.max_budget_usd:.2f}")
    if provider.name != "mock":
        print("  WARNING: live provider — this will spend real tokens and real money.")
    print()

    started = time.perf_counter()
    audit = run_audit(config, tasks=tasks, provider=provider, progress=ProgressPrinter(progress))
    elapsed = time.perf_counter() - started
    audit.stats = dict(audit.stats)
    audit.stats["wall_clock_s"] = round(elapsed, 3)

    print(
        f"  done in {elapsed:.1f}s · {audit.stats['executed_calls']} provider calls "
        f"(${audit.stats['executed_cost_usd']:.4f} executed) · "
        f"{audit.stats['cache_hits']} cache hits"
    )
    return audit


def print_comparison(audit: AuditRun, threshold: float | None = None) -> None:
    """Compact stdout table — the fastest way to see the headline result."""
    comparison = analyse(audit, threshold=threshold).comparison
    header = f"{'configuration':<22s} {'acc':>7s} {'tokens':>10s} {'cost USD':>10s} {'tok/correct':>12s} {'waste%':>7s}"
    print()
    print(header)
    print("-" * len(header))
    for summary in comparison.ordered():
        print(
            f"{summary.label:<22s} {summary.accuracy:>7.3f} {summary.total_tokens:>10,d} "
            f"{summary.total_cost_usd:>10.4f} {summary.tokens_per_correct:>12,.0f} "
            f"{summary.wasted_token_share * 100:>6.1f}%"
        )
    print()
    print(
        f"cascade vs premium: {comparison.cascade_savings_vs_premium * 100:.1f}% cheaper, "
        f"accuracy {comparison.cascade_accuracy_loss_vs_premium * 100:.1f} pp lower"
    )
    print(
        f"cascade vs cheap:   {comparison.cascade_cost_multiple_vs_cheap:.2f}x cost, "
        f"accuracy {comparison.cascade_accuracy_gain_vs_cheap * 100:+.1f} pp higher"
    )


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_audit(args: argparse.Namespace) -> int:
    audit = execute(args, progress=not args.quiet)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.save(str(out))
    print(f"\n  saved raw audit -> {out}")
    print_comparison(audit, threshold=args.threshold_for_headline)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .charts import render_all
    from .report import write_report

    if args.from_audit:
        audit = AuditRun.load(args.from_audit)
        print(f"loaded audit {audit.run_id} from {args.from_audit} (no tokens spent)")
    else:
        audit = execute(args, progress=not args.quiet)
        if args.save_audit:
            audit.save(args.save_audit)
            print(f"  saved raw audit -> {args.save_audit}")

    analysis = analyse(audit, threshold=args.threshold_for_headline)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, str] = {}
    if not args.no_figures:
        rendered = render_all(analysis, out_dir)
        figure_paths = {name: os.path.basename(str(path)) for name, path in rendered.items()}
        for name, path in rendered.items():
            print(f"  figure: {name} -> {path}")

    written = write_report(analysis, out_dir, figures=figure_paths)
    print(f"\n  report  -> {written['markdown']}")
    for name, path in (written.get("figures") or {}).items():
        print(f"  figure  -> {path}")
    print_comparison(audit, threshold=args.threshold_for_headline)
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    audit = AuditRun.load(args.audit)
    analysis = analyse(audit, threshold=args.threshold_for_headline)
    payload = analysis.to_dict()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  analysis -> {args.out}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    audit = AuditRun.load(args.audit)
    print_comparison(audit, threshold=args.threshold_for_headline)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    print(f"{'model':<16s} {'tier':<8s} {'in $/Mtok':>10s} {'out $/Mtok':>11s}  notes")
    print("-" * 78)
    for spec in CATALOGUE.values():
        print(
            f"{spec.name:<16s} {spec.tier:<8s} {spec.input_price_per_mtok:>10.2f} "
            f"{spec.output_price_per_mtok:>11.2f}  {spec.notes}"
        )
    return 0


def cmd_tasksets(args: argparse.Namespace) -> int:
    print("builtin task sets:")
    for name, spec in BUILTIN.items():
        print(f"  {name:<18s} {spec.description}")
    print("\nYou can also pass a path to a JSONL file with fields:")
    print("  task_id, prompt, reference, domain?, difficulty?, metadata?")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task-set", default="builtin-mixed", help="builtin name or path to JSONL")
    p.add_argument("--n-tasks", type=int, default=200, help="number of audit tasks (default 200)")
    p.add_argument("--seed", type=int, default=20260101, help="deterministic seed")
    p.add_argument("--cheap-model", default=DEFAULT_CHEAP, choices=sorted(CATALOGUE))
    p.add_argument("--premium-model", default=DEFAULT_PREMIUM, choices=sorted(CATALOGUE))
    p.add_argument(
        "--thresholds",
        default=",".join(f"{t:.1f}" for t in DEFAULT_THRESHOLDS),
        help="cascade thresholds, comma separated (default 0.5,0.7,0.3)",
    )
    p.add_argument("--sweep", action="store_true", help="use the full 0.1..0.9 threshold ladder")
    p.add_argument("--strategies", default="A,B,C", help="subset of A,B,C")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--price-scale", type=float, default=1.0, help="multiply all prices (discount/contract)")
    p.add_argument("--max-budget", type=float, default=25.0, help="hard USD cap on the audit")
    p.add_argument("--provider", default="mock", choices=["mock", "api"])
    p.add_argument("--no-format-gate", action="store_true", help="confidence gate only")
    p.add_argument("--no-cache", action="store_true", help="disable the call cache")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    p.add_argument(
        "--threshold-for-headline",
        type=float,
        default=0.5,
        help="which cascade threshold the headline numbers use",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="routeaudit",
        description="Offline audit of model-routing economics: what did routing mistakes cost?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m routeaudit audit  --n-tasks 200 --out runs/audit.json\n"
            "  python -m routeaudit report --sweep --out-dir out/\n"
            "  python -m routeaudit report --from-audit runs/audit.json --out-dir out/\n"
            "  python -m routeaudit compare runs/audit.json\n"
            "  python -m routeaudit models\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"route-audit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="run the audit and save raw JSON")
    _add_common(p_audit)
    p_audit.add_argument("--out", default="runs/audit.json")
    p_audit.set_defaults(func=cmd_audit)

    p_report = sub.add_parser("report", help="render the Markdown report and figures")
    _add_common(p_report)
    p_report.add_argument("--out-dir", default="out")
    p_report.add_argument("--from-audit", default=None, help="reuse a saved audit.json (no tokens spent)")
    p_report.add_argument("--save-audit", default=None, help="also persist the raw JSON here")
    p_report.add_argument("--no-figures", action="store_true")
    p_report.set_defaults(func=cmd_report)

    p_analyse = sub.add_parser("analyse", help="re-analyse a saved audit.json")
    p_analyse.add_argument("audit")
    p_analyse.add_argument("--out", default=None)
    p_analyse.add_argument("--threshold-for-headline", type=float, default=0.5)
    p_analyse.set_defaults(func=cmd_analyse)

    p_compare = sub.add_parser("compare", help="print the configuration table")
    p_compare.add_argument("audit")
    p_compare.add_argument("--threshold-for-headline", type=float, default=0.5)
    p_compare.set_defaults(func=cmd_compare)

    p_models = sub.add_parser("models", help="list the model catalogue and prices")
    p_models.set_defaults(func=cmd_models)

    p_sets = sub.add_parser("tasksets", help="list builtin task sets")
    p_sets.set_defaults(func=cmd_tasksets)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
