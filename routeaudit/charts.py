"""Matplotlib figures for the route-audit report.

Design rules
------------
* **Agg only.**  The audit runs headless (CI, cron, a laptop with no display), so
  the non-interactive backend is selected before ``pyplot`` is imported.
* **No invented numbers.**  Every value plotted is read from the
  :class:`~routeaudit.analysis.AuditAnalysis` passed in; this module owns colours,
  layout and labels, never results.
* **No tofu boxes.**  Chinese labels are used only when a CJK-capable font is
  actually installed (Noto Sans SC, Microsoft YaHei, SimHei, ...).  Otherwise
  every label falls back to English, so a missing font degrades to readable
  English rather than to a grid of hollow squares.  ``axes.unicode_minus`` is
  disabled so minus signs render with the CJK font as well.
* **Import is side-effect free.**  rcParams are mutated only inside
  :func:`_apply_style`, which each plot calls for itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

# The backend must be chosen before pyplot is imported for the first time.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from matplotlib import font_manager  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .accounting import ConfigSummary
    from .analysis import AuditAnalysis


# --------------------------------------------------------------------------- #
# Figure registry
# --------------------------------------------------------------------------- #
#: Canonical figure names.  ``routeaudit.report`` embeds figures by these keys,
#: so the two modules must agree on the spelling.
FIGURE_ACCURACY_VS_TOKENS = "accuracy_vs_tokens"
FIGURE_THRESHOLD_SENSITIVITY = "threshold_sensitivity"
FIGURE_COST_BREAKDOWN = "cost_breakdown"

FIGURE_NAMES: tuple[str, ...] = (
    FIGURE_ACCURACY_VS_TOKENS,
    FIGURE_THRESHOLD_SENSITIVITY,
    FIGURE_COST_BREAKDOWN,
)

#: Rasterisation density.  High enough for print, small enough that three PNGs
#: stay well under a megabyte.
_DPI = 170

#: Fonts known to carry Simplified Chinese glyphs, best first.
_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Noto Sans SC",
    "Microsoft YaHei",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "DengXian",
    "SimSun",
    "Arial Unicode MS",
)

_PALETTE: dict[str, str] = {
    "cheap": "#2f7d4f",
    "premium": "#b23a48",
    "cascade": "#1f5fa8",
    "operating": "#d97706",
    "useful": "#3f7d9e",
    "wasted_escalation": "#d97706",
    "wasted_downgrade": "#b23a48",
}


# --------------------------------------------------------------------------- #
# Font detection and styling
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _resolve_cjk_font() -> str | None:
    """Return the name of an installed CJK font, or ``None`` if there is none.

    Detection is by font family name against matplotlib's own font index, which
    is what actually decides whether a glyph can be rendered.
    """
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in _CJK_FONT_CANDIDATES:
        if candidate in installed:
            return candidate
    return None


def has_cjk_font() -> bool:
    """Whether Chinese labels can be rendered in this environment."""
    return _resolve_cjk_font() is not None


def _apply_style() -> None:
    """Install the shared rcParams (font stack, grid, sizes).

    Idempotent and cheap; called at the top of every plot so importing this
    module never mutates global matplotlib state.
    """
    font = _resolve_cjk_font()
    families = ([font] if font else []) + ["DejaVu Sans", "sans-serif"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": families,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 110,
            "savefig.dpi": _DPI,
            "axes.titlesize": 13.5,
            "axes.titleweight": "semibold",
            "axes.titlepad": 12,
            "axes.labelsize": 10.5,
            "axes.labelcolor": "#1f2933",
            "axes.edgecolor": "#8894a0",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.color": "#3e4c59",
            "ytick.color": "#3e4c59",
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.color": "#9aa5b1",
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# --------------------------------------------------------------------------- #
# Labels (Chinese when possible, English otherwise)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Labels:
    """Every string one chart needs, in a single language."""

    zh: bool
    # -- accuracy vs tokens ------------------------------------------------- #
    title_accuracy: str
    xlabel_tokens: str
    ylabel_accuracy: str
    series_a: str
    series_b: str
    series_c: str
    series_operating: str
    thr_fmt: str
    box_cascade_vs_premium: str
    # -- threshold sensitivity ---------------------------------------------- #
    title_sensitivity: str
    xlabel_threshold: str
    ylabel_share: str
    ylabel_cost: str
    series_cost: str
    series_acc: str
    series_waste: str
    ref_cheap_acc: str
    ref_premium_acc: str
    ref_operating: str
    box_precision: str
    # -- token breakdown ----------------------------------------------------- #
    title_breakdown: str
    xlabel_config: str
    ylabel_tokens: str
    series_useful: str
    series_wasted_escalation: str
    series_wasted_downgrade: str


def _labels() -> _Labels:
    """Build the label set for the detected language."""
    if has_cjk_font():
        return _Labels(
            zh=True,
            title_accuracy="准确率 vs 总 token 成本",
            xlabel_tokens="总 token（千）",
            ylabel_accuracy="准确率 (%)",
            series_a="A · 全廉价模型",
            series_b="B · 全高档模型",
            series_c="C · 级联路由（按阈值扫描）",
            series_operating="运营阈值点",
            thr_fmt="θ={:.1f}",
            box_cascade_vs_premium="级联 vs 全高档",
            title_sensitivity="阈值敏感性：成本、准确率与浪费占比",
            xlabel_threshold="升级阈值（越高越容易升级）",
            ylabel_share="比例 (%)",
            ylabel_cost="总成本 (USD)",
            series_cost="总成本（右轴）",
            series_acc="准确率（左轴）",
            series_waste="浪费 token 占比（左轴）",
            ref_cheap_acc="全廉价准确率",
            ref_premium_acc="全高档准确率",
            ref_operating="运营阈值",
            box_precision="升级精确率",
            title_breakdown="token 去向拆解：有效开销 vs 两类路由错误",
            xlabel_config="配置（下行数字为准确率）",
            ylabel_tokens="token（千）",
            series_useful="有效 token",
            series_wasted_escalation="过度升级浪费",
            series_wasted_downgrade="过度降级浪费",
        )
    return _Labels(
        zh=False,
        title_accuracy="Accuracy vs total token cost",
        xlabel_tokens="Total tokens (thousands)",
        ylabel_accuracy="Accuracy (%)",
        series_a="A · cheap-only",
        series_b="B · premium-only",
        series_c="C · cascade (threshold sweep)",
        series_operating="operating point",
        thr_fmt="t={:.1f}",
        box_cascade_vs_premium="Cascade vs premium",
        title_sensitivity="Threshold sensitivity: cost, accuracy and waste share",
        xlabel_threshold="Escalation threshold (higher = escalates more)",
        ylabel_share="Share (%)",
        ylabel_cost="Total cost (USD)",
        series_cost="Total cost (right axis)",
        series_acc="Accuracy (left axis)",
        series_waste="Wasted token share (left axis)",
        ref_cheap_acc="cheap-only accuracy",
        ref_premium_acc="premium-only accuracy",
        ref_operating="operating threshold",
        box_precision="escalation precision",
        title_breakdown="Where the tokens went: useful vs two kinds of routing error",
        xlabel_config="Configuration (second line = accuracy)",
        ylabel_tokens="Tokens (thousands)",
        series_useful="useful tokens",
        series_wasted_escalation="wasted escalation",
        series_wasted_downgrade="wasted downgrade",
    )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _config_tick(summary: "ConfigSummary") -> str:
    """Compact, language-neutral tick label for one configuration."""
    if summary.threshold is None:
        return summary.strategy.value
    return f"{summary.strategy.value}@{summary.threshold:.1f}"


def _money(value: float) -> str:
    """Format a USD amount with a sensible number of decimals."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"${value:,.3f}" if abs(value) < 10 else f"${value:,.2f}"


def _signed_pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def _save(fig: Figure, path: str | Path) -> Path:
    """Write one figure to disk and release it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def _cascade_summaries(analysis: "AuditAnalysis") -> list["ConfigSummary"]:
    """Cascade configurations in ascending threshold order."""
    rows = [s for s in analysis.comparison.ordered() if s.threshold is not None]
    return sorted(rows, key=lambda s: float(s.threshold or 0.0))


# --------------------------------------------------------------------------- #
# Figure 1 — accuracy vs tokens
# --------------------------------------------------------------------------- #
def plot_accuracy_vs_tokens(analysis: "AuditAnalysis", path: str | Path) -> Path:
    """Plot accuracy against total token spend for all three strategies.

    Strategies A and B are single points (one model for every task); strategy C
    is a curve, one point per swept threshold.  The operating threshold is
    ringed and the threshold value is written next to every cascade point.
    """
    labels = _labels()
    _apply_style()

    comparison = analysis.comparison
    cascades = _cascade_summaries(analysis)
    if not cascades:
        raise ValueError("analysis contains no cascade configuration to plot")
    operating = analysis.sensitivity.operating_threshold

    fig, ax = plt.subplots(figsize=(10.6, 6.6))

    xs = [s.total_tokens / 1000.0 for s in cascades]
    ys = [s.accuracy * 100.0 for s in cascades]
    ax.plot(
        xs,
        ys,
        color=_PALETTE["cascade"],
        linewidth=2.2,
        marker="o",
        markersize=6.5,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=4,
        label=labels.series_c,
    )
    # Threshold labels alternate above/below the curve so the tightly clustered
    # low-threshold points stay readable; the operating point gets extra
    # clearance because it carries a ring.
    for index, (summary, x, y) in enumerate(zip(cascades, xs, ys)):
        is_operating = abs(float(summary.threshold or 0.0) - operating) < 1e-9
        offset = 27 if is_operating else (14 if index % 2 == 0 else -23)
        ax.annotate(
            labels.thr_fmt.format(float(summary.threshold or 0.0)),
            (x, y),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=8,
            color=_PALETTE["cascade"],
            zorder=6,
        )

    endpoints = (
        # (summary, colour, marker, legend label, label offset in points, vertical align)
        (comparison.cheap, _PALETTE["cheap"], "s", labels.series_a, (11, -13), "top"),
        (comparison.premium, _PALETTE["premium"], "^", labels.series_b, (13, 0), "center"),
    )
    for summary, color, marker, series_label, (dx, dy), v_align in endpoints:
        x = summary.total_tokens / 1000.0
        y = summary.accuracy * 100.0
        ax.scatter(
            [x],
            [y],
            s=175,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=6,
            label=series_label,
        )
        # Captions are placed to dodge the two crowded regions: strategy A is the
        # lowest point, so it captions *below* itself (its left is the axes edge);
        # strategy B sits just under the legend, so it captions to the right.
        ax.annotate(
            f"{series_label}\n{summary.accuracy * 100:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left" if dx != 0 else "center",
            va=v_align,
            fontsize=8.5,
            color=color,
            zorder=6,
        )

    # Ring the operating point so the reader can find the configuration the rest
    # of the report quotes.
    operating_point = next(
        (s for s in cascades if abs(float(s.threshold or 0.0) - operating) < 1e-9), None
    )
    if operating_point is not None:
        ax.scatter(
            [operating_point.total_tokens / 1000.0],
            [operating_point.accuracy * 100.0],
            s=430,
            facecolors="none",
            edgecolors=_PALETTE["operating"],
            linewidth=1.9,
            zorder=7,
            label=labels.series_operating,
        )

    # Headline trade-off, computed from the analysis rather than typed in.
    # ``loss`` is premium-minus-cascade, so the *signed* change from premium to
    # cascade accuracy is its negation: a positive loss must print as negative.
    savings = comparison.cascade_savings_vs_premium
    loss = comparison.cascade_accuracy_loss_vs_premium
    box = (
        f"{labels.box_cascade_vs_premium}\n"
        f"cost {_signed_pct(-savings)}   accuracy {-loss * 100:+.1f}pt"
    )
    ax.text(
        0.985,
        0.04,
        box,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f4f6f8", "edgecolor": "#c7d0d9"},
        zorder=8,
    )

    all_y = ys + [comparison.cheap.accuracy * 100.0, comparison.premium.accuracy * 100.0]
    low, high = min(all_y), max(all_y)
    pad = max(1.5, (high - low) * 0.30)
    ax.set_ylim(low - pad, high + pad)
    all_x = xs + [comparison.cheap.total_tokens / 1000.0, comparison.premium.total_tokens / 1000.0]
    ax.set_xlim(min(all_x) * 0.955, max(all_x) * 1.045)

    ax.set_title(labels.title_accuracy)
    ax.set_xlabel(labels.xlabel_tokens)
    ax.set_ylabel(labels.ylabel_accuracy)
    ax.legend(loc="upper left", framealpha=0.92, frameon=True, edgecolor="#d6dde4")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Figure 2 — threshold sensitivity
# --------------------------------------------------------------------------- #
def plot_threshold_sensitivity(analysis: "AuditAnalysis", path: str | Path) -> Path:
    """Dual-axis view of the threshold sweep.

    Left axis: accuracy and wasted-token share (both percentages, so they share
    a 0-100 scale honestly).  Right axis: total cost in USD.  The story the
    figure has to tell is that accuracy is nearly flat while cost climbs.
    """
    labels = _labels()
    _apply_style()

    sensitivity = analysis.sensitivity
    points = sorted(sensitivity.points, key=lambda p: p.threshold)
    if not points:
        raise ValueError("analysis contains no threshold sweep to plot")

    thresholds = [p.threshold for p in points]
    accuracy = [p.accuracy * 100.0 for p in points]
    waste = [p.wasted_token_share * 100.0 for p in points]
    cost = [p.total_cost_usd for p in points]

    fig, ax = plt.subplots(figsize=(11.2, 6.7))
    ax2 = ax.twinx()
    ax2.patch.set_visible(False)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)

    line_cost = ax2.plot(
        thresholds,
        cost,
        color=_PALETTE["cascade"],
        linewidth=2.2,
        marker="D",
        markersize=5.5,
        markeredgecolor="white",
        markeredgewidth=0.9,
        zorder=5,
        label=labels.series_cost,
    )[0]
    line_acc = ax.plot(
        thresholds,
        accuracy,
        color=_PALETTE["cheap"],
        linewidth=2.0,
        marker="o",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.9,
        zorder=6,
        label=labels.series_acc,
    )[0]
    line_waste = ax.plot(
        thresholds,
        waste,
        color=_PALETTE["wasted_escalation"],
        linewidth=2.0,
        marker="s",
        markersize=5.5,
        markeredgecolor="white",
        markeredgewidth=0.9,
        zorder=6,
        label=labels.series_waste,
    )[0]

    ref_cheap = ax.axhline(
        sensitivity.cheap_accuracy * 100.0,
        color=_PALETTE["cheap"],
        linestyle=":",
        linewidth=1.4,
        alpha=0.85,
        zorder=2,
    )
    ref_premium = ax.axhline(
        sensitivity.premium_accuracy * 100.0,
        color=_PALETTE["premium"],
        linestyle=":",
        linewidth=1.4,
        alpha=0.85,
        zorder=2,
    )
    ref_operating = ax.axvline(
        sensitivity.operating_threshold,
        color=_PALETTE["operating"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        zorder=3,
    )

    # Left axis carries two different percentage series (accuracy and waste), so
    # a fixed 0-100 range wastes half the panel and makes the accuracy line look
    # artificially flat.  Fit the range to the data with headroom instead, while
    # keeping 0 in view so "share of total" stays interpretable.
    pct_values = (
        [p.accuracy * 100.0 for p in points]
        + [p.wasted_token_share * 100.0 for p in points]
        + [sensitivity.cheap_accuracy * 100.0, sensitivity.premium_accuracy * 100.0]
    )
    pct_high = max(pct_values) if pct_values else 100.0
    pct_top = max(10.0, pct_high * 1.25)
    cost_top = max(cost) * 1.20 if cost else 1.0
    ax.set_ylim(0, pct_top)
    ax.set_xlim(min(thresholds) - 0.05, max(thresholds) + 0.05)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{t:.1f}" for t in thresholds])
    ax2.set_ylim(0, cost_top)

    # Value labels: cost on every point (it is the exploding series), accuracy
    # on every point (it is the flat series the reader needs exact numbers for),
    # waste on every other point to keep the panel from getting crowded.
    #
    # Every label gets an opaque white pad.  The three series genuinely cross
    # (waste climbs into the accuracy band at high thresholds, and the cost line
    # crosses the accuracy line near the top right), so offsets alone cannot
    # guarantee separation — a white pad guarantees the text stays legible even
    # where it sits on top of a line or another label.
    def _label_bbox(color: str) -> dict[str, Any]:
        return {
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 0.5,
            "alpha": 0.88,
        }

    for index, point in enumerate(points):
        # Cost lives on the right axis; compare *normalised* heights so the
        # label can dodge the accuracy curve where the two lines cross.
        cost_norm = point.total_cost_usd / cost_top if cost_top else 0.0
        acc_norm = (point.accuracy * 100.0) / pct_top if pct_top else 0.0
        cost_above = cost_norm >= acc_norm
        ax2.annotate(
            _money(point.total_cost_usd),
            (point.threshold, point.total_cost_usd),
            textcoords="offset points",
            xytext=(0, -12 if cost_above else 11),
            ha="center",
            va="top" if cost_above else "bottom",
            fontsize=7.2,
            color=_PALETTE["cascade"],
            bbox=_label_bbox(_PALETTE["cascade"]),
            zorder=7,
        )
        ax.annotate(
            f"{point.accuracy * 100:.1f}",
            (point.threshold, point.accuracy * 100.0),
            textcoords="offset points",
            xytext=(0, -12 if cost_above else 9),
            ha="center",
            va="top" if cost_above else "bottom",
            fontsize=7.2,
            color=_PALETTE["cheap"],
            bbox=_label_bbox(_PALETTE["cheap"]),
            zorder=7,
        )
        if index % 2 == 0:
            ax.annotate(
                f"{point.wasted_token_share * 100:.1f}",
                (point.threshold, point.wasted_token_share * 100.0),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                va="top",
                fontsize=7.2,
                color=_PALETTE["wasted_escalation"],
                bbox=_label_bbox(_PALETTE["wasted_escalation"]),
                zorder=7,
            )

    # Left axis carries two different percentage series (accuracy and waste), so
    # a fixed 0-100 range wastes half the panel and makes the accuracy line look
    # artificially flat.  Fit the range to the data with headroom instead, while
    # keeping 0 in view so "share of total" stays interpretable.
    pct_values = (
        [p.accuracy * 100.0 for p in points]
        + [p.wasted_token_share * 100.0 for p in points]
        + [sensitivity.cheap_accuracy * 100.0, sensitivity.premium_accuracy * 100.0]
    )
    pct_high = max(pct_values) if pct_values else 100.0
    ax.set_ylim(0, max(10.0, pct_high * 1.25))
    # Extra right-hand padding: the last threshold's value labels are centred on
    # the tick, so they need room to extend past it without hitting the spine.
    ax.set_xlim(min(thresholds) - 0.05, max(thresholds) + 0.11)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{t:.1f}" for t in thresholds])
    ax2.set_ylim(0, max(cost) * 1.20)

    ax.set_title(labels.title_sensitivity)
    ax.set_xlabel(labels.xlabel_threshold)
    ax.set_ylabel(labels.ylabel_share)
    ax2.set_ylabel(labels.ylabel_cost)

    # The "paying more for noise" story: at the operating threshold, how much of
    # the spend buys nothing?  Printed from the analysis, never typed in.
    operating_points = [
        p for p in points if abs(p.threshold - sensitivity.operating_threshold) < 1e-9
    ]
    if operating_points:
        operating_point = operating_points[0]
        box = (
            f"{labels.ref_operating} θ={operating_point.threshold:.1f}\n"
            f"{labels.box_precision} {operating_point.escalation_precision * 100:.1f}%\n"
            f"{labels.series_waste} {operating_point.wasted_token_share * 100:.1f}%"
        )
        ax.text(
            0.015,
            0.965,
            box,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.4,
            linespacing=1.6,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#f4f6f8",
                "edgecolor": "#c7d0d9",
                "alpha": 0.95,
            },
            zorder=9,
        )

    handles = [
        line_cost,
        line_acc,
        line_waste,
        Line2D([], [], color=_PALETTE["cheap"], linestyle=":", linewidth=1.4),
        Line2D([], [], color=_PALETTE["premium"], linestyle=":", linewidth=1.4),
        ref_operating,
    ]
    names = [
        labels.series_cost,
        labels.series_acc,
        labels.series_waste,
        labels.ref_cheap_acc,
        labels.ref_premium_acc,
        labels.ref_operating,
    ]
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0))
    fig.legend(
        handles,
        names,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=2.2,
        columnspacing=2.0,
    )
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Figure 3 — token breakdown per configuration
# --------------------------------------------------------------------------- #
def plot_cost_breakdown(analysis: "AuditAnalysis", path: str | Path) -> Path:
    """Stacked bars: useful tokens vs the two routing-error token buckets.

    Every configuration is shown, not just the cascade, so the reader can see
    that A and B spend no *routing-error* tokens at all while C trades some of
    its savings for them.
    """
    labels = _labels()
    _apply_style()

    summaries = analysis.comparison.ordered()
    if not summaries:
        raise ValueError("analysis contains no configurations to plot")

    ticks = [
        f"{_config_tick(summary)}\n{summary.accuracy * 100:.1f}%"
        for summary in summaries
    ]
    useful = [
        max(0, summary.total_tokens - summary.wasted_escalation_tokens - summary.wasted_downgrade_tokens)
        / 1000.0
        for summary in summaries
    ]
    wasted_escalation = [summary.wasted_escalation_tokens / 1000.0 for summary in summaries]
    wasted_downgrade = [summary.wasted_downgrade_tokens / 1000.0 for summary in summaries]
    totals = [u + e + d for u, e, d in zip(useful, wasted_escalation, wasted_downgrade)]
    positions = list(range(len(summaries)))

    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    width = 0.62
    ax.bar(
        positions,
        useful,
        width=width,
        color=_PALETTE["useful"],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label=labels.series_useful,
    )
    ax.bar(
        positions,
        wasted_escalation,
        bottom=useful,
        width=width,
        color=_PALETTE["wasted_escalation"],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label=labels.series_wasted_escalation,
    )
    bottoms = [u + e for u, e in zip(useful, wasted_escalation)]
    ax.bar(
        positions,
        wasted_downgrade,
        bottom=bottoms,
        width=width,
        color=_PALETTE["wasted_downgrade"],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label=labels.series_wasted_downgrade,
    )

    ceiling = max(totals) if totals else 1.0
    for x, base, escalation, downgrade, total in zip(
        positions, useful, wasted_escalation, wasted_downgrade, totals
    ):
        ax.text(
            x,
            total,
            f"{total:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#1f2933",
            zorder=5,
        )
        if escalation > ceiling * 0.04:
            ax.text(
                x,
                base + escalation / 2.0,
                f"{escalation:,.0f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white",
                zorder=5,
            )
        if downgrade > ceiling * 0.04:
            ax.text(
                x,
                base + escalation + downgrade / 2.0,
                f"{downgrade:,.0f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white",
                zorder=5,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(ticks, fontsize=8.6)
    ax.set_ylim(0, ceiling * 1.16)
    ax.set_xlim(-0.7, len(summaries) - 0.3)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.set_title(labels.title_breakdown)
    ax.set_xlabel(labels.xlabel_config)
    ax.set_ylabel(labels.ylabel_tokens)
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0))
    fig.legend(
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=1.8,
        columnspacing=2.0,
    )
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
def render_all(analysis: "AuditAnalysis", out_dir: str | Path) -> dict[str, Path]:
    """Render every figure into ``out_dir`` and return ``{name: path}``."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        FIGURE_ACCURACY_VS_TOKENS: plot_accuracy_vs_tokens(
            analysis, root / f"{FIGURE_ACCURACY_VS_TOKENS}.png"
        ),
        FIGURE_THRESHOLD_SENSITIVITY: plot_threshold_sensitivity(
            analysis, root / f"{FIGURE_THRESHOLD_SENSITIVITY}.png"
        ),
        FIGURE_COST_BREAKDOWN: plot_cost_breakdown(
            analysis, root / f"{FIGURE_COST_BREAKDOWN}.png"
        ),
    }
