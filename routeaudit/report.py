"""Markdown report rendering for the route-audit sandbox.

The report layer is a *renderer only*: it never computes a metric.  Every
number it prints is read off the :class:`~routeaudit.analysis.AuditAnalysis`
handed to it, which in turn comes from :mod:`routeaudit.accounting` and
:mod:`routeaudit.analysis`.  That separation is deliberate — a reviewer can
recompute every figure in ``report.md`` from ``analysis.json`` alone, and this
module stays a pure function of its input.

Public API
----------
``render_markdown(analysis, *, figures=None, title=None) -> str``
    Render the Chinese technical report as a string.  ``figures`` maps figure
    names (see :data:`routeaudit.charts.FIGURE_NAMES`) to the relative paths the
    Markdown should embed; omit it to render a text-only report.
``write_report(analysis, out_dir, *, figures=True, title=None) -> dict``
    Write ``report.md``, ``analysis.json`` and (optionally) the PNGs, returning
    ``{"markdown": path, "figures": {name: path}, "json": path}``.  ``figures``
    may also be a ``{name: path}`` mapping when the caller has already rendered
    the PNGs (the CLI does this), in which case those paths are embedded as-is.
``render_json(analysis) -> dict``
    The machine-readable companion, i.e. ``AuditAnalysis.to_dict()``.

Importing this module has no side effects: matplotlib is imported lazily inside
:func:`write_report`, so ``render_markdown`` works in an environment where the
plotting stack is unavailable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .accounting import ConfigSummary
    from .analysis import AuditAnalysis, ErrorBucket, RoutingErrorAnalysis

__all__ = ["render_markdown", "write_report", "render_json"]


# --------------------------------------------------------------------------- #
# Static vocabulary (labels only — never results)
# --------------------------------------------------------------------------- #
_DEFAULT_TITLE = "模型路由成本审计报告"

_REASON_ZH: dict[str, str] = {
    "none": "未升级",
    "low_confidence": "置信度低于阈值",
    "format_failure": "格式检查失败",
    "missing_confidence": "置信度缺失",
    "provider_error": "廉价调用失败",
}

_EM_DASH = "—"


# --------------------------------------------------------------------------- #
# Formatting primitives
# --------------------------------------------------------------------------- #
def _pct(value: float | None, digits: int = 1) -> str:
    """Format a 0-1 fraction as a percentage string."""
    if value is None or not math.isfinite(value):
        return _EM_DASH
    return f"{value * 100:.{digits}f}%"


def _signed_pct(value: float | None, digits: int = 1) -> str:
    """Format a 0-1 fraction as a *signed* percentage (for deltas)."""
    if value is None or not math.isfinite(value):
        return _EM_DASH
    return f"{value * 100:+.{digits}f}%"


def _points(value: float | None, digits: int = 1) -> str:
    """Format a 0-1 fraction as a signed percentage-point delta."""
    if value is None or not math.isfinite(value):
        return _EM_DASH
    return f"{value * 100:+.{digits}f}pt"


def _usd(value: float | None, digits: int = 4) -> str:
    """Format a USD amount; ``inf``/``nan`` degrade to an em dash."""
    if value is None or not math.isfinite(value):
        return _EM_DASH
    return f"${value:,.{digits}f}"


def _tokens(value: int | float | None) -> str:
    """Format a token count with thousands separators."""
    if value is None:
        return _EM_DASH
    return f"{int(round(float(value))):,}"


def _ratio(value: float | None, digits: int = 2) -> str:
    """Format a dimensionless ratio; ``inf`` is shown explicitly, not hidden."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return _EM_DASH
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    return f"{value:.{digits}f}×"


def _plain(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return _EM_DASH
    return f"{value:,.{digits}f}"


def _share_of_waste(errors: "RoutingErrorAnalysis", part_tokens: int) -> float:
    """``part_tokens`` as a share of the *wasted* tokens, not of all tokens."""
    total = errors.wasted_escalation_tokens + errors.wasted_downgrade_tokens
    return part_tokens / total if total else 0.0


def _config_name(summary: "ConfigSummary") -> str:
    """Display name for one configuration row."""
    if summary.threshold is None:
        return summary.label
    return f"C · cascade @ {summary.threshold:.1f}"


def _table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    align: Sequence[str] | None = None,
) -> str:
    """Render a GitHub-flavoured Markdown table."""
    headers = list(headers)
    if align is None:
        align = ["---"] * len(headers)
    marks = {
        "l": ":---",
        "r": "---:",
        "c": ":---:",
    }
    separator = [marks.get(a, a) for a in align]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def _figure(figures: Mapping[str, str] | None, name: str, caption: str) -> str:
    """Embed one figure, or return an empty string when it is not available."""
    if not figures:
        return ""
    path = figures.get(name)
    if not path:
        return ""
    return f"![{caption}]({path})\n\n*图：{caption}*"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _header(analysis: "AuditAnalysis", title: str) -> list[str]:
    audit = analysis.audit
    return [
        f"# {title}",
        "",
        f"> run_id `{audit.run_id}` · provider `{audit.provider}` · 任务集 `{audit.task_set}` "
        f"· {audit.n_tasks} 个任务 · seed `{audit.seed}`",
        f"> 生成时间 {audit.created_at or _EM_DASH}",
        "",
        "本报告只呈现数字，不制造数字：表中每一个数值都来自本次审计的分析对象，"
        "并可通过同目录下的 `analysis.json` 独立复算。",
        "",
    ]


def _summary_section(analysis: "AuditAnalysis") -> list[str]:
    """TL;DR: the hidden token price of over-downgrade."""
    comparison = analysis.comparison
    errors = analysis.routing_errors
    regret = analysis.regret
    cascade = comparison.cascade

    downgrade = errors.over_downgrade
    escalation = errors.over_escalation
    missed = errors.missed_escalation
    price_ratio = errors.escalation_cost_ratio

    lines = ["## 1. 摘要（TL;DR）", ""]
    lines.append(
        f"级联路由（阈值 θ={errors.threshold:.1f}）在 {cascade.n_tasks} 个任务上拿到 "
        f"{_pct(cascade.accuracy)} 的准确率，花掉 {_tokens(cascade.total_tokens)} token / "
        f"{_usd(cascade.total_cost_usd)}，相对全高档模型（{_pct(comparison.premium.accuracy)} 准确率、"
        f"{_usd(comparison.premium.total_cost_usd)}）省下 **{_pct(comparison.cascade_savings_vs_premium)}** "
        f"的费用，代价是准确率 {_points(-comparison.cascade_accuracy_loss_vs_premium)}。"
    )
    lines.append("")
    lines.append("**核心结论：省下来的钱有一部分是假省，代价藏在“过度降级”的重复 token 里。**")
    lines.append("")
    lines.append(
        f"- **过度降级（over-downgrade）**：{downgrade.n_tasks} 个任务上廉价模型先答错、再触发升级，"
        f"这 {_tokens(downgrade.tokens)} token 的廉价尝试完全没买到正确性，占总 token 的 "
        f"**{_pct(downgrade.token_share)}**，占总成本的 **{_pct(downgrade.cost_share)}**。"
        "这是“先降级再升级”的结构性开销，而不是偶发事故。"
    )
    lines.append(
        f"- **过度升级（over-escalation）**：{escalation.n_tasks} 个任务上廉价模型本来已经答对，"
        f"仍付了高档模型的调用费，浪费 {_tokens(escalation.tokens)} token"
        f"（占总 token {_pct(escalation.token_share)}，占总成本 {_pct(escalation.cost_share)}）。"
    )
    lines.append(
        f"- **漏升级（missed escalation）**：{missed.n_tasks} 个任务廉价模型答错却未升级，"
        "直接交付错误答案（占任务 "
        f"{_pct(errors.missed_escalation_rate)}）。"
        "这一类不额外花 token，它花的是**正确性**——正是“省下大半成本”这类说法最容易被忽略的部分。"
    )
    lines.append(
        f"- 两类浪费合计吃掉级联总 token 的 **{_pct(errors.wasted_token_share)}**、总成本的 "
        f"**{_pct(errors.wasted_cost_share)}**。"
        f"在当前的价目表下，一次高档调用约等于 **{_ratio(price_ratio)}** 次廉价调用，"
        "因此每一次不必要的升级都在放大浪费。"
    )
    lines.append("")
    lines.append("三个可直接引用的数字：")
    lines.append("")
    lines.append(
        _table(
            ["指标", "数值", "含义"],
            [
                [
                    "级联成本节省（vs 全高档）",
                    _pct(comparison.cascade_savings_vs_premium),
                    f"成本从 {_usd(comparison.premium.total_cost_usd)} 降到 {_usd(cascade.total_cost_usd)}",
                ],
                [
                    "过度降级浪费 token 占比",
                    _pct(downgrade.token_share),
                    f"{_tokens(downgrade.tokens)} token 买了 0 个正确答案",
                ],
                [
                    "Regret（vs 精度匹配 oracle）",
                    _usd(regret.regret_cost_usd),
                    f"占级联成本 {_pct(regret.regret_cost_share)}，路由效率 {_pct(regret.routing_efficiency)}",
                ],
            ],
            align=["l", "r", "l"],
        )
    )
    lines.append("")
    return lines


def _method_section(analysis: "AuditAnalysis") -> list[str]:
    audit = analysis.audit
    config = dict(audit.config)
    stats = dict(audit.stats)
    cheap = dict(stats.get("cheap_model") or {})
    premium = dict(stats.get("premium_model") or {})
    summary = dict(stats.get("task_set_summary") or {})
    thresholds = ", ".join(f"{float(t):.1f}" for t in audit.thresholds)
    domains = dict(summary.get("domains") or {})

    lines = ["## 2. 方法与可复现性", ""]
    lines.append(
        "审计采用**反事实结果矩阵（counterfactual outcome matrix）**：策略 A 让廉价模型跑完全部任务并逐题打分，"
        "策略 B 让高档模型跑完全部任务并逐题打分。因为离线审计手里有金标准标签，这两列可以在不给路由器的前提下"
        "补全每一题的 `(cheap_verdict, premium_verdict)` 组合；之后每个级联阈值只是对同一矩阵的**纯重放**——"
        "廉价尝试不重付，只为该阈值真正触发的升级付高档调用。"
    )
    lines.append("")
    lines.append(
        "这正是离线审计能做到而线上路由器做不到的事：线上路由器无法为“它没有调用的模型”打分，"
        "因此无法给路由错误定价；离线审计可以为未发生的调用定价，代价为零。"
    )
    lines.append("")
    lines.append(
        _table(
            ["项目", "取值"],
            [
                ["run_id", f"`{audit.run_id}`"],
                ["provider", f"`{audit.provider}`（离线校准模拟器，非真实 API 账单）"],
                ["任务集", f"`{audit.task_set}`，{audit.n_tasks} 个任务"],
                ["seed", f"`{audit.seed}`（任务生成与模拟抽样均由此派生，可复现）"],
                ["廉价模型", f"`{cheap.get('name', _EM_DASH)}` — 输入 ${cheap.get('input_price_per_mtok', 0)}/Mtok，输出 ${cheap.get('output_price_per_mtok', 0)}/Mtok"],
                ["高档模型", f"`{premium.get('name', _EM_DASH)}` — 输入 ${premium.get('input_price_per_mtok', 0)}/Mtok，输出 ${premium.get('output_price_per_mtok', 0)}/Mtok"],
                ["价格缩放系数", f"`{config.get('price_scale', _EM_DASH)}`"],
                ["阈值扫描", f"θ ∈ {{{thresholds}}}"],
                ["格式门控", "开启（结构检查失败即升级）" if stats.get("format_gate") else "关闭"],
                ["调用缓存", "开启（同一 task/model/stage 只付一次）" if stats.get("cache_enabled") else "关闭"],
                ["实际执行调用数", _tokens(stats.get("executed_calls"))],
                ["实际执行成本", _usd(stats.get("executed_cost_usd"))],
                ["领域构成", ", ".join(f"{k}: {v}" for k, v in sorted(domains.items())) or _EM_DASH],
                ["平均难度", _plain(summary.get("mean_difficulty"), 3)],
            ],
            align=["l", "l"],
        )
    )
    lines.append("")
    lines.append(
        "注意区分两个成本口径：上表的“实际执行成本”是整个扫描真正花掉的钱（廉价阶段被多个阈值共享）；"
        "而第 3 节起引用的每个配置的成本是**账单口径**——假设单独部署该配置会付出的完整费用。"
        "两者不可混用。"
    )
    lines.append("")
    return lines


def _comparison_section(
    analysis: "AuditAnalysis", figures: Mapping[str, str] | None
) -> list[str]:
    comparison = analysis.comparison
    rows: list[list[str]] = []
    for summary in comparison.ordered():
        rows.append(
            [
                _config_name(summary),
                _pct(summary.accuracy),
                _tokens(summary.total_tokens),
                _usd(summary.total_cost_usd),
                _plain(summary.tokens_per_correct, 1),
                _usd(summary.cost_per_correct, 6),
                _pct(summary.escalation_rate),
                _pct(summary.mean_confidence),
            ]
        )

    lines = ["## 3. 三策略对比", ""]
    lines.append(
        _table(
            [
                "配置",
                "准确率",
                "总 token",
                "总成本 (USD)",
                "每正确 token",
                "每正确成本 (USD)",
                "升级率",
                "平均置信度",
            ],
            rows,
            align=["l", "r", "r", "r", "r", "r", "r", "r"],
        )
    )
    lines.append("")
    lines.append(
        "**怎么读这张表。** `每正确 token` 与 `每正确成本` 是唯一能同时惩罚“答错”和“多花钱”的单一指标："
        "把总开销除以答对的题数。全高档模型准确率最高但每正确成本也最高；全廉价模型每正确成本最低但准确率垫底；"
        "级联的定位就在这两者之间的那条曲线上。"
    )
    lines.append("")
    lines.append(
        _table(
            ["相对关系", "数值"],
            [
                ["级联 vs 全高档：成本节省", _pct(comparison.cascade_savings_vs_premium)],
                # Cascade token usage can EXCEED premium (the cheap stage is paid
                # on every task, including ones later escalated).  Report the
                # signed change rather than a "savings" figure that goes negative.
                ["级联 vs 全高档：token 变化", _signed_pct(-comparison.cascade_token_savings_vs_premium)],
                ["级联 vs 全高档：准确率差", _points(-comparison.cascade_accuracy_loss_vs_premium)],
                ["级联 vs 全廉价：成本倍数", _ratio(comparison.cascade_cost_multiple_vs_cheap)],
                ["级联 vs 全廉价：准确率增益", _points(comparison.cascade_accuracy_gain_vs_cheap)],
            ],
            align=["l", "r"],
        )
    )
    lines.append("")
    figure = _figure(figures, "accuracy_vs_tokens", "准确率 vs 总 token 成本：三种策略的位置关系")
    if figure:
        lines.extend([figure, ""])
    return lines


def _routing_error_section(
    analysis: "AuditAnalysis", figures: Mapping[str, str] | None
) -> list[str]:
    errors = analysis.routing_errors

    def bucket_row(bucket: "ErrorBucket", label: str, note: str) -> list[str]:
        return [
            label,
            str(bucket.n_tasks),
            _pct(bucket.task_share),
            _tokens(bucket.tokens),
            _pct(bucket.token_share),
            _usd(bucket.cost_usd),
            _pct(bucket.cost_share),
            note,
        ]

    rows = [
        bucket_row(errors.over_escalation, "过度升级", "廉价已答对却仍升级：纯浪费"),
        bucket_row(errors.over_downgrade, "过度降级", "廉价答错被迫升级：双倍开销"),
        bucket_row(errors.missed_escalation, "漏升级", "廉价答错却不升级：交付错误"),
    ]

    total_wasted_tokens = errors.wasted_escalation_tokens + errors.wasted_downgrade_tokens
    total_wasted_cost = errors.wasted_escalation_cost_usd + errors.wasted_downgrade_cost_usd

    lines = ["## 4. 路由错误代价归因", ""]
    lines.append(
        f"级联在阈值 θ={errors.threshold:.1f} 下的 {_tokens(errors.total_tokens)} token / "
        f"{_usd(errors.total_cost_usd)} 中，有 {errors.n_escalated} 次升级。"
        f"升级精确率为 **{_pct(errors.escalation_precision)}**，"
        f"即有 **{_pct(1.0 - errors.escalation_precision)}** 的升级是不必要的。"
    )
    lines.append("")
    lines.append(
        _table(
            [
                "路由错误",
                "任务数",
                "占任务比例",
                "token",
                "token 占总额",
                "成本 (USD)",
                "成本占总额",
                "说明",
            ],
            rows,
            align=["l", "r", "r", "r", "r", "r", "r", "l"],
        )
    )
    lines.append("")
    lines.append(
        f"**合计浪费：{_tokens(total_wasted_tokens)} token（占总额 {_pct(errors.wasted_token_share)}）、"
        f"{_usd(total_wasted_cost)}（占总额 {_pct(errors.wasted_cost_share)}）。**"
    )
    lines.append("")
    lines.append(
        "两类浪费的经济性质完全不同，不能相加了事："
        "**过度升级**每次浪费的是高档模型的钱，单次贵、次数少（占任务 "
        f"{_pct(errors.over_escalation.task_share)}）；**过度降级**每次浪费的是廉价模型的钱，"
        f"单次便宜、但几乎每个真正困难的任务都会中招（占任务 {_pct(errors.over_downgrade.task_share)}）。"
        "前者是“多花了不该花的钱”，后者是“先花钱买了一个没用的答案再花第二笔钱”，"
        "它的隐性在于：这笔开销不体现在最终答案里，只体现在 token 账单里。"
    )
    lines.append("")
    lines.append(
        f"**漏升级的成本列显示为 {_usd(0.0)} 是设计使然，不是缺数据**：这一类错误不额外消耗 token，"
        f"它的代价是 {errors.missed_escalation.n_tasks} 个错误答案（占任务 "
        f"{_pct(errors.missed_escalation_rate)}）。任何只按 token 计价的“节省率”都会把这类错误算成收益。"
    )
    lines.append("")

    if errors.n_escalation_reasons:
        reason_rows = [
            [_REASON_ZH.get(key, key), str(count), _pct(count / errors.n_escalated if errors.n_escalated else 0.0)]
            for key, count in sorted(errors.n_escalation_reasons.items(), key=lambda kv: -kv[1])
        ]
        lines.append("升级原因分布：")
        lines.append("")
        lines.append(_table(["升级原因", "次数", "占升级比例"], reason_rows, align=["l", "r", "r"]))
        lines.append("")

    figure = _figure(figures, "cost_breakdown", "每个配置的 token 去向：有效开销与两类路由错误")
    if figure:
        lines.extend([figure, ""])
    return lines


def _sensitivity_section(
    analysis: "AuditAnalysis", figures: Mapping[str, str] | None
) -> list[str]:
    sensitivity = analysis.sensitivity
    rows: list[list[str]] = []
    for point in sorted(sensitivity.points, key=lambda p: p.threshold):
        marginal = (
            _usd(point.marginal_cost_per_point, 2)
            if point.marginal_cost_per_point is not None
            else _EM_DASH
        )
        marginal_tokens = (
            _plain(point.marginal_tokens_per_point, 0)
            if point.marginal_tokens_per_point is not None
            else _EM_DASH
        )
        marker = " ◀" if abs(point.threshold - sensitivity.operating_threshold) < 1e-9 else ""
        rows.append(
            [
                f"{point.threshold:.1f}{marker}",
                _pct(point.accuracy),
                _tokens(point.total_tokens),
                _usd(point.total_cost_usd),
                _plain(point.tokens_per_correct, 1),
                _usd(point.cost_per_correct_usd, 6),
                _pct(point.escalation_rate),
                _pct(point.escalation_precision),
                _pct(point.wasted_token_share),
                marginal,
                marginal_tokens,
            ]
        )

    best = sensitivity.best_accuracy_point
    cheapest = sensitivity.cheapest_point
    operating = sensitivity.point(sensitivity.operating_threshold)

    lines = ["## 5. 阈值敏感性", ""]
    lines.append(
        f"把升级阈值从 {min(p.threshold for p in sensitivity.points):.1f} 扫到 "
        f"{max(p.threshold for p in sensitivity.points):.1f}，观察成本与准确率如何移动。"
        f"参照线：全廉价 {_pct(sensitivity.cheap_accuracy)} / {_usd(sensitivity.cheap_cost_usd)}，"
        f"全高档 {_pct(sensitivity.premium_accuracy)} / {_usd(sensitivity.premium_cost_usd)}。"
        f"标 ◀ 的是运营阈值 θ={sensitivity.operating_threshold:.1f}。"
    )
    lines.append("")
    lines.append(
        _table(
            [
                "阈值",
                "准确率",
                "总 token",
                "总成本 (USD)",
                "每正确 token",
                "每正确成本 (USD)",
                "升级率",
                "升级精确率",
                "浪费 token 占比",
                "边际成本/准确率点",
                "边际 token/准确率点",
            ],
            rows,
            align=["l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
        )
    )
    lines.append("")
    lines.append(
        f"边际列的含义是：相对上一个（更低的）阈值，为了多买到 1 个百分点的准确率，需要额外付出多少成本与 token。"
        f"运营点 θ={operating.threshold:.1f} 的边际成本为 "
        f"{_usd(operating.marginal_cost_per_point, 2) if operating.marginal_cost_per_point is not None else _EM_DASH}"
        f" / 准确率点；最高准确率点出现在 θ={best.threshold:.1f}"
        f"（{_pct(best.accuracy)}，{_usd(best.total_cost_usd)}），"
        f"最低成本点出现在 θ={cheapest.threshold:.1f}（{_usd(cheapest.total_cost_usd)}）。"
    )
    lines.append("")
    lines.append(
        "**读法提示。** 若边际成本随阈值单调上升而准确率几乎不动，说明在这段区间里继续加钱买不到准确率——"
        "这正是“阈值调优”最容易踩的陷阱：把成本曲线当成了准确率曲线。"
        f"本次扫描中，准确率从 θ={cheapest.threshold:.1f} 的 {_pct(cheapest.accuracy)} 到 "
        f"θ={best.threshold:.1f} 的 {_pct(best.accuracy)}，只移动了 "
        f"{_points(best.accuracy - cheapest.accuracy)}，而成本从 {_usd(cheapest.total_cost_usd)} "
        f"变成 {_usd(best.total_cost_usd)}。"
    )
    lines.append("")

    figure = _figure(figures, "threshold_sensitivity", "阈值敏感性：成本、准确率与浪费占比随阈值的变化")
    if figure:
        lines.extend([figure, ""])
    return lines


def _regret_section(analysis: "AuditAnalysis") -> list[str]:
    regret = analysis.regret
    lines = ["## 6. Regret 分析：距离“精度匹配 oracle”还有多远", ""]
    lines.append(
        "把级联和“永远用高档模型”或“永远用廉价模型”比都没有意义：前者准确率高但贵，后者便宜但差。"
        "有意义的基准是**精度匹配 oracle**——一个知道金标准标签、可以逐题独立路由的决策者，"
        "在达到与级联**相同准确率**的前提下，最低需要花多少钱。"
    )
    lines.append("")
    lines.append(
        f"构造方式：从“所有任务都给廉价模型”出发，只对“升级确实能把错误答案修对”的任务按增量成本从小到大升级，"
        f"直到达到级联的 {regret.n_tasks} 个任务中的 {_tokens(round(regret.target_accuracy * regret.n_tasks))} 个正确为止。"
        "由于每次升级最多买回 1 个正确答案，这个按成本排序的贪心解对该整数问题恰好最优。"
    )
    lines.append("")
    lines.append(
        _table(
            ["指标", "级联（当前路由）", "精度匹配 oracle", "差额"],
            [
                [
                    "准确率",
                    _pct(regret.cascade_accuracy),
                    _pct(regret.oracle_accuracy),
                    _points(regret.cascade_accuracy - regret.oracle_accuracy),
                ],
                [
                    "总成本 (USD)",
                    _usd(regret.cascade_cost_usd),
                    _usd(regret.oracle_cost_usd),
                    _usd(regret.regret_cost_usd),
                ],
                [
                    "总 token",
                    _tokens(regret.cascade_tokens),
                    _tokens(regret.oracle_tokens),
                    _tokens(regret.regret_tokens),
                ],
                [
                    "升级次数",
                    str(regret.cascade_n_escalations),
                    str(regret.oracle_n_escalations),
                    f"{regret.cascade_n_escalations - regret.oracle_n_escalations:+d}",
                ],
            ],
            align=["l", "r", "r", "r"],
        )
    )
    lines.append("")
    lines.append(
        f"**Regret = {_usd(regret.regret_cost_usd)}，占级联实际成本的 {_pct(regret.regret_cost_share)}；"
        f"对应 {_tokens(regret.regret_tokens)} token（占 {_pct(regret.regret_token_share)}）。**"
        f"换句话说，在准确率完全不变的前提下，一个更好的升级信号最多能再省下这笔钱。"
        f"这是路由改进的**理论上限**，不是可实现目标——oracle 拥有线上不可能存在的标签信息。"
    )
    lines.append("")
    lines.append(
        _table(
            ["派生指标", "数值", "含义"],
            [
                [
                    "路由效率",
                    _pct(regret.routing_efficiency),
                    f"相对全高档（{_usd(regret.premium_cost_usd)}）本可省下的钱中，级联实际拿到了多少",
                ],
                [
                    "成本倍数 vs oracle",
                    _ratio(regret.cost_ratio_vs_oracle),
                    "级联成本 / oracle 成本；1.0 表示已达最优",
                ],
                [
                    "全廉价基线",
                    _usd(regret.cheap_cost_usd),
                    "所有任务都走廉价模型的成本，也是 oracle 的起点",
                ],
                [
                    "全高档基线",
                    _usd(regret.premium_cost_usd),
                    "所有任务都走高档模型的成本，也是“省了多少钱”的分母",
                ],
            ],
            align=["l", "r", "l"],
        )
    )
    lines.append("")
    return lines


def _domain_section(analysis: "AuditAnalysis") -> list[str]:
    lines = ["## 7. 分领域拆解", ""]
    if not analysis.domains:
        lines.extend(["本次审计未提供领域标签，无法拆解。", ""])
        return lines

    rows: list[list[str]] = []
    for domain in analysis.domains:
        rows.append(
            [
                domain.domain,
                str(domain.n_tasks),
                _pct(domain.cheap_accuracy),
                _pct(domain.premium_accuracy),
                _pct(domain.cascade_accuracy),
                _points(domain.cascade_accuracy - domain.cheap_accuracy),
                _pct(domain.cascade_escalation_rate),
                _tokens(domain.cascade_tokens),
                _usd(domain.cascade_cost_usd),
                _pct(domain.wasted_token_share),
            ]
        )

    lines.append(
        "按领域拆分后，路由失败的位置比总量数字更清楚：准确率差距最大、浪费占比最高的领域，"
        "就是升级信号最不可靠、最值得单独调阈值的地方。"
    )
    lines.append("")
    lines.append(
        _table(
            [
                "领域",
                "任务数",
                "全廉价准确率",
                "全高档准确率",
                "级联准确率",
                "级联 vs 廉价",
                "升级率",
                "级联 token",
                "级联成本 (USD)",
                "浪费 token 占比",
            ],
            rows,
            align=["l", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
        )
    )
    lines.append("")

    ranked = sorted(analysis.domains, key=lambda d: d.wasted_token_share, reverse=True)
    worst = ranked[0]
    biggest_gap = max(
        analysis.domains, key=lambda d: d.premium_accuracy - d.cheap_accuracy
    )
    lines.append(
        f"- 浪费 token 占比最高：**{worst.domain}**（{_pct(worst.wasted_token_share)}，"
        f"{worst.n_tasks} 个任务，级联准确率 {_pct(worst.cascade_accuracy)}）。"
    )
    lines.append(
        f"- 廉价与高档差距最大：**{biggest_gap.domain}**（廉价 {_pct(biggest_gap.cheap_accuracy)} → "
        f"高档 {_pct(biggest_gap.premium_accuracy)}，差 "
        f"{_points(biggest_gap.premium_accuracy - biggest_gap.cheap_accuracy)}）。"
        "这类领域是级联最该多升级的地方，也是最该检查置信度校准的地方。"
    )
    lines.append("")
    return lines


def _conclusion_section(analysis: "AuditAnalysis") -> list[str]:
    comparison = analysis.comparison
    errors = analysis.routing_errors
    regret = analysis.regret
    sensitivity = analysis.sensitivity
    operating = sensitivity.point(sensitivity.operating_threshold)
    best = sensitivity.best_accuracy_point

    lines = ["## 8. 结论与局限", "", "### 结论", ""]
    lines.append(
        f"1. **级联确实省钱，但不是免费的。** 相对全高档节省 {_pct(comparison.cascade_savings_vs_premium)}，"
        f"代价是 {_points(-comparison.cascade_accuracy_loss_vs_premium)} 的准确率，"
        f"以及 {_pct(errors.wasted_token_share)} 的 token 花在了路由错误上。"
    )
    lines.append(
        f"2. **过度降级是隐藏成本的主要来源。** 它贡献了 {_tokens(errors.wasted_downgrade_tokens)} token "
        f"（占总额 {_pct(errors.over_downgrade.token_share)}，占过度类浪费的 "
        f"{_pct(_share_of_waste(errors, errors.wasted_downgrade_tokens))}），"
        "且随难度上升而系统性地增加——廉价模型越弱，这笔钱越像固定税。"
    )
    lines.append(
        f"3. **继续提高阈值买不到多少准确率。** 从 θ={operating.threshold:.1f} 到最高准确率点 "
        f"θ={best.threshold:.1f}，准确率只增加 {_points(best.accuracy - operating.accuracy)}，"
        f"成本却从 {_usd(operating.total_cost_usd)} 涨到 {_usd(best.total_cost_usd)}。"
    )
    lines.append(
        f"4. **改进空间有明确上界。** Regret 为 {_usd(regret.regret_cost_usd)}"
        f"（占成本 {_pct(regret.regret_cost_share)}，路由效率 {_pct(regret.routing_efficiency)}）；"
        "超过这个数字的“优化”只可能来自换模型或换价目表，而不是更好的路由。"
    )
    lines.append("")
    lines.append("### 局限")
    lines.append("")
    lines.append(
        "- **provider 是校准过的离线模拟器，不是真实 API 账单。** 本仓库默认的 mock provider 依据 "
        "`ModelSpec.quality` / `overconfidence` 与任务难度抽样生成答案与置信度。因此绝对数值只说明"
        "“在给定校准下会发生什么”，不能当作对某个厂商的实测结论；能直接迁移的是**结构**（浪费的方向、"
        "阈值曲线的形状、regret 的量级），而不是小数点。"
    )
    lines.append(
        "- **结论依赖价目表。** 成本、成本占比与“一次升级值多少次廉价调用”全部由 "
        "`config.py` 的价格表与 `--price-scale` 决定。换一份合同价，百分比会变，"
        "但归因表的结构不变——这正是把价格做成数据而非常量的原因。"
    )
    lines.append(
        "- **结论依赖校准假设。** 置信度校准（`overconfidence`）直接决定升级门控的质量，"
        "进而决定过度升级与漏升级的比例。真实模型的校准通常比模拟器更差，"
        "所以本报告的升级精确率应当视为**上限估计**。"
    )
    lines.append(
        "- **准确率是单一指标。** 报告只按答对/答错计分，未区分“错得离谱”与“差一点”，"
        "也未计入延迟、限流与重试成本。真实部署里这些因素可能改变阈值选择。"
    )
    lines.append(
        "- **样本量与置信区间。** 本报告未给出置信区间；任务数较少时，"
        "单个阈值之间的准确率差异可能落在抽样噪声内。"
    )
    lines.append("")
    lines.append("### 复现方式")
    lines.append("")
    lines.append("```bash")
    lines.append("# 重跑本次审计并生成同样的报告与图（含阈值扫描）")
    lines.append("python -m routeaudit report --sweep --out-dir out/")
    lines.append("")
    lines.append("# 仅用已保存的审计结果重新渲染（不再花任何调用）")
    lines.append("python -m routeaudit report --from-audit runs/audit.json --out-dir out/")
    lines.append("")
    lines.append("# 只重算分析、不渲染图")
    lines.append("python -m routeaudit analyse runs/audit.json --out runs/analysis.json")
    lines.append("```")
    lines.append("")
    lines.append(
        f"机器可读的同源数据见 `analysis.json`（run_id `{analysis.audit.run_id}`），"
        "其中包含本报告全部表格的原始字段。"
    )
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def render_markdown(
    analysis: "AuditAnalysis",
    *,
    figures: Mapping[str, str] | None = None,
    title: str | None = None,
) -> str:
    """Render the full Chinese Markdown report.

    Parameters
    ----------
    analysis:
        The analysed audit.  All numbers are read from it; nothing is computed
        here beyond display formatting and simple ratios between its fields.
    figures:
        Optional ``{figure_name: relative_path}`` map.  Recognised keys are
        ``accuracy_vs_tokens``, ``threshold_sensitivity`` and ``cost_breakdown``;
        unknown keys are ignored and missing keys simply omit that figure.
    title:
        Report title; defaults to a fixed Chinese heading.

    Returns
    -------
    str
        The complete Markdown document, newline-terminated.
    """
    lines: list[str] = []
    lines += _header(analysis, title or _DEFAULT_TITLE)
    lines += _summary_section(analysis)
    lines += _method_section(analysis)
    lines += _comparison_section(analysis, figures)
    lines += _routing_error_section(analysis, figures)
    lines += _sensitivity_section(analysis, figures)
    lines += _regret_section(analysis)
    lines += _domain_section(analysis)
    lines += _conclusion_section(analysis)
    lines.append("---")
    lines.append("")
    lines.append(
        "本报告由 `routeaudit.report` 生成，仅做渲染，不做计算。"
        "所有指标均由 `routeaudit.accounting` 与 `routeaudit.analysis` 产出。"
    )
    lines.append("")
    return "\n".join(lines)


def render_json(analysis: "AuditAnalysis") -> dict[str, Any]:
    """Return the machine-readable companion to the Markdown report."""
    return analysis.to_dict()


def write_report(
    analysis: "AuditAnalysis",
    out_dir: str | Path,
    *,
    figures: bool | Mapping[str, str] = True,
    title: str | None = None,
) -> dict[str, Any]:
    """Write the report bundle into ``out_dir``.

    Files written: ``report.md``, ``analysis.json`` and, when figures are
    requested, three PNGs.

    ``figures`` accepts three shapes, because the CLI and the tests use
    different ones:

    * ``True`` (default) — render the PNGs into ``<out_dir>/figures/``.
    * ``False`` — text-only report, no PNGs.
    * a ``{figure_name: path}`` mapping — the caller already rendered the PNGs
      (this is what ``cli.cmd_report`` does) and the report embeds *those* paths
      instead of re-rendering.  Paths are embedded as given, so the caller is
      responsible for them being reachable from ``report.md``.

    Returns
    -------
    dict
        ``{"markdown": <path>, "figures": {name: <path>}, "json": <path>}``.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, str] = {}
    figure_refs: dict[str, str] | None = None

    if isinstance(figures, Mapping):
        # Caller-supplied (already rendered).  Embed verbatim.
        figure_refs = {str(name): str(path) for name, path in figures.items()}
        figure_paths = dict(figure_refs)
    elif figures:
        # Imported lazily so ``render_markdown`` works without matplotlib.
        from .charts import render_all

        rendered = render_all(analysis, root / "figures")
        figure_paths = {name: str(path) for name, path in rendered.items()}
        figure_refs = {
            name: Path(path).relative_to(root).as_posix()
            for name, path in rendered.items()
        }

    markdown_path = root / "report.md"
    markdown_path.write_text(
        render_markdown(analysis, figures=figure_refs, title=title),
        encoding="utf-8",
    )

    json_path = root / "analysis.json"
    json_path.write_text(
        json.dumps(render_json(analysis), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "markdown": str(markdown_path),
        "figures": figure_paths,
        "json": str(json_path),
    }
