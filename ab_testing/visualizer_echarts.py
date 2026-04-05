"""
ECharts option dict generators for use with streamlit-echarts st_echarts().
All functions return standard ECharts option dicts (Python dicts); no pyecharts dependency.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import gaussian_kde
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .metrics import BayesianResult, FrequentistResult
    from .sequential import SequentialResult

# ── Colors ────────────────────────────────────────────────────────
C_A      = "#e74c3c"   # red:    Group A
C_B      = "#27ae60"   # green:  Group B
C_DELTA  = "#8e44ad"   # purple: delta
C_THRESH = "#e67e22"   # orange: threshold line


def _kde(data: np.ndarray, n: int = 300):
    """Return a KDE curve as [[x, y], ...] for ECharts value axis."""
    bw = gaussian_kde(data, bw_method="scott")
    margin = (data.max() - data.min()) * 0.12
    xs = np.linspace(data.min() - margin, data.max() + margin, n)
    ys = bw(xs)
    return [[round(float(x), 6), round(float(y), 6)] for x, y in zip(xs, ys)]


def _line_series(name: str, data: list, color: str, opacity: float = 0.25) -> dict:
    """Generic smooth area line series."""
    return {
        "name": name,
        "type": "line",
        "data": data,
        "smooth": True,
        "symbol": "none",
        "lineStyle": {"color": color, "width": 2},
        "areaStyle": {"color": color, "opacity": opacity},
        "emphasis": {"focus": "series"},
    }


# ── 1. Posterior distribution chart (A vs B) ─────────────────────
def posterior_chart(result: "BayesianResult", metric_label: str = "Metric") -> dict:
    """Dual posterior KDE curves showing the parameter distribution for A and B."""
    data_a = _kde(result.posterior_a)
    data_b = _kde(result.posterior_b)

    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": {"data": ["Group A Posterior", "Group B Posterior"], "top": 30},
        "grid": {"top": 70, "bottom": 50, "left": 60, "right": 20},
        "xAxis": {
            "type": "value",
            "name": metric_label,
            "nameLocation": "middle",
            "nameGap": 30,
            "axisLabel": {"formatter": "{value}"},
        },
        "yAxis": {"type": "value", "name": "Density", "axisLabel": {"formatter": "{value}"}},
        "series": [
            _line_series("Group A Posterior", data_a, C_A),
            _line_series("Group B Posterior", data_b, C_B),
        ],
        "color": [C_A, C_B],
    }


# ── 2. Delta posterior distribution (with MDE reference line) ─────
def delta_chart(result: "BayesianResult", mde: float, metric_label: str = "Metric") -> dict:
    """Posterior of delta = B - A, with vertical reference lines at 0 and MDE."""
    data_d = _kde(result.delta_samples)

    prob_text = f"P(B>A)={result.prob_b_better:.1%}  |  P(delta>MDE)={result.prob_practical:.1%}"

    series = _line_series("delta = B − A", data_d, C_DELTA)
    series["markLine"] = {
        "symbol": "none",
        "label": {"show": True, "position": "insideEndTop", "fontSize": 11},
        "lineStyle": {"type": "dashed", "width": 1.5},
        "data": [
            {"xAxis": 0,   "name": "Δ=0",  "lineStyle": {"color": "#555"}},
            {"xAxis": mde, "name": f"MDE={mde}", "lineStyle": {"color": C_A}},
        ],
    }

    return {
        "title": {"text": prob_text, "textStyle": {"fontSize": 12, "fontWeight": "normal"}, "top": 5},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "grid": {"top": 55, "bottom": 50, "left": 60, "right": 20},
        "xAxis": {
            "type": "value",
            "name": f"B − A ({metric_label})",
            "nameLocation": "middle",
            "nameGap": 30,
        },
        "yAxis": {"type": "value", "name": "Density"},
        "series": [series],
        "color": [C_DELTA],
    }


# ── 3. Expected loss comparison chart ─────────────────────────────
def loss_chart(result: "BayesianResult", loss_threshold: float) -> dict:
    """Bar chart comparing expected loss for choosing A vs B, with threshold line."""
    loss_a = round(result.expected_loss_a, 6)
    loss_b = round(result.expected_loss_b, 6)

    return {
        "tooltip": {"trigger": "axis"},
        "grid": {"top": 40, "bottom": 60, "left": 70, "right": 20},
        "xAxis": {
            "type": "category",
            "data": ["Expected Loss (Keep A)", "Expected Loss (Launch B)"],
            "axisLabel": {"fontSize": 12},
        },
        "yAxis": {"type": "value", "name": "Expected Loss"},
        "series": [
            {
                "type": "bar",
                "data": [
                    {"value": loss_a, "itemStyle": {"color": C_A}},
                    {"value": loss_b, "itemStyle": {"color": C_B}},
                ],
                "barWidth": "40%",
                "label": {"show": True, "position": "top", "formatter": "{c}"},
                "markLine": {
                    "symbol": "none",
                    "label": {
                        "show": True,
                        "formatter": f"Threshold {loss_threshold}",
                        "position": "insideEndTop",
                    },
                    "lineStyle": {"color": C_THRESH, "type": "dashed", "width": 2},
                    "data": [{"yAxis": loss_threshold}],
                },
            }
        ],
    }


# ── 4. Frequentist mean comparison chart (with CI markArea) ───────
def freq_chart(result: "FrequentistResult", metric_label: str = "Metric", mde: float | None = None) -> dict:
    """
    Bar chart of A/B means.
    Group B includes a shaded 95% CI band (markArea) for delta,
    plus a delta=0 reference line to aid significance interpretation.
    """
    ci_low, ci_high = result.ci
    sig_label = "✓ Significant" if result.significant else "✗ Not Significant"
    sig_color = C_B if result.significant else "#888"

    # Reference: Group A mean (horizontal baseline)
    ref_a = round(result.mean_a, 6)
    ci_abs_low  = round(ref_a + ci_low, 6)
    ci_abs_high = round(ref_a + ci_high, 6)

    # MDE reference line (A mean + MDE)
    mark_lines = [{"yAxis": ref_a, "name": f"A Mean {ref_a}", "lineStyle": {"color": C_A}}]
    if mde is not None:
        ref_mde = round(ref_a + mde, 6)
        mark_lines.append({"yAxis": ref_mde, "name": f"A+MDE {ref_mde}", "lineStyle": {"color": C_THRESH, "type": "dotted"}})

    title_text = (
        f"p = {result.p_value:.4f}  {sig_label}  |  "
        f"Effect size = {result.effect_size:.4f}  |  "
        f"delta 95% CI = [{ci_low:+.4f}, {ci_high:+.4f}]"
    )
    if mde is not None:
        title_text += f"  |  MDE = {mde:.4f}"

    return {
        "title": {
            "text": title_text,
            "textStyle": {"fontSize": 12, "fontWeight": "normal", "color": sig_color},
            "top": 5,
        },
        "tooltip": {"trigger": "axis"},
        "legend": {"show": False},
        "grid": {"top": 60, "bottom": 60, "left": 70, "right": 20},
        "xAxis": {
            "type": "category",
            "data": ["Group A (Control)", "Group B (Treatment)"],
            "axisLabel": {"fontSize": 13},
        },
        "yAxis": {"type": "value", "name": metric_label},
        "series": [
            {
                "name": "Mean",
                "type": "bar",
                "data": [
                    {"value": round(result.mean_a, 5), "itemStyle": {"color": C_A}},
                    {"value": round(result.mean_b, 5), "itemStyle": {"color": C_B}},
                ],
                "barWidth": "40%",
                "label": {"show": True, "position": "top", "formatter": "{c}"},
                "markLine": {
                    "symbol": "none",
                    "lineStyle": {"type": "dashed", "width": 1.5},
                    "label": {
                        "position": "insideEndTop",
                        "fontSize": 10,
                    },
                    "data": mark_lines,
                },
                # delta 95% CI shaded band over Group B bar
                "markArea": {
                    "silent": True,
                    "itemStyle": {"color": C_B, "opacity": 0.10},
                    "label": {
                        "show": True,
                        "position": "inside",
                        "fontSize": 9,
                        "color": "#333",
                        "formatter": "95% CI",
                    },
                    "data": [[
                        {"xAxis": "Group B (Treatment)", "yAxis": ci_abs_low},
                        {"xAxis": "Group B (Treatment)", "yAxis": ci_abs_high},
                    ]],
                },
            }
        ],
    }


# ── 5. Sequential testing boundary chart ─────────────────────────
def sequential_chart(result: "SequentialResult") -> dict:
    """
    Chart showing sequential test statistics over time with rejection boundaries.

    Displays:
    - Test statistic path over looks
    - Upper and lower rejection boundaries
    - Information rates on x-axis
    """
    looks = result.looks
    boundaries = result.boundary_values
    info_rates = result.information_rates

    # Create x-axis labels as percentages
    x_labels = [f"{int(t * 100)}%" for t in info_rates]

    # Prepare data points - use category indices instead of values
    look_indices = list(range(1, len(looks) + 1))
    stats = [look.statistic for look in looks]
    decisions = [look.decision for look in looks]

    # Symmetric boundaries (two-sided test)
    upper_bound = boundaries[:len(looks)]
    lower_bound = [-b for b in upper_bound]

    # Determine colors for each point based on decision
    point_colors = []
    for dec in decisions:
        if dec == "Reject H0":
            point_colors.append("#e74c3c")  # Red - rejected
        else:
            point_colors.append("#3498db")  # Blue - continuing

    # Build series - use category axis for simplicity
    series = []

    # Planned boundary lines (dashed)
    series.append({
        "name": "Upper Bound",
        "type": "line",
        "data": planned_upper if 'planned_upper' in locals() else boundaries,
        "lineStyle": {"color": "#95a5a6", "type": "dashed", "width": 2},
        "symbol": "none",
    })
    series.append({
        "name": "Lower Bound",
        "type": "line",
        "data": [-b for b in boundaries],
        "lineStyle": {"color": "#95a5a6", "type": "dashed", "width": 2},
        "symbol": "none",
    })

    # Zero line
    series.append({
        "name": "Zero",
        "type": "line",
        "data": [0 for _ in info_rates],
        "lineStyle": {"color": "#7f8c8d", "type": "solid", "width": 1},
        "symbol": "none",
    })

    # Actual statistic path - only for executed looks
    # Create data with None for unexecuted looks
    stat_data = []
    for i in range(len(info_rates)):
        if i < len(stats):
            stat_data.append(stats[i])
        else:
            stat_data.append(None)

    series.append({
        "name": "Test Statistic",
        "type": "line",
        "data": stat_data,
        "lineStyle": {"color": "#3498db", "width": 3},
        "itemStyle": {
            "color": point_colors + ["#3498db"] * (len(info_rates) - len(point_colors)),
            "borderColor": "#2980b9",
            "borderWidth": 2,
        },
        "symbol": "circle",
        "symbolSize": 10,
        "connectNulls": False,
    })

    # Determine y-axis range
    all_values = boundaries + [-b for b in boundaries] + stats
    y_min = min(all_values) * 1.2
    y_max = max(all_values) * 1.2
    y_range = max(abs(y_min), abs(y_max))

    return {
        "title": {
            "text": f"Sequential Test: {result.method} | α = {result.alpha:.3f}",
            "subtext": f"Final Decision: {result.final_decision}",
            "left": "center",
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "cross"},
        },
        "legend": {
            "data": ["Test Statistic", "Upper Bound", "Lower Bound"],
            "top": 50,
        },
        "grid": {
            "top": 100,
            "bottom": 70,
            "left": 60,
            "right": 40,
        },
        "xAxis": {
            "type": "category",
            "name": "Information Rate",
            "nameLocation": "middle",
            "nameGap": 40,
            "data": x_labels,
        },
        "yAxis": {
            "type": "value",
            "name": "Test Statistic (z-score)",
            "min": -y_range,
            "max": y_range,
        },
        "series": series,
        "color": ["#3498db", "#95a5a6", "#95a5a6"],
    }


def sequential_metrics_chart(result: "SequentialResult") -> dict:
    """
    Chart showing key metrics over sequential looks:
    - Sample sizes over time
    - Treatment effect (delta) over time
    - P-values over time
    """
    looks = result.looks

    look_indices = list(range(1, len(looks) + 1))
    n_a = [look.n_a for look in looks]
    n_b = [look.n_b for look in looks]
    deltas = [look.mean_b - look.mean_a for look in looks]
    p_values = [look.p_value for look in looks]

    return {
        "title": {
            "text": "Metrics Evolution Over Looks",
            "left": "center",
        },
        "tooltip": {
            "trigger": "axis",
        },
        "legend": {
            "data": ["N (A)", "N (B)", "Delta (B-A)", "P-value"],
            "top": 40,
        },
        "grid": {
            "top": 80,
            "bottom": 50,
            "left": 60,
            "right": 60,
        },
        "xAxis": {
            "type": "category",
            "data": [f"Look {i}" for i in look_indices],
        },
        "yAxis": [
            {
                "type": "value",
                "name": "Sample Size",
                "position": "left",
            },
            {
                "type": "value",
                "name": "Delta",
                "position": "right",
                "offset": 0,
            },
            {
                "type": "value",
                "name": "P-value",
                "position": "right",
                "offset": 60,
                "min": 0,
                "max": 1,
            },
        ],
        "series": [
            {
                "name": "N (A)",
                "type": "bar",
                "data": n_a,
                "itemStyle": {"color": "#e74c3c", "opacity": 0.7},
            },
            {
                "name": "N (B)",
                "type": "bar",
                "data": n_b,
                "itemStyle": {"color": "#27ae60", "opacity": 0.7},
            },
            {
                "name": "Delta (B-A)",
                "type": "line",
                "yAxisIndex": 1,
                "data": deltas,
                "itemStyle": {"color": "#8e44ad"},
            },
            {
                "name": "P-value",
                "type": "line",
                "yAxisIndex": 2,
                "data": p_values,
                "itemStyle": {"color": "#f39c12"},
                "markLine": {
                    "data": [{"yAxis": 0.05, "name": "α=0.05"}],
                    "lineStyle": {"type": "dashed", "color": "#e74c3c"},
                },
            },
        ],
    }
