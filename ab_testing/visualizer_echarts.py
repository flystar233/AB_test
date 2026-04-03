"""
ECharts option dict generators for use with streamlit-echarts st_echarts().
All functions return standard ECharts option dicts (Python dicts); no pyecharts dependency.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import gaussian_kde
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import BayesianResult, FrequentistResult

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
