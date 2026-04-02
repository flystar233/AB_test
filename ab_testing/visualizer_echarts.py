"""
ECharts option dict 生成器，供 streamlit-echarts 的 st_echarts() 直接使用。
所有函数返回标准 ECharts option dict（Python dict），不依赖 pyecharts。
"""
from __future__ import annotations
import numpy as np
from scipy.stats import gaussian_kde
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import BayesianResult, FrequentistResult

# ── 配色 ──────────────────────────────────────────────────────────
C_A        = "#e74c3c"   # 红：A 组
C_B        = "#27ae60"   # 绿：B 组
C_DELTA    = "#8e44ad"   # 紫：delta
C_THRESH   = "#e67e22"   # 橙：阈值线


def _kde(data: np.ndarray, n: int = 300):
    """返回 KDE 曲线的 [[x, y], ...] 列表，用于 ECharts value 轴。"""
    bw = gaussian_kde(data, bw_method="scott")
    margin = (data.max() - data.min()) * 0.12
    xs = np.linspace(data.min() - margin, data.max() + margin, n)
    ys = bw(xs)
    return [[round(float(x), 6), round(float(y), 6)] for x, y in zip(xs, ys)]


def _line_series(name: str, data: list, color: str, opacity: float = 0.25) -> dict:
    """通用平滑面积折线 series。"""
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


# ── 1. 后验分布图（A vs B）───────────────────────────────────────
def posterior_chart(result: "BayesianResult", metric_label: str = "指标") -> dict:
    """双后验 KDE 曲线，展示 A/B 的参数分布。"""
    data_a = _kde(result.posterior_a)
    data_b = _kde(result.posterior_b)

    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": {"data": ["A 组后验", "B 组后验"], "top": 30},
        "grid": {"top": 70, "bottom": 50, "left": 60, "right": 20},
        "xAxis": {
            "type": "value",
            "name": metric_label,
            "nameLocation": "middle",
            "nameGap": 30,
            "axisLabel": {"formatter": "{value}"},
        },
        "yAxis": {"type": "value", "name": "密度", "axisLabel": {"formatter": "{value}"}},
        "series": [
            _line_series("A 组后验", data_a, C_A),
            _line_series("B 组后验", data_b, C_B),
        ],
        "color": [C_A, C_B],
    }


# ── 2. Delta 后验分布图（含 MDE 基准线）──────────────────────────
def delta_chart(result: "BayesianResult", mde: float, metric_label: str = "指标") -> dict:
    """delta = B - A 的后验分布，含 delta=0 和 MDE 垂直参考线。"""
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
            "name": f"B − A（{metric_label}）",
            "nameLocation": "middle",
            "nameGap": 30,
        },
        "yAxis": {"type": "value", "name": "密度"},
        "series": [series],
        "color": [C_DELTA],
    }


# ── 3. 期望损失对比图 ─────────────────────────────────────────────
def loss_chart(result: "BayesianResult", loss_threshold: float) -> dict:
    """选 A / 选 B 的期望损失柱状图，含阈值参考线。"""
    loss_a = round(result.expected_loss_a, 6)
    loss_b = round(result.expected_loss_b, 6)

    return {
        "tooltip": {"trigger": "axis"},
        "grid": {"top": 40, "bottom": 60, "left": 70, "right": 20},
        "xAxis": {
            "type": "category",
            "data": ["选 A 的期望损失", "选 B 的期望损失"],
            "axisLabel": {"fontSize": 12},
        },
        "yAxis": {"type": "value", "name": "期望损失"},
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
                        "formatter": f"阈值 {loss_threshold}",
                        "position": "insideEndTop",
                    },
                    "lineStyle": {"color": C_THRESH, "type": "dashed", "width": 2},
                    "data": [{"yAxis": loss_threshold}],
                },
            }
        ],
    }


# ── 4. 频率派均值对比图（含置信区间 markArea）───────────────────
def freq_chart(result: "FrequentistResult", metric_label: str = "指标") -> dict:
    """
    A/B 均值柱状图。
    B 组附加 delta 95% CI 的阴影区间（markArea），delta=0 参考线辅助判断显著性。
    """
    ci_low, ci_high = result.ci
    sig_label = "✓ 显著" if result.significant else "✗ 不显著"
    sig_color = C_B if result.significant else "#888"

    # 基准线：A 组均值（水平参考）
    ref_a = round(result.mean_a, 6)
    ci_abs_low  = round(ref_a + ci_low, 6)   # B 均值 + CI下界 ≈ 置信区间下端绝对值
    ci_abs_high = round(ref_a + ci_high, 6)  # B 均值 + CI上界 ≈ 置信区间上端绝对值

    return {
        "title": {
            "text": (
                f"p = {result.p_value:.4f}  {sig_label}  |  "
                f"效应量 = {result.effect_size:.4f}  |  "
                f"delta 95% CI = [{ci_low:+.4f}, {ci_high:+.4f}]"
            ),
            "textStyle": {"fontSize": 12, "fontWeight": "normal", "color": sig_color},
            "top": 5,
        },
        "tooltip": {"trigger": "axis"},
        "legend": {"show": False},
        "grid": {"top": 60, "bottom": 60, "left": 70, "right": 20},
        "xAxis": {
            "type": "category",
            "data": ["A 组（对照）", "B 组（实验）"],
            "axisLabel": {"fontSize": 13},
        },
        "yAxis": {"type": "value", "name": metric_label},
        "series": [
            {
                "name": "均值",
                "type": "bar",
                "data": [
                    {"value": round(result.mean_a, 5), "itemStyle": {"color": C_A}},
                    {"value": round(result.mean_b, 5), "itemStyle": {"color": C_B}},
                ],
                "barWidth": "40%",
                "label": {"show": True, "position": "top", "formatter": "{c}"},
                # A 组均值参考线
                "markLine": {
                    "symbol": "none",
                    "lineStyle": {"color": C_A, "type": "dashed", "width": 1.5},
                    "label": {
                        "formatter": f"A 均值 {ref_a}",
                        "position": "insideEndTop",
                        "fontSize": 10,
                        "color": C_A,
                    },
                    "data": [{"yAxis": ref_a}],
                },
                # delta 95% CI 阴影区间（覆盖 B 柱）
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
                        {"xAxis": "B 组（实验）", "yAxis": ci_abs_low},
                        {"xAxis": "B 组（实验）", "yAxis": ci_abs_high},
                    ]],
                },
            }
        ],
    }
