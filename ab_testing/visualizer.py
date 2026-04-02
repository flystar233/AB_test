"""
可视化模块：输入分析结果，输出 matplotlib 图表。
支持 show（交互展示）和 save（保存文件）两种输出模式。
"""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm

# Windows 中文字体回退：优先使用 Microsoft YaHei，其次 SimHei
def _set_chinese_font():
    candidates = ["Microsoft YaHei", "SimHei", "STHeiti", "Noto Sans CJK SC"]
    for name in candidates:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    # 未找到中文字体时不报错，图表标签退化为 ASCII

_set_chinese_font()

if TYPE_CHECKING:
    from .metrics import BayesianResult, FrequentistResult


# 配色方案
COLOR_A = "#A60628"
COLOR_B = "#467821"
COLOR_DELTA = "#7A68A6"
COLOR_THRESHOLD = "#2B2B2B"


def plot_bayesian(
    result: "BayesianResult",
    mde: float,
    loss_threshold: float,
    metric_label: str = "指标",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    贝叶斯结果三联图：
      - 图1：A/B 后验分布
      - 图2：delta 后验分布（含 MDE 阈值）
      - 图3：期望损失对比
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    fig.suptitle("贝叶斯 A/B 测试结果", fontsize=14, fontweight="bold")

    # ── 图1：后验分布 ──────────────────────────────────────────────
    ax = axes[0]
    ax.hist(result.posterior_a, bins=60, alpha=0.7, density=True,
            color=COLOR_A, label=f"A 组后验  (均值={result.mean_a:.4f})")
    ax.hist(result.posterior_b, bins=60, alpha=0.7, density=True,
            color=COLOR_B, label=f"B 组后验  (均值={result.mean_b:.4f})")
    ax.set_title(f"后验分布：A vs B（{metric_label}）")
    ax.set_xlabel(metric_label)
    ax.legend()

    # ── 图2：delta 后验分布 ────────────────────────────────────────
    ax = axes[1]
    ax.hist(result.delta_samples, bins=60, alpha=0.85, density=True,
            color=COLOR_DELTA, label=f"delta = B - A  (均值={result.delta_mean:.4f})")
    ax.axvline(0, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.5, label="delta = 0")
    ax.axvline(mde, color="red", linestyle="--", linewidth=1.5, label=f"MDE = {mde}")

    # 填充 P(B > A) 区域
    d = result.delta_samples
    kde_x = np.linspace(d.min(), d.max(), 500)
    prob_text = f"P(B>A)={result.prob_b_better:.1%}  P(delta>MDE)={result.prob_practical:.1%}"
    ax.set_title(f"delta 后验分布  |  {prob_text}")
    ax.set_xlabel(f"B - A ({metric_label})")
    ax.legend()

    # ── 图3：期望损失 ──────────────────────────────────────────────
    ax = axes[2]
    bars = ax.bar(
        ["选择 A 的期望损失", "选择 B 的期望损失"],
        [result.expected_loss_a, result.expected_loss_b],
        color=[COLOR_A, COLOR_B],
        alpha=0.85,
        width=0.5,
    )
    ax.axhline(loss_threshold, color=COLOR_THRESHOLD, linestyle="--",
               linewidth=1.5, label=f"阈值 = {loss_threshold}")

    for bar, val in zip(bars, [result.expected_loss_a, result.expected_loss_b]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + loss_threshold * 0.05,
                f"{val:.5f}", ha="center", va="bottom", fontsize=10)

    ax.set_title("期望损失对比（工业停止准则）")
    ax.set_ylabel(f"期望损失（{metric_label}）")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存至：{save_path}")
    if show:
        plt.show()
    return fig


def plot_frequentist(
    result: "FrequentistResult",
    metric_label: str = "指标",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    频率派结果双联图：
      - 图1：均值对比 + 置信区间
      - 图2：效应量与显著性标注
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("频率派 A/B 测试结果", fontsize=14, fontweight="bold")

    # ── 图1：均值 + CI ─────────────────────────────────────────────
    ax = axes[0]
    means = [result.mean_a, result.mean_b]
    colors = [COLOR_A, COLOR_B]
    bars = ax.bar(["A 组", "B 组"], means, color=colors, alpha=0.85, width=0.4)

    # CI 标注在 B 组 delta 上
    ci_low, ci_high = result.ci
    mid = (result.mean_a + result.mean_b) / 2
    ax.errorbar(
        x=1, y=result.mean_b,
        yerr=[[result.mean_b - (result.mean_b + ci_low)],
              [(result.mean_b + ci_high) - result.mean_b]],
        fmt="none", color="black", capsize=6, linewidth=2,
    )
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_title(f"均值对比（{metric_label}）")
    ax.set_ylabel(metric_label)

    # ── 图2：显著性摘要 ───────────────────────────────────────────
    ax = axes[1]
    ax.axis("off")
    sig_color = COLOR_B if result.significant else "gray"
    sig_text = "显著 ✓" if result.significant else "不显著 ✗"

    summary_lines = [
        ("统计量", f"{result.statistic:.4f}"),
        ("p 值", f"{result.p_value:.4f}"),
        ("显著性", sig_text),
        ("效应量", f"{result.effect_size:.4f}"),
        ("delta", f"{result.delta:+.4f}"),
        (f"delta 95% CI", f"[{result.ci[0]:+.4f}, {result.ci[1]:+.4f}]"),
    ]

    y_pos = 0.85
    for label, value in summary_lines:
        color = sig_color if label == "显著性" else "black"
        ax.text(0.1, y_pos, f"{label}：", fontsize=11, transform=ax.transAxes,
                ha="left", color="gray")
        ax.text(0.5, y_pos, value, fontsize=11, transform=ax.transAxes,
                ha="left", color=color, fontweight="bold")
        y_pos -= 0.12

    ax.set_title("显著性摘要")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存至：{save_path}")
    if show:
        plt.show()
    return fig
