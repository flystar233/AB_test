"""
ABTestPipeline：统一入口，路由到正确的分析器，聚合并输出结果。

支持：
  - metric_type: "binary"（转化率/留存率）或 "continuous"（收入/GMV）
  - method:      "frequentist"、"bayesian" 或 "both"（同时运行两种，默认）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import textwrap

import numpy as np
import pandas as pd

from .metrics import FrequentistResult, BayesianResult, bayesian_decision, frequentist_decision
from .frequentist import two_proportion_ztest, welch_ttest
from .bayesian_binary import BayesianBinary
from .bayesian_continuous import BayesianContinuous


@dataclass
class ABTestResult:
    metric_type: str
    method: str
    frequentist: Optional[FrequentistResult] = None
    bayesian: Optional[BayesianResult] = None
    decision_freq: Optional[str] = None
    decision_bayes: Optional[str] = None

    def summary(self) -> str:
        lines = [
            "=" * 56,
            f"  A/B 测试结果摘要  |  指标类型：{self.metric_type}",
            "=" * 56,
        ]

        if self.frequentist:
            f = self.frequentist
            stat_label = "z 统计量" if self.metric_type == "binary" else "t 统计量"
            lines += [
                "",
                "【频率派】",
                f"  A 组均值：{f.mean_a:.4f}",
                f"  B 组均值：{f.mean_b:.4f}",
                f"  delta  ：{f.delta:+.4f}  (B - A)",
                f"  95% CI ：[{f.ci[0]:+.4f}, {f.ci[1]:+.4f}]",
                f"  {stat_label}：{f.statistic:.4f}",
                f"  p 值   ：{f.p_value:.4f}",
                f"  效应量 ：{f.effect_size:.4f}",
                f"  → 决策 ：{self.decision_freq}",
            ]

        if self.bayesian:
            b = self.bayesian
            lines += [
                "",
                "【贝叶斯】",
                f"  A 组后验均值：{b.mean_a:.4f}",
                f"  B 组后验均值：{b.mean_b:.4f}",
                f"  delta 后验均值：{b.delta_mean:+.4f}",
                f"  P(B > A)        ：{b.prob_b_better:.1%}",
                f"  P(delta > MDE)  ：{b.prob_practical:.1%}",
                f"  选 A 的期望损失 ：{b.expected_loss_a:.5f}",
                f"  选 B 的期望损失 ：{b.expected_loss_b:.5f}",
                f"  → 决策 ：{self.decision_bayes}",
            ]

        lines.append("=" * 56)
        return "\n".join(lines)

    def print_summary(self):
        print(self.summary())


class ABTestPipeline:
    """
    A/B 测试主流程。

    Args:
        metric_type:       "binary" 或 "continuous"
        method:            "frequentist"、"bayesian" 或 "both"
        alpha:             频率派显著性水平（默认 0.05）
        mde:               最小可检测提升（与指标同量纲）
        loss_threshold:    贝叶斯期望损失停止阈值
        prior_strength:    贝叶斯先验强度（等效历史样本量）
        historical_rate:   二值先验：历史转化率（binary only）
        historical_mean:   连续先验：历史均值（continuous only）
        historical_std:    连续先验：历史标准差（continuous only）
        n_samples:         贝叶斯后验蒙特卡洛采样数（binary only）
        mcmc_draws:        MCMC 采样数（continuous only）

    Examples:
        >>> pipeline = ABTestPipeline(metric_type="binary", method="both",
        ...                           historical_rate=0.44, mde=0.005)
        >>> result = pipeline.run(data_a, data_b)
        >>> result.print_summary()
        >>> pipeline.plot(result)
    """

    def __init__(
        self,
        metric_type: Literal["binary", "continuous"] = "binary",
        method: Literal["frequentist", "bayesian", "both"] = "both",
        alpha: float = 0.05,
        mde: float = 0.005,
        loss_threshold: float = 0.001,
        prior_strength: int = 100,
        historical_rate: float = 0.5,
        historical_mean: float = None,
        historical_std: float = None,
        nu_expected: float = 30.0,
        n_samples: int = 200_000,
        mcmc_draws: int = 2000,
    ):
        self.metric_type = metric_type
        self.method = method
        self.alpha = alpha
        self.mde = mde
        self.loss_threshold = loss_threshold
        self.prior_strength = prior_strength
        self.historical_rate = historical_rate
        self.historical_mean = historical_mean
        self.historical_std = historical_std
        self.nu_expected = nu_expected
        self.n_samples = n_samples
        self.mcmc_draws = mcmc_draws

    def run(self, data_a: np.ndarray, data_b: np.ndarray) -> ABTestResult:
        """
        运行 A/B 测试分析。

        Args:
            data_a: A 组（对照组）数据
            data_b: B 组（实验组）数据

        Returns:
            ABTestResult 包含所有指标和决策建议
        """
        data_a = np.asarray(data_a, dtype=float)
        data_b = np.asarray(data_b, dtype=float)

        result = ABTestResult(metric_type=self.metric_type, method=self.method)

        # ── 频率派 ─────────────────────────────────────────────────
        if self.method in ("frequentist", "both"):
            if self.metric_type == "binary":
                freq = two_proportion_ztest(data_a, data_b, alpha=self.alpha)
            else:
                freq = welch_ttest(data_a, data_b, alpha=self.alpha)
            result.frequentist = freq
            result.decision_freq = frequentist_decision(freq)

        # ── 贝叶斯 ─────────────────────────────────────────────────
        if self.method in ("bayesian", "both"):
            if self.metric_type == "binary":
                model = BayesianBinary(
                    historical_rate=self.historical_rate,
                    prior_strength=self.prior_strength,
                    n_samples=self.n_samples,
                    mde=self.mde,
                )
            else:
                model = BayesianContinuous(
                    historical_mean=self.historical_mean,
                    historical_std=self.historical_std,
                    prior_strength=self.prior_strength,
                    nu_expected=self.nu_expected,
                    mcmc_draws=self.mcmc_draws,
                    mde=self.mde,
                )
            bayes = model.fit(data_a, data_b)
            result.bayesian = bayes
            result.decision_bayes = bayesian_decision(bayes, self.loss_threshold)

        return result

    def run_from_csv(
        self,
        filepath: str,
        group_col: str,
        metric_col: str,
        control_label: str = "gate_30",
        treatment_label: str = "gate_40",
    ) -> ABTestResult:
        """
        从 CSV 文件加载数据并运行分析。

        Args:
            filepath:         CSV 文件路径
            group_col:        分组列名（如 'version'）
            metric_col:       指标列名（如 'retention_1' 或 'revenue'）
            control_label:    对照组标签（A 组）
            treatment_label:  实验组标签（B 组）
        """
        df = pd.read_csv(filepath)

        data_a = df[df[group_col] == control_label][metric_col].values.astype(float)
        data_b = df[df[group_col] == treatment_label][metric_col].values.astype(float)

        print(f"已加载数据：A 组 {len(data_a)} 条，B 组 {len(data_b)} 条")
        print(f"A 组均值：{data_a.mean():.4f}  |  B 组均值：{data_b.mean():.4f}")
        print()

        return self.run(data_a, data_b)

    def plot(
        self,
        result: ABTestResult,
        metric_label: str = "指标",
        save_dir: Optional[str] = None,
        show: bool = True,
    ):
        """
        绘制分析结果图表。

        Args:
            result:       ABTestResult 对象
            metric_label: 图表中指标的显示名称
            save_dir:     图片保存目录（None 则不保存）
            show:         是否弹出交互窗口
        """
        from .visualizer import plot_bayesian, plot_frequentist
        import os

        if result.frequentist:
            save_path = os.path.join(save_dir, "frequentist_result.png") if save_dir else None
            plot_frequentist(result.frequentist, metric_label=metric_label,
                             save_path=save_path, show=show)

        if result.bayesian:
            save_path = os.path.join(save_dir, "bayesian_result.png") if save_dir else None
            plot_bayesian(result.bayesian, mde=self.mde, loss_threshold=self.loss_threshold,
                          metric_label=metric_label, save_path=save_path, show=show)
