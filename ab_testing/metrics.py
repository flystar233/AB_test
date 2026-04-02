"""
决策指标：纯函数层，输入后验样本 / 统计量，输出三大决策指标。
与具体模型解耦，贝叶斯和频率派共用同一套数据类。
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class FrequentistResult:
    statistic: float                     # z-score 或 t-statistic
    p_value: float
    ci: Tuple[float, float]              # 置信区间 (lower, upper)
    significant: bool
    effect_size: float                   # Cohen's h (二值) 或 Cohen's d (连续)
    mean_a: float
    mean_b: float
    delta: float                         # mean_b - mean_a


@dataclass
class BayesianResult:
    prob_b_better: float                 # P(B > A)
    prob_practical: float                # P(delta > MDE)
    expected_loss_a: float               # 选 A 的期望损失
    expected_loss_b: float               # 选 B 的期望损失
    posterior_a: np.ndarray              # 后验样本（mu 或 p）
    posterior_b: np.ndarray
    delta_samples: np.ndarray
    mean_a: float
    mean_b: float
    delta_mean: float


def compute_bayesian_metrics(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    mde: float,
) -> BayesianResult:
    """
    给定 A/B 后验样本，计算所有贝叶斯决策指标。

    Args:
        samples_a: A 组后验参数样本（转化率 p 或收入均值 mu）
        samples_b: B 组后验参数样本
        mde:       最小可检测提升（与指标同量纲）
    """
    delta = samples_b - samples_a

    prob_b_better = float(np.mean(delta > 0))
    prob_practical = float(np.mean(delta > mde))
    expected_loss_a = float(np.mean(np.maximum(delta, 0)))   # 选 A 时错过 B 的收益
    expected_loss_b = float(np.mean(np.maximum(-delta, 0)))  # 选 B 时错过 A 的收益

    return BayesianResult(
        prob_b_better=prob_b_better,
        prob_practical=prob_practical,
        expected_loss_a=expected_loss_a,
        expected_loss_b=expected_loss_b,
        posterior_a=samples_a,
        posterior_b=samples_b,
        delta_samples=delta,
        mean_a=float(samples_a.mean()),
        mean_b=float(samples_b.mean()),
        delta_mean=float(delta.mean()),
    )


def bayesian_decision(result: BayesianResult, loss_threshold: float) -> str:
    """
    根据期望损失给出停止决策。

    expected_loss_a = E[max(B-A, 0)] = 选 A 的机会成本（B 有多大概率更好且好多少）
    expected_loss_b = E[max(A-B, 0)] = 选 B 的机会成本（A 有多大概率更好且好多少）

    - 若选 A 的机会成本很低 → B 几乎不可能更好 → 安全地保持 A
    - 若选 B 的机会成本很低 → A 几乎不可能更好 → 可以安全地切换到 B
    """
    if result.expected_loss_a < loss_threshold:
        return "保持 A"
    elif result.expected_loss_b < loss_threshold:
        return "上线 B"
    else:
        return "继续收集数据"


def frequentist_decision(result: FrequentistResult) -> str:
    """根据显著性给出停止决策。"""
    if not result.significant:
        return "保持 A（无显著差异）"
    return "上线 B" if result.delta > 0 else "保持 A（B 更差）"
