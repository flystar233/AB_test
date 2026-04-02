"""
频率派方法：
  - 二值指标：双比例 Z-test
  - 连续指标：Welch t-test（不假设等方差）
"""
import numpy as np
from scipy import stats

from .metrics import FrequentistResult


def two_proportion_ztest(
    data_a: np.ndarray,
    data_b: np.ndarray,
    alpha: float = 0.05,
) -> FrequentistResult:
    """
    双比例 Z-test，适用于 0/1 二值数据（点击、留存、转化）。

    前提：n*p > 5 且 n*(1-p) > 5（大样本正态近似）
    """
    n_a, n_b = len(data_a), len(data_b)
    x_a, x_b = data_a.sum(), data_b.sum()
    p_a, p_b = x_a / n_a, x_b / n_b

    # 合并比例（H0 下两组相同）
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

    z = (p_b - p_a) / se_pool
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    # 置信区间（用各自比例，不合并方差）
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    delta = p_b - p_a
    ci = (delta - z_crit * se_diff, delta + z_crit * se_diff)

    # Cohen's h（二值效应量）
    h = 2 * np.arcsin(np.sqrt(p_b)) - 2 * np.arcsin(np.sqrt(p_a))

    return FrequentistResult(
        statistic=float(z),
        p_value=p_value,
        ci=ci,
        significant=p_value < alpha,
        effect_size=float(h),
        mean_a=float(p_a),
        mean_b=float(p_b),
        delta=float(delta),
    )


def welch_ttest(
    data_a: np.ndarray,
    data_b: np.ndarray,
    alpha: float = 0.05,
) -> FrequentistResult:
    """
    Welch t-test，适用于连续指标（收入、金额）。
    不假设两组方差相等，对重尾分布更稳健。
    """
    t_stat, p_value = stats.ttest_ind(data_b, data_a, equal_var=False)

    mean_a, mean_b = float(data_a.mean()), float(data_b.mean())
    delta = mean_b - mean_a

    # Welch-Satterthwaite 自由度
    var_a, var_b = data_a.var(ddof=1), data_b.var(ddof=1)
    n_a, n_b = len(data_a), len(data_b)
    se = np.sqrt(var_a / n_a + var_b / n_b)

    num = (var_a / n_a + var_b / n_b) ** 2
    den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / den

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci = (delta - t_crit * se, delta + t_crit * se)

    # Cohen's d（连续效应量）
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    d = delta / pooled_std if pooled_std > 0 else 0.0

    return FrequentistResult(
        statistic=float(t_stat),
        p_value=float(p_value),
        ci=ci,
        significant=float(p_value) < alpha,
        effect_size=float(d),
        mean_a=mean_a,
        mean_b=mean_b,
        delta=delta,
    )
