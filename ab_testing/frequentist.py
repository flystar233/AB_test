"""
Frequentist methods:
  - Binary metrics:     Two-proportion Z-test
  - Continuous metrics: Welch t-test (unequal variances)
"""
import warnings
import numpy as np
from scipy import stats

from .metrics import FrequentistResult


def two_proportion_ztest(
    data_a: np.ndarray,
    data_b: np.ndarray,
    alpha: float = 0.05,
) -> FrequentistResult:
    """
    Two-proportion Z-test for binary data (clicks, retention, conversions).

    Assumption: n*p > 5 and n*(1-p) > 5 (large-sample normal approximation)
    """
    n_a, n_b = len(data_a), len(data_b)
    x_a, x_b = data_a.sum(), data_b.sum()
    p_a, p_b = x_a / n_a, x_b / n_b

    # Check large-sample assumption
    if x_a < 5 or (n_a - x_a) < 5 or x_b < 5 or (n_b - x_b) < 5:
        warnings.warn(
            "One or more cells have fewer than 5 successes/failures. "
            "Normal approximation may be inaccurate; consider increasing sample size.",
            UserWarning,
            stacklevel=2,
        )

    # Pooled proportion under H0
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

    if se_pool == 0:
        # Both groups have 0% or 100% conversion — z is undefined
        return FrequentistResult(
            statistic=0.0,
            p_value=1.0,
            ci=(0.0, 0.0),
            significant=False,
            effect_size=0.0,
            mean_a=float(p_a),
            mean_b=float(p_b),
            delta=float(p_b - p_a),
        )

    z = (p_b - p_a) / se_pool
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Confidence interval (unpooled SE for the difference)
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    delta = p_b - p_a
    ci = (delta - z_crit * se_diff, delta + z_crit * se_diff)

    # Cohen's h (binary effect size)
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
    Welch t-test for continuous metrics (revenue, spend, etc.).
    Does not assume equal variances; more robust to heavy-tailed distributions.
    """
    t_stat, p_value = stats.ttest_ind(data_b, data_a, equal_var=False)

    mean_a, mean_b = float(data_a.mean()), float(data_b.mean())
    delta = mean_b - mean_a

    # Welch-Satterthwaite degrees of freedom
    var_a, var_b = data_a.var(ddof=1), data_b.var(ddof=1)
    n_a, n_b = len(data_a), len(data_b)
    se = np.sqrt(var_a / n_a + var_b / n_b)

    num = (var_a / n_a + var_b / n_b) ** 2
    den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / den

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci = (delta - t_crit * se, delta + t_crit * se)

    # Cohen's d (continuous effect size)
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
