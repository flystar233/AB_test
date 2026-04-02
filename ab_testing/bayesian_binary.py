"""
贝叶斯二值指标：Beta-Bernoulli 共轭模型（转化率、留存率、点击率）。

Prior:     p ~ Beta(alpha, beta)
Likelihood: x_i ~ Bernoulli(p)
Posterior:  p | data ~ Beta(alpha + successes, beta + failures)

无需 MCMC，有解析闭合解。
"""
import numpy as np

from .metrics import BayesianResult, compute_bayesian_metrics


class BayesianBinary:
    """
    Args:
        historical_rate:  历史转化率，用于设置先验均值（默认 0.5 无信息先验）
        prior_strength:   等效历史样本量，越大表示越信任历史数据（默认 2 近似无信息）
        n_samples:        后验蒙特卡洛采样数
        mde:              最小可检测提升（绝对值，与转化率同量纲）
    """

    def __init__(
        self,
        historical_rate: float = 0.5,
        prior_strength: int = 2,
        n_samples: int = 200_000,
        mde: float = 0.005,
    ):
        self.historical_rate = historical_rate
        self.prior_strength = prior_strength
        self.n_samples = n_samples
        self.mde = mde

        # 先验参数
        self.prior_alpha = historical_rate * prior_strength
        self.prior_beta = (1 - historical_rate) * prior_strength

    def fit(self, data_a: np.ndarray, data_b: np.ndarray) -> BayesianResult:
        """
        更新后验分布并计算决策指标。

        Args:
            data_a: A 组 0/1 数组
            data_b: B 组 0/1 数组
        """
        # 后验参数（共轭更新）
        post_alpha_a = self.prior_alpha + data_a.sum()
        post_beta_a = self.prior_beta + (len(data_a) - data_a.sum())

        post_alpha_b = self.prior_alpha + data_b.sum()
        post_beta_b = self.prior_beta + (len(data_b) - data_b.sum())

        # 后验均值（用于输出）
        self.posterior_mean_a = post_alpha_a / (post_alpha_a + post_beta_a)
        self.posterior_mean_b = post_alpha_b / (post_alpha_b + post_beta_b)

        # 后验抽样
        rng = np.random.default_rng()
        samples_a = rng.beta(post_alpha_a, post_beta_a, self.n_samples)
        samples_b = rng.beta(post_alpha_b, post_beta_b, self.n_samples)

        return compute_bayesian_metrics(samples_a, samples_b, self.mde)
