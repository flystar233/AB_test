"""
贝叶斯连续指标：StudentT 模型（收入、GMV、重尾连续数据）。

Prior:      mu ~ Normal, sigma ~ HalfNormal, nu ~ Exponential
Likelihood: revenue_i ~ StudentT(nu, mu, sigma)
Posterior:  无解析解，使用 PyMC v5 NUTS MCMC 采样

PyMC 为可选依赖，仅在调用 .fit() 时才需要：pip install pymc
"""
import numpy as np

from .metrics import BayesianResult, compute_bayesian_metrics


class BayesianContinuous:
    """
    Args:
        historical_mean:   历史数据均值，用于设置 mu 先验（None 则从数据估计）
        historical_std:    历史数据标准差，用于设置 sigma 先验（None 则从数据估计）
        prior_strength:    先验强度，越大先验均值越集中（等效历史样本量）
        nu_expected:       自由度先验期望值，控制尾部厚度
                           nu≈3 极厚尾（高度离群），nu≈30 接近正态分布
                           对应 Exponential 先验：lam = 1 / nu_expected
        mcmc_draws:        NUTS 采样数
        mcmc_tune:         NUTS 调整步数
        target_accept:     NUTS 目标接受率
        mde:               最小可检测提升（绝对值，与收入同量纲）
    """

    def __init__(
        self,
        historical_mean: float = None,
        historical_std: float = None,
        prior_strength: int = 50,
        nu_expected: float = 30.0,
        mcmc_draws: int = 2000,
        mcmc_tune: int = 1000,
        target_accept: float = 0.9,
        mde: float = 3.0,
    ):
        self.historical_mean = historical_mean
        self.historical_std = historical_std
        self.prior_strength = prior_strength
        self.nu_expected = nu_expected
        self.mcmc_draws = mcmc_draws
        self.mcmc_tune = mcmc_tune
        self.target_accept = target_accept
        self.mde = mde

    def fit(self, data_a: np.ndarray, data_b: np.ndarray) -> BayesianResult:
        """
        运行 MCMC 推断后验分布并计算决策指标。

        Args:
            data_a: A 组收入数组（正实数）
            data_b: B 组收入数组（正实数）
        """
        try:
            import pymc as pm
        except ImportError:
            raise ImportError(
                "连续指标贝叶斯分析需要 PyMC v5。\n"
                "请运行：pip install pymc"
            )

        all_data = np.concatenate([data_a, data_b])
        mu_hat    = self.historical_mean if self.historical_mean is not None else float(all_data.mean())
        sigma_hat = self.historical_std  if self.historical_std  is not None else float(all_data.std())

        with pm.Model():
            # 先验
            mu_a    = pm.Normal("mu_a",    mu=mu_hat,    sigma=mu_hat / np.sqrt(self.prior_strength))
            mu_b    = pm.Normal("mu_b",    mu=mu_hat,    sigma=mu_hat / np.sqrt(self.prior_strength))
            sigma_a = pm.HalfNormal("sigma_a", sigma=sigma_hat)
            sigma_b = pm.HalfNormal("sigma_b", sigma=sigma_hat)
            nu      = pm.Exponential("nu", lam=1.0 / self.nu_expected)  # lam 越大→期望 nu 越小→尾部越厚

            # 关注量
            pm.Deterministic("delta",     mu_b - mu_a)
            pm.Deterministic("delta_pct", (mu_b - mu_a) / mu_a)

            # 似然
            pm.StudentT("obs_a", nu=nu, mu=mu_a, sigma=sigma_a, observed=data_a)
            pm.StudentT("obs_b", nu=nu, mu=mu_b, sigma=sigma_b, observed=data_b)

            # PyMC v5 默认返回 InferenceData，取后验样本需通过 .posterior
            idata = pm.sample(
                self.mcmc_draws,
                tune=self.mcmc_tune,
                target_accept=self.target_accept,
                progressbar=True,
            )

        # 合并所有链的样本，形状 (chains, draws) → flatten
        samples_a = idata.posterior["mu_a"].values.flatten()
        samples_b = idata.posterior["mu_b"].values.flatten()

        return compute_bayesian_metrics(samples_a, samples_b, self.mde)
