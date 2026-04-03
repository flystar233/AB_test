、"""
Bayesian binary metrics: Beta-Bernoulli conjugate model
(conversion rate, retention rate, click-through rate).

Prior:      p ~ Beta(alpha, beta)
Likelihood: x_i ~ Bernoulli(p)
Posterior:  p | data ~ Beta(alpha + successes, beta + failures)

No MCMC required — closed-form analytical solution.
"""
import numpy as np

from .metrics import BayesianResult, compute_bayesian_metrics


class BayesianBinary:
    """
    Args:
        historical_rate:  Historical conversion rate, sets the prior mean (default 0.5 = uninformative)
        prior_strength:   Equivalent historical sample size; higher = stronger prior (default 2 ≈ uninformative)
        n_samples:        Number of posterior Monte Carlo samples
        mde:              Minimum detectable effect (absolute, same unit as conversion rate)
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

        # Prior parameters (clipped to [0.01, 0.99] to keep Beta valid)
        clipped_rate = float(np.clip(historical_rate, 0.01, 0.99))
        self.prior_alpha = clipped_rate * prior_strength
        self.prior_beta = (1 - clipped_rate) * prior_strength

    def fit(self, data_a: np.ndarray, data_b: np.ndarray) -> BayesianResult:
        """
        Update posteriors and compute decision metrics.

        Args:
            data_a: Group A binary array (0/1)
            data_b: Group B binary array (0/1)
        """
        # Conjugate posterior update
        post_alpha_a = self.prior_alpha + data_a.sum()
        post_beta_a  = self.prior_beta  + (len(data_a) - data_a.sum())

        post_alpha_b = self.prior_alpha + data_b.sum()
        post_beta_b  = self.prior_beta  + (len(data_b) - data_b.sum())

        # Posterior means
        self.posterior_mean_a = post_alpha_a / (post_alpha_a + post_beta_a)
        self.posterior_mean_b = post_alpha_b / (post_alpha_b + post_beta_b)

        # Posterior sampling
        rng = np.random.default_rng()
        samples_a = rng.beta(post_alpha_a, post_beta_a, self.n_samples)
        samples_b = rng.beta(post_alpha_b, post_beta_b, self.n_samples)

        return compute_bayesian_metrics(samples_a, samples_b, self.mde)
