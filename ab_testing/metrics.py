"""
Decision metrics: pure functions that take posterior samples / statistics
and return the three core decision metrics.
Decoupled from specific models; shared by both Bayesian and Frequentist paths.
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class FrequentistResult:
    statistic: float                     # z-score or t-statistic
    p_value: float
    ci: Tuple[float, float]              # confidence interval (lower, upper)
    significant: bool
    effect_size: float                   # Cohen's h (binary) or Cohen's d (continuous)
    mean_a: float
    mean_b: float
    delta: float                         # mean_b - mean_a


@dataclass
class BayesianResult:
    prob_b_better: float                 # P(B > A)
    prob_practical: float                # P(delta > MDE)
    expected_loss_a: float               # expected loss of choosing A
    expected_loss_b: float               # expected loss of choosing B
    posterior_a: np.ndarray              # posterior samples (mu or p)
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
    Compute all Bayesian decision metrics from posterior samples.

    Args:
        samples_a: Posterior parameter samples for group A (conversion rate p or mean mu)
        samples_b: Posterior parameter samples for group B
        mde:       Minimum detectable effect (same unit as the metric)
    """
    delta = samples_b - samples_a

    prob_b_better  = float(np.mean(delta > 0))
    prob_practical = float(np.mean(delta > mde))
    expected_loss_a = float(np.mean(np.maximum(delta, 0)))   # opportunity cost of choosing A
    expected_loss_b = float(np.mean(np.maximum(-delta, 0)))  # opportunity cost of choosing B

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
    Stopping decision based on expected loss.

    expected_loss_a = E[max(B-A, 0)] = opportunity cost of choosing A
    expected_loss_b = E[max(A-B, 0)] = opportunity cost of choosing B

    - Low loss for A → B is unlikely to be better → safe to Keep A
    - Low loss for B → A is unlikely to be better → safe to Ship B
    """
    if result.expected_loss_a < loss_threshold:
        return "Keep A"
    elif result.expected_loss_b < loss_threshold:
        return "Launch B"
    else:
        return "Collect More Data"


def frequentist_decision(result: FrequentistResult) -> str:
    """Stopping decision based on statistical significance."""
    if not result.significant:
        return "Keep A (No Significant Difference)"
    return "Launch B" if result.delta > 0 else "Keep A (B is Worse)"
