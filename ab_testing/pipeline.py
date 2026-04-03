"""
ABTestPipeline: unified entry point that routes to the correct analyzer
and aggregates results.

Supports:
  - metric_type: "binary" (conversion/retention) or "continuous" (revenue/GMV)
  - method:      "frequentist", "bayesian", or "both" (default)
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
    detected_model: Optional[str] = None   # 'lognormal' / 'student_t' (continuous only)
    skewness: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "=" * 56,
            f"  A/B Test Summary  |  Metric type: {self.metric_type}",
            "=" * 56,
        ]

        if self.frequentist:
            f = self.frequentist
            stat_label = "z-statistic" if self.metric_type == "binary" else "t-statistic"
            lines += [
                "",
                "[Frequentist]",
                f"  Group A mean : {f.mean_a:.4f}",
                f"  Group B mean : {f.mean_b:.4f}",
                f"  delta        : {f.delta:+.4f}  (B - A)",
                f"  95% CI       : [{f.ci[0]:+.4f}, {f.ci[1]:+.4f}]",
                f"  {stat_label:<13}: {f.statistic:.4f}",
                f"  p-value      : {f.p_value:.4f}",
                f"  effect size  : {f.effect_size:.4f}",
                f"  → Decision   : {self.decision_freq}",
            ]

        if self.bayesian:
            b = self.bayesian
            model_label = ""
            if self.detected_model:
                model_label = (
                    f"  Model: LogNormal (skewness {self.skewness:.2f}, log-transformed)"
                    if self.detected_model == "lognormal"
                    else f"  Model: StudentT (skewness {self.skewness:.2f}, original scale)"
                )
            lines += [
                "",
                "[Bayesian]",
                *([model_label] if model_label else []),
                f"  Group A posterior mean : {b.mean_a:.4f}",
                f"  Group B posterior mean : {b.mean_b:.4f}",
                f"  delta posterior mean   : {b.delta_mean:+.4f}",
                f"  P(B > A)               : {b.prob_b_better:.1%}",
                f"  P(delta > MDE)         : {b.prob_practical:.1%}",
                f"  Expected loss (Keep A) : {b.expected_loss_a:.5f}",
                f"  Expected loss (Launch B) : {b.expected_loss_b:.5f}",
                f"  → Decision             : {self.decision_bayes}",
            ]

        lines.append("=" * 56)
        return "\n".join(lines)

    def print_summary(self):
        print(self.summary())


class ABTestPipeline:
    """
    Main A/B test pipeline.

    Args:
        metric_type:       "binary" or "continuous"
        method:            "frequentist", "bayesian", or "both"
        alpha:             Frequentist significance level (default 0.05)
        mde:               Minimum detectable effect (same unit as metric)
        loss_threshold:    Bayesian expected-loss stopping threshold
        prior_strength:    Bayesian prior strength (equivalent historical sample size)
        historical_rate:   Binary prior: historical conversion rate (binary only)
        historical_mean:   Continuous prior: historical mean (continuous only)
        historical_std:    Continuous prior: historical std dev (continuous only)
        n_samples:         Bayesian posterior Monte Carlo samples (binary only)
        mcmc_draws:        MCMC draws (continuous only)

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
        max_mcmc_samples: int = 800,
        n_samples: int = 200_000,
        mcmc_draws: int = 1000,
        mcmc_tune: int = 500,
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
        self.max_mcmc_samples = max_mcmc_samples
        self.n_samples = n_samples
        self.mcmc_draws = mcmc_draws
        self.mcmc_tune = mcmc_tune

    def run(self, data_a: np.ndarray, data_b: np.ndarray) -> ABTestResult:
        """
        Run the A/B test analysis.

        Args:
            data_a: Group A (control) data
            data_b: Group B (treatment) data

        Returns:
            ABTestResult containing all metrics and decision recommendations
        """
        data_a = np.asarray(data_a, dtype=float)
        data_b = np.asarray(data_b, dtype=float)

        # ── Validation ────────────────────────────────────────────
        if len(data_a) == 0 or len(data_b) == 0:
            raise ValueError("Group A or B is empty. Check your group column and labels.")
        if np.any(np.isnan(data_a)) or np.any(np.isnan(data_b)):
            raise ValueError("Data contains NaN values. Please clean the data first.")
        if np.any(np.isinf(data_a)) or np.any(np.isinf(data_b)):
            raise ValueError("Data contains Inf values. Please clean the data first.")
        if self.metric_type == "binary":
            unique_a = set(np.unique(data_a))
            unique_b = set(np.unique(data_b))
            if not unique_a.issubset({0.0, 1.0}) or not unique_b.issubset({0.0, 1.0}):
                raise ValueError(
                    "Binary metric data must contain only 0 and 1. "
                    "Check that you selected the correct metric column."
                )

        result = ABTestResult(metric_type=self.metric_type, method=self.method)

        # ── Frequentist ───────────────────────────────────────────
        if self.method in ("frequentist", "both"):
            if self.metric_type == "binary":
                freq = two_proportion_ztest(data_a, data_b, alpha=self.alpha)
            else:
                freq = welch_ttest(data_a, data_b, alpha=self.alpha)
            result.frequentist = freq
            result.decision_freq = frequentist_decision(freq)

        # ── Bayesian ──────────────────────────────────────────────
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
                    max_mcmc_samples=self.max_mcmc_samples,
                    mcmc_draws=self.mcmc_draws,
                    mcmc_tune=self.mcmc_tune,
                    mde=self.mde,
                )
            bayes = model.fit(data_a, data_b)
            result.bayesian = bayes
            result.decision_bayes = bayesian_decision(bayes, self.loss_threshold)
            if self.metric_type == "continuous":
                result.detected_model = model.detected_model
                result.skewness       = model.skewness

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
        Load data from a CSV file and run the analysis.

        Args:
            filepath:         Path to the CSV file
            group_col:        Column name for group labels (e.g. 'version')
            metric_col:       Column name for the metric (e.g. 'retention_1' or 'revenue')
            control_label:    Label for group A (control)
            treatment_label:  Label for group B (treatment)
        """
        df = pd.read_csv(filepath)

        data_a = df[df[group_col] == control_label][metric_col].values.astype(float)
        data_b = df[df[group_col] == treatment_label][metric_col].values.astype(float)

        print(f"Data loaded: Group A {len(data_a)} rows, Group B {len(data_b)} rows")
        print(f"Group A mean: {data_a.mean():.4f}  |  Group B mean: {data_b.mean():.4f}")
        print()

        return self.run(data_a, data_b)

    def plot(
        self,
        result: ABTestResult,
        metric_label: str = "Metric",
        save_dir: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot analysis results.

        Args:
            result:       ABTestResult object
            metric_label: Display name for the metric in charts
            save_dir:     Directory to save plots (None = don't save)
            show:         Whether to open an interactive window
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
