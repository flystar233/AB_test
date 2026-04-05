"""
ABTestPipeline: unified entry point that routes to the correct analyzer
and aggregates results.

Supports:
  - metric_type: "binary" (conversion/retention) or "continuous" (revenue/GMV)
  - method:      "frequentist", "bayesian", "both" (default), or "sequential"
  - sequential:  Group sequential testing with alpha spending (O'Brien-Fleming, Pocock)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, List, Dict, Any
import textwrap

import numpy as np
import pandas as pd

from .metrics import FrequentistResult, BayesianResult, bayesian_decision, frequentist_decision
from .frequentist import two_proportion_ztest, welch_ttest
from .bayesian_binary import BayesianBinary
from .bayesian_continuous import BayesianContinuous
from .sequential import SequentialTest, SequentialResult


@dataclass
class ABTestResult:
    metric_type: str
    method: str
    frequentist: Optional[FrequentistResult] = None
    bayesian: Optional[BayesianResult] = None
    sequential: Optional[SequentialResult] = None
    decision_freq: Optional[str] = None
    decision_bayes: Optional[str] = None
    decision_seq: Optional[str] = None
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

        if self.sequential:
            s = self.sequential
            lines += [
                "",
                "[Sequential]",
                f"  Method       : {s.method}",
                f"  Total looks  : {len(s.looks)}",
                f"  Current look : {s.current_look}",
                f"  → Decision   : {self.decision_seq}",
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
        method:            "frequentist", "bayesian", "both", or "sequential"
        alpha:             Frequentist significance level (default 0.05)
        mde:               Minimum detectable effect (same unit as metric)
        loss_threshold:    Bayesian expected-loss stopping threshold
        prior_strength:    Bayesian prior strength (equivalent historical sample size)
        historical_rate:   Binary prior: historical conversion rate (binary only)
        historical_mean:   Continuous prior: historical mean (continuous only)
        historical_std:    Continuous prior: historical std dev (continuous only)
        n_samples:         Bayesian posterior Monte Carlo samples (binary only)
        mcmc_draws:        MCMC draws (continuous only)
        sequential_method: Sequential testing method: "obrien_fleming", "pocock", "wang_tsiatis"
        sequential_looks:  Maximum number of looks for sequential testing
        sequential_n:      Expected total sample size for sequential testing

    Examples:
        >>> pipeline = ABTestPipeline(metric_type="binary", method="both",
        ...                           historical_rate=0.44, mde=0.005)
        >>> result = pipeline.run(data_a, data_b)
        >>> result.print_summary()
        >>> pipeline.plot(result)

        >>> # Sequential testing
        >>> seq_pipeline = ABTestPipeline(
        ...     metric_type="binary",
        ...     method="sequential",
        ...     sequential_method="obrien_fleming",
        ...     sequential_looks=5,
        ... )
        >>> # Look 1
        >>> result1 = seq_pipeline.run_sequential(data_a[:2000], data_b[:2000], look=1)
        >>> # Look 2
        >>> result2 = seq_pipeline.run_sequential(data_a[:4000], data_b[:4000], look=2)
    """

    def __init__(
        self,
        metric_type: Literal["binary", "continuous"] = "binary",
        method: Literal["frequentist", "bayesian", "both", "sequential"] = "both",
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
        sequential_method: Literal["obrien_fleming", "pocock", "wang_tsiatis"] = "obrien_fleming",
        sequential_looks: int = 5,
        sequential_n: Optional[int] = None,
        sequential_wang_delta: float = 0.5,
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

        # Sequential testing parameters
        self.sequential_method = sequential_method
        self.sequential_looks = sequential_looks
        self.sequential_n = sequential_n
        self.sequential_wang_delta = sequential_wang_delta
        self._sequential_test: Optional[SequentialTest] = None

    def _validate_data(self, data_a: np.ndarray, data_b: np.ndarray) -> None:
        """Validate input data."""
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

        self._validate_data(data_a, data_b)
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

        # ── Sequential (single look for backward compatibility) ────
        if self.method == "sequential":
            # For single look, just initialize but don't run
            pass

        return result

    def init_sequential(self) -> "ABTestPipeline":
        """
        Initialize or reset the sequential test.

        Call this before starting a new sequential test with multiple looks.
        """
        self._sequential_test = SequentialTest(
            method=self.sequential_method,
            metric_type=self.metric_type,
            alpha=self.alpha,
            max_looks=self.sequential_looks,
            expected_total_n=self.sequential_n,
            wang_tsiatis_delta=self.sequential_wang_delta,
            two_sided=True,
        )
        return self

    def run_sequential(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        look: Optional[int] = None,
    ) -> ABTestResult:
        """
        Run sequential test with accumulating data.

        Call this method repeatedly with more data each time (e.g., first 20%,
        then 40%, etc.). The pipeline automatically tracks the state between calls.

        Args:
            data_a: Group A data (cumulative up to this look)
            data_b: Group B data (cumulative up to this look)
            look: Optional look index (1-based). If None, auto-increments.

        Returns:
            ABTestResult with sequential test results

        Example:
            >>> pipeline = ABTestPipeline(method="sequential", sequential_looks=5)
            >>> pipeline.init_sequential()
            >>> # Look 1 with 20% data
            >>> result1 = pipeline.run_sequential(data_a[:2000], data_b[:2000])
            >>> # Look 2 with 40% data
            >>> result2 = pipeline.run_sequential(data_a[:4000], data_b[:4000])
            >>> # Check final result
            >>> print(result2.decision_seq)
        """
        data_a = np.asarray(data_a, dtype=float)
        data_b = np.asarray(data_b, dtype=float)

        self._validate_data(data_a, data_b)

        # Initialize sequential test if not already initialized
        if self._sequential_test is None:
            self.init_sequential()

        # Add this look
        decision = self._sequential_test.add_look(data_a, data_b)

        # Get result
        seq_result = self._sequential_test.get_result()

        # Create ABTestResult
        result = ABTestResult(metric_type=self.metric_type, method="sequential")
        result.sequential = seq_result
        result.decision_seq = decision

        # Also run frequentist for comparison
        if self.metric_type == "binary":
            freq = two_proportion_ztest(data_a, data_b, alpha=self.alpha)
        else:
            freq = welch_ttest(data_a, data_b, alpha=self.alpha)
        result.frequentist = freq
        result.decision_freq = frequentist_decision(freq)

        return result

    def get_sequential_result(self) -> Optional[ABTestResult]:
        """Get the current sequential test result without adding new data."""
        if self._sequential_test is None or not self._sequential_test.looks:
            return None

        seq_result = self._sequential_test.get_result()
        result = ABTestResult(metric_type=self.metric_type, method="sequential")
        result.sequential = seq_result
        result.decision_seq = seq_result.final_decision
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
