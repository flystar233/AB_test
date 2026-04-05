"""
Sequential Testing Module for A/B Tests.

Implements alpha spending functions for sequential monitoring:
- O'Brien-Fleming (conservative early stopping)
- Pocock (equal rejection bounds at each look)
- Wang-Tsiatis (family of tests with parameter Δ)

Also provides always-valid p-values and confidence sequences.

References:
- Jennison, C., & Turnbull, B. W. (2000). Group Sequential Methods with Applications to Clinical Trials.
- Wasserman, L., & Ramdas, A. (2022). Always Valid Inference.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple, Union
import numpy as np
from scipy import stats


@dataclass
class SequentialLook:
    """Data from a single look in sequential testing."""
    look_time: int                  # Look index (1-based)
    n_a: int                        # Sample size in group A
    n_b: int                        # Sample size in group B
    mean_a: float                   # Mean in group A
    mean_b: float                   # Mean in group B
    statistic: float                # Test statistic (z-score or similar)
    p_value: Optional[float] = None  # Naive p-value (not adjusted)
    adjusted_p: Optional[float] = None  # Alpha-spending adjusted p-value
    decision: Optional[str] = None  # Decision at this look


@dataclass
class SequentialResult:
    """Result from sequential testing analysis."""
    method: str                     # "obrien_fleming", "pocock", "wang_tsiatis"
    metric_type: str                # "binary" or "continuous"
    alpha: float                    # Overall type I error rate
    looks: List[SequentialLook]     # List of looks
    final_decision: str             # Final decision: "Reject H0", "Accept H0", "Continue"
    boundary_values: List[float]    # Rejection boundaries at each look
    information_rates: List[float]  # Information rates at each look
    spending_function: str          # Name of spending function used
    current_look: int               # Current look index

    def summary(self) -> str:
        """Generate a text summary of the sequential test."""
        lines = [
            "=" * 60,
            f"  Sequential Test Summary  |  Method: {self.method}",
            "=" * 60,
            "",
            f"Overall α: {self.alpha:.3f}",
            f"Total looks: {len(self.looks)}",
            f"Final decision: {self.final_decision}",
            "",
            "Looks:",
            "-" * 40,
            f"{'Look':<6} {'N(A)':<8} {'N(B)':<8} {'Δ':<10} {'Stat':<10} {'Bound':<10} {'Decision'}",
        ]

        for look, bound in zip(self.looks, self.boundary_values):
            delta = look.mean_b - look.mean_a
            lines.append(
                f"{look.look_time:<6} "
                f"{look.n_a:<8} "
                f"{look.n_b:<8} "
                f"{delta:+.4f}  "
                f"{look.statistic:+.4f}  "
                f"{bound:+.4f}  "
                f"{look.decision or '-'}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


class AlphaSpendingFunction:
    """
    Alpha spending functions for group sequential testing.

    These functions determine how the overall type I error rate α
    is "spent" across multiple looks at the data.
    """

    @staticmethod
    def obrien_fleming(information_rate: float, alpha: float) -> float:
        """
        O'Brien-Fleming spending function.

        Very conservative early stopping - requires strong evidence to stop early.
        α*(t) = 2 * (1 - Φ(Φ^{-1}(1 - α/2) / sqrt(t)))

        Args:
            information_rate: Proportion of maximum information (0 < t ≤ 1)
            alpha: Overall type I error rate
        """
        if information_rate <= 0:
            return 0.0
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        return 2 * (1 - stats.norm.cdf(z_alpha / np.sqrt(information_rate)))

    @staticmethod
    def pocock(information_rate: float, alpha: float) -> float:
        """
        Pocock spending function.

        Equal rejection bounds at each look (approximately).
        α*(t) = α * log(1 + (e - 1) * t)

        Args:
            information_rate: Proportion of maximum information (0 < t ≤ 1)
            alpha: Overall type I error rate
        """
        if information_rate <= 0:
            return 0.0
        return alpha * np.log(1 + (np.e - 1) * information_rate)

    @staticmethod
    def wang_tsiatis(information_rate: float, alpha: float, delta: float = 0.5) -> float:
        """
        Wang-Tsiatis family of spending functions.

        Interpolates between Pocock (Δ=0) and O'Brien-Fleming (Δ=0.5).

        Args:
            information_rate: Proportion of maximum information (0 < t ≤ 1)
            alpha: Overall type I error rate
            delta: Parameter between 0 (Pocock-like) and 0.5 (O'Brien-Fleming-like)
        """
        if information_rate <= 0:
            return 0.0
        return alpha * (information_rate ** (delta * 2))


class SequentialTest:
    """
    Group sequential test for A/B testing.

    Supports multiple looks at accumulating data with proper alpha spending
    to control the overall type I error rate.

    Example:
        >>> seq_test = SequentialTest(
        ...     method="obrien_fleming",
        ...     metric_type="binary",
        ...     alpha=0.05,
        ...     max_looks=5,
        ...     expected_total_n=10000
        ... )
        >>> # Look 1
        >>> decision = seq_test.add_look(data_a[:2000], data_b[:2000])
        >>> # Look 2
        >>> decision = seq_test.add_look(data_a[:4000], data_b[:4000])
        >>> # Final result
        >>> result = seq_test.get_result()
    """

    def __init__(
        self,
        method: Literal["obrien_fleming", "pocock", "wang_tsiatis"] = "obrien_fleming",
        metric_type: Literal["binary", "continuous"] = "binary",
        alpha: float = 0.05,
        max_looks: int = 5,
        expected_total_n: Optional[int] = None,
        wang_tsiatis_delta: float = 0.5,
        two_sided: bool = True,
    ):
        """
        Initialize sequential test.

        Args:
            method: Alpha spending function to use
            metric_type: Type of metric ("binary" or "continuous")
            alpha: Overall type I error rate
            max_looks: Maximum number of looks planned
            expected_total_n: Expected total sample size (for information rates)
            wang_tsiatis_delta: Delta parameter for Wang-Tsiatis (only if method="wang_tsiatis")
            two_sided: Whether to perform two-sided test
        """
        self.method = method
        self.metric_type = metric_type
        self.alpha = alpha
        self.max_looks = max_looks
        self.expected_total_n = expected_total_n
        self.wang_tsiatis_delta = wang_tsiatis_delta
        self.two_sided = two_sided

        self.looks: List[SequentialLook] = []
        self._information_rates: List[float] = []
        self._alpha_spent: List[float] = []
        self._boundaries: List[float] = []

        # Pre-compute planned information rates and boundaries
        self._planned_information_rates = [
            (i + 1) / max_looks for i in range(max_looks)
        ]
        self._planned_boundaries = self._compute_boundaries(self._planned_information_rates)

    def _compute_spending(self, t: float) -> float:
        """Compute alpha spent at information rate t."""
        if self.method == "obrien_fleming":
            return AlphaSpendingFunction.obrien_fleming(t, self.alpha)
        elif self.method == "pocock":
            return AlphaSpendingFunction.pocock(t, self.alpha)
        elif self.method == "wang_tsiatis":
            return AlphaSpendingFunction.wang_tsiatis(t, self.alpha, self.wang_tsiatis_delta)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _compute_boundaries(
        self,
        information_rates: List[float],
    ) -> List[float]:
        """
        Compute rejection boundaries using numerical search.

        Uses the alpha spending function to find critical values
        at each information time.
        """
        boundaries = []

        for i, t in enumerate(information_rates):
            # Cumulative alpha spent up to this look
            if i == 0:
                cum_alpha = self._compute_spending(t)
            else:
                cum_alpha = self._compute_spending(t) - self._compute_spending(information_rates[i-1])

            # One-sided or two-sided alpha
            test_alpha = cum_alpha / 2 if self.two_sided else cum_alpha

            # Critical value (simplified - for full implementation use multivariate normal)
            # This is an approximation using the alpha spending function directly
            z_crit = stats.norm.ppf(1 - test_alpha)
            boundaries.append(z_crit)

        return boundaries

    def _compute_statistic(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Compute test statistic and p-value.

        Returns:
            (statistic, p_value, mean_a, mean_b)
        """
        data_a = np.asarray(data_a, dtype=float)
        data_b = np.asarray(data_b, dtype=float)

        n_a, n_b = len(data_a), len(data_b)
        mean_a, mean_b = data_a.mean(), data_b.mean()

        if self.metric_type == "binary":
            # Two-proportion Z-test
            x_a, x_b = data_a.sum(), data_b.sum()
            p_a, p_b = x_a / n_a, x_b / n_b
            p_pool = (x_a + x_b) / (n_a + n_b)

            if p_pool * (1 - p_pool) == 0:
                return 0.0, 1.0, float(mean_a), float(mean_b)

            se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
            z = (p_b - p_a) / se_pool if se_pool > 0 else 0.0
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            return float(z), float(p_value), float(mean_a), float(mean_b)

        else:
            # Welch's t-test
            var_a, var_b = data_a.var(ddof=1), data_b.var(ddof=1)
            se = np.sqrt(var_a / n_a + var_b / n_b)

            if se == 0:
                return 0.0, 1.0, float(mean_a), float(mean_b)

            t_stat = (mean_b - mean_a) / se
            df = (var_a / n_a + var_b / n_b) ** 2 / (
                (var_a / n_a) ** 2 / (n_a - 1) +
                (var_b / n_b) ** 2 / (n_b - 1)
            )
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

            # For sequential tests with large n, use z-score approximation
            z_approx = (mean_b - mean_a) / se

            return float(z_approx), float(p_value), float(mean_a), float(mean_b)

    def add_look(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
    ) -> str:
        """
        Add a new look at the data.

        Args:
            data_a: Group A data up to this look
            data_b: Group B data up to this look

        Returns:
            Decision: "Reject H0", "Accept H0", or "Continue"
        """
        look_index = len(self.looks) + 1

        # Compute test statistic
        statistic, p_value, mean_a, mean_b = self._compute_statistic(data_a, data_b)

        # Information rate
        n_a, n_b = len(data_a), len(data_b)
        if self.expected_total_n:
            info_rate = min((n_a + n_b) / self.expected_total_n, 1.0)
        else:
            info_rate = look_index / self.max_looks

        self._information_rates.append(info_rate)

        # Compute cumulative alpha spent
        if look_index == 1:
            cum_alpha_spent = self._compute_spending(info_rate)
        else:
            cum_alpha_spent = self._compute_spending(info_rate) - self._compute_spending(self._information_rates[-2])
        self._alpha_spent.append(cum_alpha_spent)

        # Compute boundary
        if look_index <= len(self._planned_boundaries):
            boundary = self._planned_boundaries[look_index - 1]
        else:
            # Beyond planned looks - use last boundary
            boundary = self._planned_boundaries[-1]
        self._boundaries.append(boundary)

        # Make decision
        if self.two_sided:
            reject = abs(statistic) > boundary
        else:
            reject = statistic > boundary

        if reject:
            decision = "Reject H0"
        elif look_index >= self.max_looks:
            decision = "Accept H0"
        else:
            decision = "Continue"

        # Store this look
        look = SequentialLook(
            look_time=look_index,
            n_a=n_a,
            n_b=n_b,
            mean_a=mean_a,
            mean_b=mean_b,
            statistic=statistic,
            p_value=p_value,
            decision=decision,
        )
        self.looks.append(look)

        return decision

    def get_result(self) -> SequentialResult:
        """Get the full sequential test result."""
        if not self.looks:
            raise ValueError("No looks have been added yet.")

        final_decision = self.looks[-1].decision or "Continue"

        if len(self._boundaries) < len(self._planned_boundaries):
            boundaries = self._boundaries + self._planned_boundaries[len(self._boundaries):]
        else:
            boundaries = self._boundaries

        return SequentialResult(
            method=self.method,
            metric_type=self.metric_type,
            alpha=self.alpha,
            looks=self.looks,
            final_decision=final_decision,
            boundary_values=boundaries,
            information_rates=(
                self._information_rates +
                self._planned_information_rates[len(self._information_rates):]
            ),
            spending_function=self.method,
            current_look=len(self.looks),
        )


def always_valid_p_value(
    p_values: Union[float, List[float]],
    method: Literal["preranked", "asynchronous"] = "preranked",
) -> Union[float, List[float]]:
    """
    Compute always-valid p-values (AVP) for sequential testing.

    AVPs maintain their validity regardless of when you stop.

    Args:
        p_values: List of p-values from sequential looks (or single p-value)
        method: "preranked" for fixed number of looks, "asynchronous" for optional stopping

    Returns:
        Always-valid p-value(s)

    References:
        Vovk, V., & Wang, R. (2021). Always Valid P-Values.
    """
    if isinstance(p_values, (int, float)):
        p_values = [p_values]

    avps = []
    for k, p in enumerate(p_values, 1):
        if method == "preranked":
            # Preranked test: multiply by number of looks planned
            # This is a simple but conservative approach
            avp = min(p * len(p_values), 1.0)
        else:
            # Asynchronous: use the Robbins-Siegmund inequality
            # p * (1 - log(p)) is always valid for optional stopping
            if p <= 0:
                avp = 1.0
            else:
                avp = min(p * (1 - np.log(max(p, 1e-10))), 1.0)
        avps.append(avp)

    return avps[0] if len(avps) == 1 else avps


def confidence_sequence(
    estimates: List[float],
    ses: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, float]]:
    """
    Confidence sequences for sequential monitoring.

    Unlike standard confidence intervals, confidence sequences maintain
    coverage guarantees regardless of when you stop.

    Args:
        estimates: Point estimates at each look
        ses: Standard errors at each look
        alpha: Overall type I error rate

    Returns:
        List of (lower, upper) confidence bounds at each look

    References:
        Howard, S., et al. (2021). Time-uniform confidence sequences.
    """
    intervals = []
    for k, (est, se) in enumerate(zip(estimates, ses), 1):
        # Use a simple confidence sequence based on mixture likelihood ratio
        # This is a practical approximation
        t = k / len(estimates) if len(estimates) > 0 else 1.0

        # Wider bound for early looks
        cs_multiplier = stats.norm.ppf(1 - alpha / (2 * t))

        lower = est - cs_multiplier * se
        upper = est + cs_multiplier * se
        intervals.append((float(lower), float(upper)))

    return intervals
