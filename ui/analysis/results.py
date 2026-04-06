"""Results display: SRM check, decision banners, KPI cards, and charts."""
import streamlit as st
from scipy.stats import chisquare
from streamlit_echarts import st_echarts

from ab_testing.visualizer_echarts import (
    delta_chart,
    freq_chart,
    loss_chart,
    posterior_chart,
    sequential_chart,
    sequential_metrics_chart,
)
from ui.components import kpi_row


# ── SRM check ─────────────────────────────────────────────────────────────────

def _check_srm(n_a: int, n_b: int, expected_ratio: float = 0.5) -> float:
    total = n_a + n_b
    _, p_value = chisquare(
        [n_a, n_b],
        f_exp=[total * expected_ratio, total * (1 - expected_ratio)],
    )
    return float(p_value)


def _render_srm_banner(n_a: int, n_b: int) -> None:
    srm_p   = _check_srm(n_a, n_b)
    ratio_a = n_a / (n_a + n_b)
    abs_dev = abs(ratio_a - 0.5)
    if srm_p < 0.01 and abs_dev > 0.02:
        st.error(
            f"**Sample Ratio Mismatch (SRM) Detected**  —  p = {srm_p:.4f}, deviation {abs_dev:.1%}\n\n"
            f"Expected a 50 / 50 split, but observed **{n_a:,}** (A) vs **{n_b:,}** (B)  "
            f"({ratio_a:.1%} / {1 - ratio_a:.1%}).  \n"
            "A significant imbalance usually indicates a bug in the randomization or data pipeline "
            "(e.g. cookie dropping, bot filtering applied to only one arm, or a logging gap).  \n"
            "**The analysis results below may be biased — investigate the root cause before drawing conclusions.**"
        )
    else:
        st.success(
            f"**No SRM detected**  —  p = {srm_p:.4f}, deviation {abs_dev:.1%}  "
            f"({n_a:,} A  /  {n_b:,} B)"
        )


# ── Decision banners ──────────────────────────────────────────────────────────

def _decision_banner(decision: str, method_name: str) -> None:
    if "Launch B" in decision or "Reject H0" in decision:
        st.success(f"**{method_name} Decision: {decision}**")
    elif "Keep A" in decision or "Accept H0" in decision:
        st.info(f"**{method_name} Decision: {decision}**")
    else:
        st.warning(f"**{method_name} Decision: {decision}**")


# ── Chart sections ────────────────────────────────────────────────────────────

def _render_frequentist_section(result, pipeline, m_label: str) -> None:
    st.markdown("#### Frequentist: Mean Comparison & Confidence Interval")
    st_echarts(
        options=freq_chart(result.frequentist, metric_label=m_label, mde=pipeline.mde),
        height="380px",
        key="chart_freq",
    )


def _render_bayesian_section(result, pipeline, m_label: str) -> None:
    st.markdown("#### Bayesian: Posterior Distribution & Decision Metrics")
    b = result.bayesian

    if result.detected_model == "lognormal":
        st.info(
            f"**Model auto-selected: log1p transform + Normal** (skewness {result.skewness:.2f} > 1.0)  \n"
            "Models log(1+x); zero-safe. Posterior mean corresponds to the geometric mean on the original scale, "
            "more robust to high-value outliers."
        )
    elif result.detected_model == "student_t":
        st.info(
            f"**Model auto-selected: StudentT** (skewness {result.skewness:.2f} ≤ 1.0 or negatives present)  \n"
            "Models directly on the original scale. Posterior mean is a robust estimate of the arithmetic mean."
        )

    tab1, tab2, tab3 = st.tabs(["Posterior Distribution", "Delta Distribution", "Expected Loss"])
    with tab1:
        st.caption(f"Group A posterior mean {b.mean_a:.4f}  |  Group B posterior mean {b.mean_b:.4f}")
        st_echarts(options=posterior_chart(b, metric_label=m_label), height="400px", key="chart_posterior")
    with tab2:
        st.caption(
            f"P(B > A) = {b.prob_b_better:.1%}  |  "
            f"P(delta > MDE={pipeline.mde}) = {b.prob_practical:.1%}"
        )
        st_echarts(options=delta_chart(b, mde=pipeline.mde, metric_label=m_label), height="400px", key="chart_delta")
    with tab3:
        st.caption(
            f"Expected loss (Keep A): {b.expected_loss_a:.6f}  |  "
            f"Expected loss (Ship B): {b.expected_loss_b:.6f}  |  "
            f"Threshold: {pipeline.loss_threshold}"
        )
        st_echarts(options=loss_chart(b, loss_threshold=pipeline.loss_threshold), height="380px", key="chart_loss")


def _render_sequential_section(result) -> None:
    s = result.sequential
    st.markdown("#### Sequential: Test Boundaries & Metrics Over Time")

    if s.looks:
        last = s.looks[-1]
        kpi_row([
            ("Current Look",   f"{s.current_look} / {len(s.boundary_values)}"),
            ("N (A)",          f"{last.n_a:,}"),
            ("N (B)",          f"{last.n_b:,}"),
            ("Test Statistic", f"{last.statistic:+.4f}"),
            ("Naive p-value",  f"{last.p_value:.4f}" if last.p_value else "-"),
        ])

    tab1, tab2 = st.tabs(["Boundary Chart", "Metrics Evolution"])
    with tab1:
        st.caption(
            "Test statistic path vs. rejection boundaries. "
            "Crossing a boundary indicates a statistically significant result."
        )
        st_echarts(options=sequential_chart(s), height="420px", key="chart_seq_boundary")
    with tab2:
        st_echarts(options=sequential_metrics_chart(s), height="400px", key="chart_seq_metrics")

    with st.expander("View Sequential Test Details"):
        st.code(s.summary(), language=None)


# ── Main render function ──────────────────────────────────────────────────────

def render_results(params: dict) -> None:
    """Render the full results section if a result is available in session state."""
    if "result" not in st.session_state:
        return

    result   = st.session_state["result"]
    pipeline = st.session_state["pipeline"]
    m_label  = st.session_state.get("metric_col", "Metric")
    meta     = st.session_state.get("analysis_meta", {})

    st.divider()
    st.subheader("3. Results")

    # SRM check
    n_a = meta.get("n_control",   0)
    n_b = meta.get("n_treatment", 0)
    if n_a and n_b:
        _render_srm_banner(n_a, n_b)

    # Decision banners
    if result.frequentist and result.decision_freq:
        _decision_banner(result.decision_freq, "[Frequentist]")
    if result.bayesian and result.decision_bayes:
        _decision_banner(result.decision_bayes, "[Bayesian]")
    if result.sequential and result.decision_seq:
        _decision_banner(result.decision_seq, "[Sequential]")

    # KPI summary cards
    st.markdown("#### Key Metrics")
    if result.frequentist:
        f = result.frequentist
        sig_color = "#27ae60" if f.significant else "#888"
        mde_color = "#27ae60" if f.delta >= pipeline.mde else "#888"
        st.caption("**Frequentist**")
        kpi_row([
            ("Group A Mean",         f"{f.mean_a:.4f}"),
            ("Group B Mean",         f"{f.mean_b:.4f}"),
            ("p-value",              f"{f.p_value:.4f}",
             "Significant ✓" if f.significant else "Not Significant ✗", sig_color),
            ("Effect Size",          f"{f.effect_size:.4f}"),
            ("delta vs MDE",         f"{f.delta:+.4f} vs {pipeline.mde:.4f}",
             "Reached MDE ✓" if f.delta >= pipeline.mde else "Below MDE ✗", mde_color),
            ("95% CI",               f"[{f.ci[0]:+.4f}, {f.ci[1]:+.4f}]"),
        ])

    if result.bayesian:
        b = result.bayesian
        if result.frequentist:
            st.divider()
        st.caption("**Bayesian**")
        kpi_row([
            ("A Posterior Mean",       f"{b.mean_a:.4f}"),
            ("B Posterior Mean",       f"{b.mean_b:.4f}"),
            ("P(B > A)",               f"{b.prob_b_better:.1%}"),
            ("P(delta > MDE)",         f"{b.prob_practical:.1%}"),
            ("Exp. Loss (Keep A)",     f"{b.expected_loss_a:.5f}"),
            ("Exp. Loss (Launch B)",   f"{b.expected_loss_b:.5f}"),
        ])

    st.markdown("---")

    # Charts
    if result.frequentist and not result.sequential:
        _render_frequentist_section(result, pipeline, m_label)

    if result.bayesian and not result.sequential:
        _render_bayesian_section(result, pipeline, m_label)

    if result.sequential:
        _render_sequential_section(result)

    with st.expander("View Full Text Summary"):
        st.code(result.summary(), language=None)
