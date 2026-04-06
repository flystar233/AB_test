"""Sample size calculator — pre-experiment planning tool."""
import numpy as np
import streamlit as st
from scipy.stats import norm as _norm


def render_sample_calculator() -> None:
    """Render the sample size calculator inside a collapsed expander."""
    with st.expander("Sample Size Calculator — run this before your experiment", expanded=False):
        st.caption("Calculate how many samples you need to detect an effect with sufficient power.")

        c1, c2, c3 = st.columns(3)
        with c1:
            calc_alpha = st.number_input(
                "Significance level α", value=0.05,
                min_value=0.01, max_value=0.20, step=0.01, format="%.2f",
                key="_calc_alpha",
            )
        with c2:
            calc_power = st.number_input(
                "Power (1 − β)", value=0.80,
                min_value=0.50, max_value=0.99, step=0.05, format="%.2f",
                key="_calc_power",
            )
        with c3:
            calc_type = st.radio("Metric type", ["Binary", "Continuous"], horizontal=True, key="_calc_type")

        z_a = _norm.ppf(1 - calc_alpha / 2)
        z_b = _norm.ppf(calc_power)

        if calc_type == "Binary":
            c4, c5 = st.columns(2)
            with c4:
                calc_p = st.number_input(
                    "Baseline conversion rate", value=0.10,
                    min_value=0.001, max_value=0.999, step=0.01, format="%.3f",
                    key="_calc_p",
                )
            with c5:
                calc_mde = st.number_input(
                    "MDE (absolute, e.g. 0.01 = +1pp)", value=0.01,
                    min_value=0.001, step=0.005, format="%.4f",
                    key="_calc_mde_bin",
                )

            p_trt  = calc_p + calc_mde
            p_pool = (calc_p + p_trt) / 2
            n = int(np.ceil(
                ((z_a * (2 * p_pool * (1 - p_pool)) ** 0.5
                  + z_b * (calc_p * (1 - calc_p) + p_trt * (1 - p_trt)) ** 0.5)
                 / calc_mde) ** 2
            ))
            st.metric("Required samples per group", f"{n:,}", help="Total experiment size = 2×")
            st.info(
                f"Total experiment size: **{2 * n:,}** users  \n"
                f"Detecting {calc_p:.1%} → {p_trt:.1%} (Δ = +{calc_mde:.3f}) "
                f"at α={calc_alpha}, power={calc_power:.0%}"
            )

        else:  # Continuous
            c4, c5, c6 = st.columns(3)
            with c4:
                calc_mean = st.number_input("Baseline mean", value=50.0, key="_calc_mean")
            with c5:
                calc_std = st.number_input("Standard deviation", value=20.0, min_value=0.01, key="_calc_std")
            with c6:
                calc_mde_c = st.number_input(
                    "MDE (absolute)", value=2.0,
                    min_value=0.001, step=0.5, format="%.2f",
                    key="_calc_mde_con",
                )

            n = int(np.ceil(2 * ((z_a + z_b) * calc_std / calc_mde_c) ** 2))
            rel_lift = calc_mde_c / calc_mean * 100 if calc_mean != 0 else 0
            st.metric("Required samples per group", f"{n:,}", help="Total experiment size = 2×")
            st.info(
                f"Total experiment size: **{2 * n:,}** users  \n"
                f"Detecting mean shift of **{calc_mde_c:+.2f}** ({rel_lift:+.1f}%) "
                f"with σ={calc_std:.1f}, α={calc_alpha}, power={calc_power:.0%}"
            )
