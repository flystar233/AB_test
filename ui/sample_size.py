"""Standalone Sample Size Calculator page."""
import numpy as np
import streamlit as st
from scipy.stats import norm as _norm

from ui.components import page_header, section_label


def show_sample_size_page() -> None:
    if st.button("← Back to Analysis"):
        st.session_state["page"] = "Analysis"
        st.rerun()

    page_header(
        "Sample Size Calculator",
        "Plan your experiment before you run it — calculate how many samples you need.",
    )

    tab_binary, tab_continuous = st.tabs(["Binary Metric", "Continuous Metric"])

    # ── Binary ────────────────────────────────────────────────────────────────
    with tab_binary:
        st.markdown(
            '<p style="font-family:Inter,sans-serif;font-size:0.9rem;color:#5e5d59;'
            'margin-bottom:1.25rem;">'
            'Typical use cases: conversion rate, click-through rate, retention, signup rate</p>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            alpha = st.number_input(
                "Significance level α",
                value=0.05, min_value=0.01, max_value=0.20, step=0.01, format="%.2f",
                key="bin_alpha",
                help="Probability of a false positive (Type I error). Common choice: 0.05",
            )
        with c2:
            power = st.number_input(
                "Power (1 − β)",
                value=0.80, min_value=0.50, max_value=0.99, step=0.05, format="%.2f",
                key="bin_power",
                help="Probability of detecting a true effect (1 − Type II error). Common choice: 0.80",
            )
        with c3:
            tails = st.radio(
                "Test type",
                ["Two-tailed", "One-tailed"],
                horizontal=True,
                key="bin_tails",
                help="Two-tailed is the standard conservative choice",
            )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        with c4:
            baseline_pct = st.number_input(
                "Baseline conversion rate (%)",
                value=10.0, min_value=0.1, max_value=99.9, step=1.0, format="%.1f",
                key="bin_baseline",
                help="Current conversion rate of the control group",
            )
            baseline = baseline_pct / 100.0
        with c5:
            mde_pct = st.number_input(
                "MDE — Minimum Detectable Effect (absolute, % points)",
                value=1.0, min_value=0.1, step=0.5, format="%.2f",
                key="bin_mde",
                help="Smallest absolute change worth detecting, e.g. 1 = +1 percentage point",
            )
            mde = mde_pct / 100.0

        # Calculation
        alpha_adj = alpha / 2 if tails == "Two-tailed" else alpha
        z_a = _norm.ppf(1 - alpha_adj)
        z_b = _norm.ppf(power)
        p_trt  = baseline + mde
        p_pool = (baseline + p_trt) / 2
        n = int(np.ceil(
            ((z_a * (2 * p_pool * (1 - p_pool)) ** 0.5
              + z_b * (baseline * (1 - baseline) + p_trt * (1 - p_trt)) ** 0.5)
             / mde) ** 2
        ))

        st.divider()

        # KPI result cards
        from ui.components import kpi_row
        kpi_row([
            ("Per group", f"{n:,}"),
            ("Total experiment size", f"{2 * n:,}"),
            ("Baseline → Target", f"{baseline:.1%} → {p_trt:.1%}"),
            ("Absolute MDE", f"{mde:+.2%}"),
        ])

        st.info(
            f"Detecting **{baseline:.1%} → {p_trt:.1%}** (Δ = {mde:+.2%})  \n"
            f"at α = {alpha} ({tails.lower()}), power = {power:.0%}  \n"
            f"→ **{n:,} samples per group**, {2 * n:,} total"
        )

        # MDE sensitivity table
        with st.expander("Sensitivity: required N at different MDEs"):
            mde_values = [mde * m for m in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]]
            rows = []
            for m in mde_values:
                p2 = baseline + m
                if not (0 < p2 < 1):
                    continue
                pp = (baseline + p2) / 2
                n_m = int(np.ceil(
                    ((z_a * (2 * pp * (1 - pp)) ** 0.5
                      + z_b * (baseline * (1 - baseline) + p2 * (1 - p2)) ** 0.5)
                     / m) ** 2
                ))
                rows.append({
                    "MDE (abs, % pts)": f"{m*100:+.2f}",
                    "MDE (relative)": f"{m / baseline:+.1%}",
                    "Target rate":    f"{p2:.3%}",
                    "N per group":    f"{n_m:,}",
                    "N total":        f"{2 * n_m:,}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    # ── Continuous ────────────────────────────────────────────────────────────
    with tab_continuous:
        st.markdown(
            '<p style="font-family:Inter,sans-serif;font-size:0.9rem;color:#5e5d59;'
            'margin-bottom:1.25rem;">'
            'Typical use cases: revenue per user, session duration, page views, ARPU</p>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            alpha_c = st.number_input(
                "Significance level α",
                value=0.05, min_value=0.01, max_value=0.20, step=0.01, format="%.2f",
                key="con_alpha",
            )
        with c2:
            power_c = st.number_input(
                "Power (1 − β)",
                value=0.80, min_value=0.50, max_value=0.99, step=0.05, format="%.2f",
                key="con_power",
            )
        with c3:
            tails_c = st.radio(
                "Test type",
                ["Two-tailed", "One-tailed"],
                horizontal=True,
                key="con_tails",
            )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            mean_c = st.number_input(
                "Baseline mean",
                value=50.0,
                key="con_mean",
                help="Current average value in the control group",
            )
        with c5:
            std_c = st.number_input(
                "Standard deviation",
                value=20.0, min_value=0.01,
                key="con_std",
                help="Standard deviation of the metric (use historical data)",
            )
        with c6:
            mde_c = st.number_input(
                "MDE (absolute shift)",
                value=2.0, min_value=0.001, step=0.5, format="%.2f",
                key="con_mde",
                help="Smallest absolute change worth detecting",
            )

        # Calculation
        alpha_adj_c = alpha_c / 2 if tails_c == "Two-tailed" else alpha_c
        z_a_c = _norm.ppf(1 - alpha_adj_c)
        z_b_c = _norm.ppf(power_c)
        n_c = int(np.ceil(2 * ((z_a_c + z_b_c) * std_c / mde_c) ** 2))
        rel_lift = mde_c / mean_c * 100 if mean_c != 0 else 0

        st.divider()

        from ui.components import kpi_row
        kpi_row([
            ("Per group", f"{n_c:,}"),
            ("Total experiment size", f"{2 * n_c:,}"),
            ("Relative lift", f"{rel_lift:+.1f}%"),
            ("Cohen's d", f"{mde_c / std_c:.3f}"),
        ])

        st.info(
            f"Detecting mean shift of **{mde_c:+.2f}** ({rel_lift:+.1f}%) from baseline **{mean_c}**  \n"
            f"with σ = {std_c:.1f}, α = {alpha_c} ({tails_c.lower()}), power = {power_c:.0%}  \n"
            f"→ **{n_c:,} samples per group**, {2 * n_c:,} total"
        )

        # MDE sensitivity table
        with st.expander("Sensitivity: required N at different MDEs"):
            mde_values_c = [mde_c * m for m in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]]
            rows_c = []
            for m in mde_values_c:
                n_m = int(np.ceil(2 * ((z_a_c + z_b_c) * std_c / m) ** 2))
                rows_c.append({
                    "MDE (absolute)":  f"{m:+.3f}",
                    "MDE (relative)":  f"{m / mean_c:+.1%}" if mean_c != 0 else "-",
                    "Cohen's d":       f"{m / std_c:.3f}",
                    "N per group":     f"{n_m:,}",
                    "N total":         f"{2 * n_m:,}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows_c), hide_index=True, width='stretch')
