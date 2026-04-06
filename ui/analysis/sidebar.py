"""Sidebar: analysis mode, metric type, and parameter configuration.

Returns a single ``params`` dict consumed by the rest of the analysis page.
"""
import streamlit as st

_RESULT_KEYS = ["result", "pipeline", "worker_state", "worker_thread", "running"]


def _clear_results() -> None:
    for key in _RESULT_KEYS:
        st.session_state.pop(key, None)


def render_sidebar() -> dict:
    """Render the full sidebar and return a params dict."""
    with st.sidebar:
        # ── 1. Analysis Mode ─────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.65rem;font-weight:500;letter-spacing:0.5px;'
            'text-transform:uppercase;color:#87867f;margin-bottom:0.25rem;'
            'font-family:Inter,sans-serif;">1 — Analysis Mode</p>',
            unsafe_allow_html=True,
        )

        prev_mode = st.session_state.get("_prev_mode")
        mode = st.radio(
            "Choose analysis mode",
            ["Quick Analysis", "Monitor Over Time", "Expert Mode", "Sample Size"],
            label_visibility="collapsed",
        )

        # Sample Size is a standalone page — navigate immediately
        if mode == "Sample Size":
            st.session_state.pop("_prev_mode", None)
            st.session_state["page"] = "Sample Size"
            st.rerun()

        if prev_mode is not None and prev_mode != mode:
            _clear_results()
        st.session_state["_prev_mode"] = mode

        if mode == "Quick Analysis":
            method = "both"
            show_advanced = False
        elif mode == "Monitor Over Time":
            method = "sequential"
            show_advanced = False
        else:  # Expert Mode
            method = "both"
            show_advanced = True

        st.divider()

        # ── 2. Metric Type ────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.65rem;font-weight:500;letter-spacing:0.5px;'
            'text-transform:uppercase;color:#87867f;margin-bottom:0.25rem;'
            'font-family:Inter,sans-serif;">2 — Metric Type</p>',
            unsafe_allow_html=True,
        )
        metric_type = st.radio(
            "What type of metric?",
            ["Binary", "Continuous"],
            horizontal=True,
            label_visibility="collapsed",
        ).lower()

        default_mde   = 0.005 if metric_type == "binary" else 3.0
        default_alpha = 0.05

        # ── Advanced / Expert settings ────────────────────────────
        alpha            = default_alpha
        mde              = default_mde
        sequential_method = "obrien_fleming"
        sequential_looks  = 5

        if show_advanced:
            st.divider()
            st.markdown(
                '<p style="font-size:0.65rem;font-weight:500;letter-spacing:0.5px;'
                'text-transform:uppercase;color:#87867f;margin-bottom:0.25rem;'
                'font-family:Inter,sans-serif;">Advanced Settings</p>',
                unsafe_allow_html=True,
            )

            if mode == "Expert Mode":
                method = st.radio(
                    "Analysis Method",
                    ["both", "bayesian", "frequentist", "sequential"],
                    format_func=lambda x: {
                        "both":        "Both Frequentist + Bayesian",
                        "bayesian":    "Bayesian only",
                        "frequentist": "Frequentist only",
                        "sequential":  "Sequential testing",
                    }[x],
                )

            if method != "bayesian":
                alpha = st.slider(
                    "Significance level α",
                    0.01, 0.10, default_alpha, 0.01,
                    help="Probability of Type I error (false positive)",
                )

            mde = st.number_input(
                "MDE (Minimum Detectable Effect)",
                value=default_mde,
                format="%.4f",
                help="Smallest effect you care about detecting",
            )

            if method == "sequential":
                sequential_method = st.selectbox(
                    "Sequential method",
                    ["obrien_fleming", "pocock"],
                    format_func=lambda x: {
                        "obrien_fleming": "O'Brien-Fleming (conservative)",
                        "pocock":         "Pocock (easier to stop early)",
                    }[x],
                    width='stretch',
                )
                sequential_looks = st.slider(
                    "Max number of looks",
                    min_value=2, max_value=10, value=5,
                )

        # ── Bayesian defaults ─────────────────────────────────────
        loss_threshold       = 0.001 if metric_type == "binary" else 1.0
        prior_strength       = 100
        historical_rate      = 0.44
        historical_mean      = 50.0
        historical_std       = 30.0
        nu_expected          = 30.0
        n_samples            = 200_000
        mcmc_draws           = 1000
        mcmc_tune            = 500
        max_mcmc_samples     = 1000
        sequential_n         = None
        sequential_wang_delta = 0.5

        if show_advanced and method != "frequentist":
            st.divider()
            st.markdown(
                '<p style="font-size:0.65rem;font-weight:500;letter-spacing:0.5px;'
                'text-transform:uppercase;color:#87867f;margin-bottom:0.25rem;'
                'font-family:Inter,sans-serif;">Bayesian Settings</p>',
                unsafe_allow_html=True,
            )

            loss_threshold = st.number_input(
                "Expected Loss Threshold",
                value=0.001 if metric_type == "binary" else 1.0,
                format="%.4f",
                help="Decision threshold for expected loss",
            )
            prior_strength = st.slider(
                "Prior strength",
                min_value=1, max_value=500, value=100,
                help="How much to trust historical data",
            )

            # ── Prior parameters: upload file or enter manually ───
            prior_source = st.radio(
                "Prior source",
                ["Enter manually", "Compute from file"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if prior_source == "Compute from file":
                prior_file = st.file_uploader(
                    "Upload historical CSV",
                    type=["csv"],
                    key="_prior_file",
                    help="Upload a historical dataset to auto-compute prior parameters",
                )
                if prior_file is not None:
                    import pandas as pd
                    _pdf = pd.read_csv(prior_file)
                    _num_cols = _pdf.select_dtypes(include="number").columns.tolist()
                    if _num_cols:
                        _prior_col = st.selectbox(
                            "Select metric column",
                            _num_cols,
                            key="_prior_col",
                            width='stretch',
                        )
                        _col_data = _pdf[_prior_col].dropna()
                        if metric_type == "binary":
                            historical_rate = float(_col_data.mean())
                            historical_rate = max(0.01, min(0.99, historical_rate))
                            st.caption(f"Computed rate: **{historical_rate:.4f}**")
                        else:
                            historical_mean = float(_col_data.mean())
                            historical_std  = float(_col_data.std())
                            historical_std  = max(0.01, historical_std)
                            st.caption(
                                f"Computed mean: **{historical_mean:.4f}**  "
                                f"|  std: **{historical_std:.4f}**  "
                                f"|  n = {len(_col_data):,}"
                            )
                    else:
                        st.warning("No numeric columns found in the uploaded file.")
            else:
                if metric_type == "binary":
                    historical_rate = st.slider(
                        "Historical conversion rate",
                        0.01, 0.99, 0.44, 0.01,
                    )
                else:
                    historical_mean = st.number_input("Historical mean", value=50.0)
                    historical_std  = st.number_input("Historical std dev", value=30.0, min_value=0.01)

    return dict(
        mode=mode,
        method=method,
        metric_type=metric_type,
        alpha=alpha,
        mde=float(mde),
        loss_threshold=float(loss_threshold),
        prior_strength=prior_strength,
        historical_rate=historical_rate,
        historical_mean=historical_mean,
        historical_std=historical_std,
        nu_expected=nu_expected,
        n_samples=n_samples,
        mcmc_draws=mcmc_draws,
        mcmc_tune=mcmc_tune,
        max_mcmc_samples=max_mcmc_samples,
        sequential_method=sequential_method,
        sequential_looks=sequential_looks,
        sequential_n=sequential_n,
        sequential_wang_delta=sequential_wang_delta,
    )
