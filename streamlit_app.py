"""
A/B Test Analysis Platform — Streamlit + streamlit-echarts

Usage:
    streamlit run streamlit_app.py
"""
import os
import threading
import time
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
from datetime import datetime

from auth_db import (
    init_db,
    register_user,
    authenticate_user,
    change_password,
    save_analysis,
    get_user_history,
)

# ── Initialize database ───────────────────────────────────────────
init_db()

# ── Background thread ─────────────────────────────────────────────
def _analysis_worker(pipeline, data_a, data_b, state: dict) -> None:
    """Background daemon thread; writes results back to shared state dict."""
    try:
        result = pipeline.run(data_a, data_b)
        state["result"]   = result
        state["pipeline"] = pipeline
        state["status"]   = "done"
    except Exception as e:
        state["error"]  = str(e)
        state["status"] = "error"


def _sequential_worker(pipeline, data_a, data_b, reset_seq: bool, state: dict) -> None:
    """Background daemon thread for sequential testing."""
    try:
        if reset_seq or pipeline._sequential_test is None:
            pipeline.init_sequential()
        result = pipeline.run_sequential(data_a, data_b)
        state["result"]   = result
        state["pipeline"] = pipeline
        state["status"]   = "done"
    except Exception as e:
        state["error"]  = str(e)
        state["status"] = "error"

from ab_testing import ABTestPipeline
from ab_testing.visualizer_echarts import (
    posterior_chart,
    delta_chart,
    loss_chart,
    freq_chart,
    sequential_chart,
    sequential_metrics_chart,
)
from scipy.stats import chisquare


def _check_srm(n_a: int, n_b: int, expected_ratio: float = 0.5) -> float:
    """
    Chi-squared goodness-of-fit test for Sample Ratio Mismatch.

    Args:
        n_a:            Actual count in group A (control)
        n_b:            Actual count in group B (treatment)
        expected_ratio: Expected fraction of total assigned to A (default 0.5 = 50/50)

    Returns:
        p-value; small values indicate SRM.
    """
    total = n_a + n_b
    expected_a = total * expected_ratio
    expected_b = total * (1 - expected_ratio)
    _, p_value = chisquare([n_a, n_b], f_exp=[expected_a, expected_b])
    return float(p_value)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Test Analysis Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_DATASETS = {
    "Cookie Cats (binary, high, no effect)": {
        "file": "cookie_cats.csv",
        "group_col": "version",
        "metric_cols": ["retention_1", "retention_7"],
        "control_label": "gate_30",
        "treatment_label": "gate_40",
        "description": "Binary metric (retention), ~90k samples, no significant effect",
    },
    "Simulated Revenue (continuous, low, effect)": {
        "file": "simulated_revenue.csv",
        "group_col": "group",
        "metric_cols": ["revenue"],
        "control_label": "control",
        "treatment_label": "treatment",
        "description": "Continuous metric (revenue), ~1.5k samples, with effect",
    },
    "Binary Low Sample (binary, low, effect)": {
        "file": "binary_low_sample_effect.csv",
        "group_col": "group",
        "metric_cols": ["converted"],
        "control_label": "control",
        "treatment_label": "treatment",
        "description": "Binary metric (conversion), 1.5k samples, with clear effect",
    },
    "Continuous High Sample (continuous, high, effect)": {
        "file": "continuous_high_sample_effect.csv",
        "group_col": "group",
        "metric_cols": ["revenue"],
        "control_label": "control",
        "treatment_label": "treatment",
        "description": "Continuous metric (revenue), 25k samples, with effect",
    },
}

# Default for backward compatibility
SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "cookie_cats.csv")


# ══════════════════════════════════════════════════════════════════
# Login / Register page
# ══════════════════════════════════════════════════════════════════
def show_auth_page():
    st.title("🧪 A/B Test Analysis Platform")
    st.markdown("---")

    _, col_center, _ = st.columns([1, 1.2, 1])
    with col_center:
        st.markdown("### Welcome")

        if "auth_tab" not in st.session_state:
            st.session_state["auth_tab"] = "login"

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if st.button("Login", use_container_width=True,
                         type="primary" if st.session_state["auth_tab"] == "login" else "secondary"):
                st.session_state["auth_tab"] = "login"
                st.rerun()
        with col_t2:
            if st.button("Register", use_container_width=True,
                         type="primary" if st.session_state["auth_tab"] == "register" else "secondary"):
                st.session_state["auth_tab"] = "register"
                st.rerun()

        st.markdown("")

        if st.session_state["auth_tab"] == "login":
            with st.form("login_form"):
                username  = st.text_input("Username")
                password  = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.error("Please enter your username and password.")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state["user"] = user
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        else:
            with st.form("register_form"):
                new_username  = st.text_input("Username")
                new_password  = st.text_input("Password (min. 6 characters)", type="password")
                confirm_pwd   = st.text_input("Confirm Password", type="password")
                reg_submitted = st.form_submit_button("Register", use_container_width=True, type="primary")

            if reg_submitted:
                if new_password != confirm_pwd:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(new_username, new_password)
                    if ok:
                        st.session_state["auth_tab"] = "login"
                        st.session_state["register_success"] = msg
                        st.rerun()
                    else:
                        st.error(msg)

        if st.session_state.pop("register_success", None):
            st.success("Registration successful — please log in.")


# ══════════════════════════════════════════════════════════════════
# Analysis history page
# ══════════════════════════════════════════════════════════════════
def show_history_page():
    user = st.session_state["user"]
    if st.button("← Back to Analysis"):
        st.session_state["page"] = "Analysis"
        st.rerun()
    st.subheader(f"📋 Analysis History  —  {user['username']}")

    records = get_user_history(user["id"], limit=100)
    if not records:
        st.info("No analysis records yet. Run an analysis and it will be saved automatically.")
        return

    rows = []
    for r in records:
        rows.append({
            "ID":             r["id"],
            "Timestamp":      r["created_at"][:19].replace("T", " "),
            "Data Source":    r["data_source"] or "-",
            "Group Column":   r["group_col"] or "-",
            "Control (A)":    r["control_label"] or "-",
            "Treatment (B)":  r["treatment_label"] or "-",
            "Metric":         r["metric_col"] or "-",
            "Metric Type":    r["metric_type"],
            "Method":         r["method"],
            "N (A)":          r["n_control"],
            "N (B)":          r["n_treatment"],
            "MDE":            r["mde"],
            "Freq. Decision": r["freq_decision"] or "-",
            "Bayes Decision": r["bayes_decision"] or "-",
        })
    df_hist = pd.DataFrame(rows)

    st.caption("Click a row to view its details.")
    selection = st.dataframe(
        df_hist,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = selection.selection.rows
    if not selected_rows:
        return

    r = records[selected_rows[0]]
    ctrl_lbl  = r["control_label"]  or "A"
    treat_lbl = r["treatment_label"] or "B"

    st.divider()
    st.markdown(f"#### Record #{r['id']} — {r['created_at'][:19].replace('T', ' ')}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Experiment Config**")
        st.json({
            "Data Source":     r["data_source"],
            "Group Column":    r["group_col"] or "-",
            "Control (A)":     r["control_label"] or "-",
            "Treatment (B)":   r["treatment_label"] or "-",
            "Metric":          r["metric_col"],
            "Metric Type":     r["metric_type"],
            "Method":          r["method"],
            "Alpha":           r["alpha"],
            "MDE":             r["mde"],
            "Loss Threshold":  r["loss_threshold"],
            "N Control (A)":   r["n_control"],
            "N Treatment (B)": r["n_treatment"],
        })
    with c2:
        if r["freq_decision"]:
            st.markdown("**Frequentist Results**")
            st.json({
                f"Mean {ctrl_lbl} (A)":  r["freq_mean_a"],
                f"Mean {treat_lbl} (B)": r["freq_mean_b"],
                "p-value":     r["freq_p_value"],
                "Effect Size": r["freq_effect_size"],
                "Delta":       r["freq_delta"],
                "CI Low":      r["freq_ci_low"],
                "CI High":     r["freq_ci_high"],
                "Significant": bool(r["freq_significant"]),
                "Decision":    r["freq_decision"],
            })
    with c3:
        if r["bayes_decision"]:
            st.markdown("**Bayesian Results**")
            st.json({
                f"Posterior Mean {ctrl_lbl} (A)":  r["bayes_mean_a"],
                f"Posterior Mean {treat_lbl} (B)": r["bayes_mean_b"],
                "P(B > A)":                        r["bayes_prob_b_better"],
                "P(delta > MDE)":                  r["bayes_prob_practical"],
                f"Exp. Loss (Keep {ctrl_lbl})":    r["bayes_loss_a"],
                f"Exp. Loss (Launch {treat_lbl})": r["bayes_loss_b"],
                "Decision":                        r["bayes_decision"],
            })


# ══════════════════════════════════════════════════════════════════
# Change password page
# ══════════════════════════════════════════════════════════════════
def show_change_password_page():
    user = st.session_state["user"]
    if st.button("← Back to Analysis"):
        st.session_state["page"] = "Analysis"
        st.rerun()
    st.subheader("🔑 Change Password")
    col, _ = st.columns([1, 2])
    with col:
        with st.form("change_pwd_form"):
            old_pwd  = st.text_input("Current Password", type="password")
            new_pwd  = st.text_input("New Password (min. 6 characters)", type="password")
            new_pwd2 = st.text_input("Confirm New Password", type="password")
            submitted = st.form_submit_button("Change Password", type="primary")

        if submitted:
            if new_pwd != new_pwd2:
                st.error("New passwords do not match.")
            else:
                ok, msg = change_password(user["id"], old_pwd, new_pwd)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


# ══════════════════════════════════════════════════════════════════
# Main analysis page - SIMPLIFIED VERSION
# ══════════════════════════════════════════════════════════════════
def show_analysis_page():
    # ── Sidebar: Simple configuration ─────────────────────────────
    with st.sidebar:
        st.title("🧪 A/B Test")

        # Step 1: Quick Mode Selection
        st.markdown("### 1. Analysis Mode")

        # Track mode changes to clear results when switching
        prev_mode = st.session_state.get("_prev_mode", None)
        mode = st.radio(
            "Choose analysis mode",
            ["Quick Analysis", "Monitor Over Time", "Expert Mode"],
            horizontal=True,
            label_visibility="collapsed",
        )

        # Clear results if mode changed
        if prev_mode is not None and prev_mode != mode:
            for key in ["result", "pipeline", "worker_state", "worker_thread", "running"]:
                if key in st.session_state:
                    st.session_state.pop(key, None)

        st.session_state["_prev_mode"] = mode

        # Map mode to internal parameters
        if mode == "Quick Analysis":
            method = "both"
            show_advanced = False
        elif mode == "Monitor Over Time":
            method = "sequential"
            show_advanced = False
        else:
            method = "both"
            show_advanced = st.checkbox("Show advanced settings", value=False)

        st.divider()

        # Step 2: Metric Type (simple)
        st.markdown("### 2. Metric Type")
        metric_type = st.radio(
            "What type of metric?",
            ["Binary", "Continuous"],
            horizontal=True,
            label_visibility="collapsed",
        )
        metric_type = metric_type.lower()

        # Smart defaults based on metric type
        if metric_type == "binary":
            default_mde = 0.005
            default_alpha = 0.05
        else:
            default_mde = 3.0
            default_alpha = 0.05

        # Advanced settings (hidden by default)
        alpha = default_alpha
        mde = default_mde
        sequential_method = "obrien_fleming"
        sequential_looks = 5

        if show_advanced or mode == "Expert Mode":
            st.divider()
            st.markdown("### ⚙️ Advanced Settings")

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
                    help="Probability of Type I error (false positive)"
                )

            mde = st.number_input(
                "MDE (Minimum Detectable Effect)",
                value=default_mde,
                format="%.4f",
                help="Smallest effect you care about detecting"
            )

            if method == "sequential":
                sequential_method = st.selectbox(
                    "Sequential method",
                    ["obrien_fleming", "pocock"],
                    format_func=lambda x: {
                        "obrien_fleming": "O'Brien-Fleming (conservative)",
                        "pocock": "Pocock (easier to stop early)",
                    }[x],
                )
                sequential_looks = st.slider(
                    "Max number of looks",
                    min_value=2, max_value=10, value=5,
                )

        # Bayesian parameters - hide in quick mode
        loss_threshold = 0.001 if metric_type == "binary" else 1.0
        prior_strength = 100
        historical_rate = 0.44
        historical_mean = 50.0
        historical_std = 30.0
        nu_expected = 30.0
        n_samples = 200000
        mcmc_draws = 1000
        mcmc_tune = 500
        max_mcmc_samples = 1000
        sequential_n = None
        sequential_wang_delta = 0.5

        if (show_advanced or mode == "Expert Mode") and method != "frequentist":
            st.divider()
            st.markdown("### 🎲 Bayesian Settings")
            loss_threshold = st.number_input(
                "Expected Loss Threshold",
                value=0.001 if metric_type == "binary" else 1.0,
                format="%.4f",
                help="Decision threshold for expected loss"
            )
            prior_strength = st.slider(
                "Prior strength",
                min_value=1, max_value=500, value=100,
                help="How much to trust historical data"
            )
            if metric_type == "binary":
                historical_rate = st.slider(
                    "Historical conversion rate",
                    0.01, 0.99, 0.44, 0.01,
                )
            else:
                historical_mean = st.number_input("Historical mean", value=50.0)
                historical_std = st.number_input("Historical std dev", value=30.0, min_value=0.01)

    # ── Main area: Clean and simple ─────────────────────────────
    st.title("🧪 A/B Test Analysis")

    if mode == "Quick Analysis":
        st.caption("Fast, smart defaults — perfect for most analyses")
    elif mode == "Monitor Over Time":
        st.caption("Watch results accumulate — stop early when evidence is clear")
    else:
        st.caption("Full control for experts — tweak every parameter")

    st.divider()

    # ── Step 1: Load Data ─────────────────────────────────────────
    st.subheader("① Load Data")

    # Track data source changes to clear results when switching
    prev_data_source = st.session_state.get("_prev_data_source", None)
    data_source = st.radio(
        "Data source",
        ["Try sample data", "Upload my CSV"],
        horizontal=True,
    )

    # Clear results if data source changed
    if prev_data_source is not None and prev_data_source != data_source:
        for key in ["result", "pipeline", "worker_state", "worker_thread", "running"]:
            if key in st.session_state:
                st.session_state.pop(key, None)

    st.session_state["_prev_data_source"] = data_source

    df = None
    group_col = metric_col = control_label = treatment_label = None
    data_source_name = ""

    if data_source == "Try sample data":
        sample_options = list(SAMPLE_DATASETS.keys())

        # Track sample dataset changes
        prev_sample = st.session_state.get("_prev_sample", None)
        selected_sample = st.selectbox(
            "Choose a sample dataset",
            sample_options,
            index=0,
            help="Quickly try different scenarios"
        )

        # Clear results if sample changed
        if prev_sample is not None and prev_sample != selected_sample:
            for key in ["result", "pipeline", "worker_state", "worker_thread", "running"]:
                if key in st.session_state:
                    st.session_state.pop(key, None)
        st.session_state["_prev_sample"] = selected_sample

        if selected_sample:
            config = SAMPLE_DATASETS[selected_sample]
            csv_path = os.path.join(os.path.dirname(__file__), config["file"])
            df = pd.read_csv(csv_path)
            group_col = config["group_col"]
            metric_col = st.selectbox("Select metric column", config["metric_cols"])
            control_label = config["control_label"]
            treatment_label = config["treatment_label"]
            data_source_name = selected_sample

            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.caption(f"Control: `{control_label}`  |  Treatment: `{treatment_label}`")
            with col_info2:
                st.caption(f"{len(df):,} total records")
            st.caption(f"💡 {config['description']}")
            st.dataframe(df.head(5), width="stretch")

    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            data_source_name = uploaded.name
            st.dataframe(df.head(5), width="stretch")

            cols = df.columns.tolist()

            def _guess_group_col(dataframe: pd.DataFrame) -> str:
                for c in dataframe.columns:
                    n_unique = dataframe[c].nunique()
                    if 2 <= n_unique <= 10 and dataframe[c].dtype == object:
                        return c
                return min(dataframe.columns, key=lambda c: dataframe[c].nunique())

            def _guess_metric_col(dataframe: pd.DataFrame, exclude: str) -> str:
                numeric_cols = [
                    c for c in dataframe.select_dtypes(include="number").columns
                    if c != exclude and dataframe[c].nunique() < len(dataframe) * 0.9
                ]
                return numeric_cols[0] if numeric_cols else cols[0]

            default_group  = _guess_group_col(df)
            default_metric = _guess_metric_col(df, exclude=default_group)

            c1, c2 = st.columns(2)
            with c1:
                group_col = st.selectbox(
                    "Group column (A/B label)", cols,
                    index=cols.index(default_group),
                )
                metric_col = st.selectbox(
                    "Metric column", [c for c in cols if c != group_col],
                    index=max(0, [c for c in cols if c != group_col].index(default_metric))
                    if default_metric in cols else 0,
                )
            with c2:
                if group_col:
                    groups = df[group_col].unique().tolist()
                    control_label   = st.selectbox("Control group (A)", groups)
                    treatment_label = st.selectbox(
                        "Treatment group (B)", [g for g in groups if g != control_label]
                    )

    st.divider()

    # ── Step 2: Run Analysis ──────────────────────────────────────
    st.subheader("② Run Analysis")

    ready   = df is not None and all([group_col, metric_col, control_label, treatment_label])
    running = st.session_state.get("running", False)

    # Always create status placeholder
    status_placeholder = st.empty()

    # Handle reset button first (before any other logic)
    if method == "sequential" and ready:
        # Check if reset button was clicked in a previous rerun
        if st.session_state.get("_reset_seq_pending", False):
            st.session_state.pop("pipeline", None)
            st.session_state.pop("result", None)
            st.session_state.pop("_reset_seq_pending", None)
            st.rerun()

    # Sequential testing specific UI
    if method == "sequential" and ready:
        st.info(
            "📊 **Monitor Mode**  \n"
            "Use the slider to simulate data accumulating over time. "
            "Start small and increase gradually!"
        )

        full_data_a = df[df[group_col] == control_label][metric_col].values.astype(float)
        full_data_b = df[df[group_col] == treatment_label][metric_col].values.astype(float)

        col_pct, col_reset, col_run = st.columns([2, 1, 1])
        with col_pct:
            data_pct = st.slider(
                "Percentage of data to use",
                min_value=10, max_value=100, value=50, step=10,
                help="Select what percentage of the full dataset to use for this look"
            )
        with col_reset:
            if st.button("🔄 New Test", help="Reset and start a new sequential test"):
                st.session_state["_reset_seq_pending"] = True
                st.rerun()
        with col_run:
            run_btn = st.button("📊 Add Look", type="primary", disabled=not ready or running, use_container_width=True)

        # Calculate current data based on percentage
        n_a_current = max(10, int(len(full_data_a) * data_pct / 100))
        n_b_current = max(10, int(len(full_data_b) * data_pct / 100))

        st.caption(f"Current: {n_a_current:,} (A) + {n_b_current:,} (B) = {n_a_current + n_b_current:,} total samples")

        data_a = full_data_a[:n_a_current]
        data_b = full_data_b[:n_b_current]
    else:
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            run_btn = st.button("🚀 Run Analysis", type="primary",
                                disabled=not ready or running, use_container_width=True)

        if ready:
            data_a = df[df[group_col] == control_label][metric_col].values.astype(float)
            data_b = df[df[group_col] == treatment_label][metric_col].values.astype(float)

    # ── Launch analysis ───────────────────────────────────────────
    if not running and ready and "run_btn" in dir() and run_btn:
        pipeline_kwargs = dict(
            metric_type=metric_type,
            method=method,
            alpha=alpha,
            mde=float(mde),
            loss_threshold=float(loss_threshold),
        )

        if method != "frequentist":
            pipeline_kwargs["prior_strength"] = prior_strength
            pipeline_kwargs["n_samples"]      = n_samples
            if metric_type == "binary":
                pipeline_kwargs["historical_rate"] = historical_rate
            else:
                pipeline_kwargs["mcmc_draws"]       = int(mcmc_draws)
                pipeline_kwargs["mcmc_tune"]        = int(mcmc_tune)
                pipeline_kwargs["max_mcmc_samples"] = int(max_mcmc_samples)
                pipeline_kwargs["nu_expected"]      = float(nu_expected)
                pipeline_kwargs["historical_mean"] = historical_mean
                pipeline_kwargs["historical_std"]  = historical_std

        if method == "sequential":
            pipeline_kwargs["sequential_method"] = sequential_method
            pipeline_kwargs["sequential_looks"] = sequential_looks
            pipeline_kwargs["sequential_n"] = sequential_n
            pipeline_kwargs["sequential_wang_delta"] = sequential_wang_delta

        pipeline = ABTestPipeline(**pipeline_kwargs)

        # Reuse existing pipeline for sequential testing if available
        reset_for_worker = False
        if method == "sequential" and "pipeline" in st.session_state:
            pipeline = st.session_state["pipeline"]
        elif method == "sequential":
            reset_for_worker = True

        worker_state: dict = {"status": "running", "result": None,
                              "pipeline": None, "error": None}

        if method == "sequential":
            thread = threading.Thread(
                target=_sequential_worker,
                args=(pipeline, data_a, data_b, reset_for_worker, worker_state),
                daemon=True,
            )
        else:
            thread = threading.Thread(
                target=_analysis_worker,
                args=(pipeline, data_a, data_b, worker_state),
                daemon=True,
            )
        thread.start()

        st.session_state["running"]          = True
        st.session_state["worker_state"]     = worker_state
        st.session_state["worker_thread"]    = thread
        st.session_state["metric_col"]       = metric_col
        st.session_state["data_source_name"] = data_source_name
        st.session_state["analysis_meta"]    = {
            "metric_type":     metric_type,
            "method":          method,
            "alpha":           alpha,
            "mde":             float(mde),
            "loss_threshold":  float(loss_threshold),
            "n_control":       int(len(data_a)),
            "n_treatment":     int(len(data_b)),
            "group_col":       group_col,
            "control_label":   str(control_label),
            "treatment_label": str(treatment_label),
        }
        st.rerun()

    # ── Poll status ───────────────────────────────────────────────
    if running:
        worker_state = st.session_state.get("worker_state", {})
        status = worker_state.get("status", "running")

        if status == "running":
            status_placeholder.info("⏳ Analysis in progress...")
            time.sleep(0.5)
            st.rerun()

        elif status == "done":
            result   = worker_state["result"]
            pipeline = worker_state["pipeline"]

            st.session_state["result"]   = result
            st.session_state["pipeline"] = pipeline
            st.session_state["running"]  = False
            st.session_state.pop("worker_state",  None)
            st.session_state.pop("worker_thread", None)
            status_placeholder.success("✅ Analysis complete")

            # For sequential testing, keep the pipeline in session state
            if method == "sequential":
                st.session_state["pipeline"] = pipeline

            # ── Auto-save to database ─────────────────────────────
            user = st.session_state.get("user")
            if user:
                meta = st.session_state.get("analysis_meta", {})
                record = {
                    "created_at":      datetime.now().isoformat(),
                    "data_source":     st.session_state.get("data_source_name", ""),
                    "group_col":       meta.get("group_col"),
                    "control_label":   meta.get("control_label"),
                    "treatment_label": meta.get("treatment_label"),
                    "metric_col":      st.session_state.get("metric_col", ""),
                    "metric_type":     meta.get("metric_type"),
                    "method":          meta.get("method"),
                    "alpha":           meta.get("alpha"),
                    "mde":             meta.get("mde"),
                    "loss_threshold":  meta.get("loss_threshold"),
                    "n_control":       meta.get("n_control"),
                    "n_treatment":     meta.get("n_treatment"),
                    # Frequentist
                    "freq_mean_a":      result.frequentist.mean_a      if result.frequentist else None,
                    "freq_mean_b":      result.frequentist.mean_b      if result.frequentist else None,
                    "freq_p_value":     result.frequentist.p_value     if result.frequentist else None,
                    "freq_effect_size": result.frequentist.effect_size if result.frequentist else None,
                    "freq_delta":       result.frequentist.delta       if result.frequentist else None,
                    "freq_ci_low":      result.frequentist.ci[0]       if result.frequentist else None,
                    "freq_ci_high":     result.frequentist.ci[1]       if result.frequentist else None,
                    "freq_significant": int(result.frequentist.significant) if result.frequentist else None,
                    "freq_decision":    result.decision_freq            if result.frequentist else None,
                    # Bayesian
                    "bayes_mean_a":         result.bayesian.mean_a          if result.bayesian else None,
                    "bayes_mean_b":         result.bayesian.mean_b          if result.bayesian else None,
                    "bayes_prob_b_better":  result.bayesian.prob_b_better   if result.bayesian else None,
                    "bayes_prob_practical": result.bayesian.prob_practical  if result.bayesian else None,
                    "bayes_loss_a":         result.bayesian.expected_loss_a if result.bayesian else None,
                    "bayes_loss_b":         result.bayesian.expected_loss_b if result.bayesian else None,
                    "bayes_decision":       result.decision_bayes           if result.bayesian else None,
                }
                rec_id = save_analysis(user["id"], record)
                st.session_state["last_saved_id"] = rec_id

            st.rerun()

        elif status == "error":
            st.session_state["running"] = False
            status_placeholder.error(f"❌ Analysis error: {worker_state.get('error')}")
            st.session_state.pop("worker_state",  None)
            st.session_state.pop("worker_thread", None)
            st.rerun()

    if st.session_state.get("last_saved_id"):
        st.toast(f"✅ Result saved (Record ID: {st.session_state['last_saved_id']})", icon="💾")
        st.session_state.pop("last_saved_id", None)

    # ── Display results ───────────────────────────────────────────
    if "result" in st.session_state:
        result   = st.session_state["result"]
        pipeline = st.session_state["pipeline"]
        m_label  = st.session_state.get("metric_col", "Metric")

        st.divider()
        st.subheader("③ Results")

        # ── SRM check ────────────────────────────────────────────────
        _meta = st.session_state.get("analysis_meta", {})
        _n_a  = _meta.get("n_control",   0)
        _n_b  = _meta.get("n_treatment", 0)
        if _n_a and _n_b:
            _srm_p = _check_srm(_n_a, _n_b)
            if _srm_p < 0.01:
                _ratio_a = _n_a / (_n_a + _n_b)
                st.error(
                    f"**⚠️ Sample Ratio Mismatch (SRM) Detected**  —  p = {_srm_p:.4f} (threshold: 0.01)\n\n"
                    f"Expected a 50 / 50 split, but observed **{_n_a:,}** (A) vs **{_n_b:,}** (B)  "
                    f"({_ratio_a:.1%} / {1 - _ratio_a:.1%}).  \n"
                    "A significant imbalance usually indicates a bug in the randomization or data pipeline "
                    "(e.g. cookie dropping, bot filtering applied to only one arm, or a logging gap).  \n"
                    "**The analysis results below may be biased — investigate the root cause before drawing conclusions.**"
                )
            else:
                st.success(
                    f"✅ **No SRM detected**  —  p = {_srm_p:.4f}  "
                    f"({_n_a:,} A  /  {_n_b:,} B,  ratio {_n_a / (_n_a + _n_b):.1%} / {_n_b / (_n_a + _n_b):.1%})"
                )

        def _decision_banner(decision: str, method_name: str):
            if "Launch B" in decision or "Reject H0" in decision:
                st.success(f"**{method_name} Decision: {decision}**")
            elif "Keep A" in decision or "Accept H0" in decision:
                st.info(f"**{method_name} Decision: {decision}**")
            else:
                st.warning(f"**{method_name} Decision: {decision}**")

        if result.frequentist and result.decision_freq:
            _decision_banner(result.decision_freq, "[Frequentist]")
        if result.bayesian and result.decision_bayes:
            _decision_banner(result.decision_bayes, "[Bayesian]")
        if result.sequential and result.decision_seq:
            _decision_banner(result.decision_seq, "[Sequential]")

        def _card(label: str, value: str, sub: str = "", sub_color: str = "#555") -> str:
            return (
                f'<div style="background:#f5f6fa;border-radius:8px;padding:8px 10px;'
                f'border:1px solid #e2e4ea;min-width:0;overflow:hidden">'
                f'<div style="font-size:0.68rem;color:#888;font-weight:500;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{label}</div>'
                f'<div style="font-size:0.95rem;font-weight:700;color:#1a1a2e;'
                f'margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{value}</div>'
                + (f'<div style="font-size:0.65rem;color:{sub_color};margin-top:1px;'
                   f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{sub}</div>' if sub else "")
                + "</div>"
            )

        def _kpi_row(cards: list[tuple]) -> None:
            cols = len(cards)
            html = (
                f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);'
                f'gap:8px;margin-bottom:10px">'
            )
            for item in cards:
                html += _card(*item)
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

        st.markdown("#### Key Metrics")
        if result.frequentist:
            f = result.frequentist
            sig_sub   = "Significant ✓" if f.significant else "Not Significant ✗"
            sig_color = "#27ae60" if f.significant else "#888"

            delta_vs_mde = f.delta >= pipeline.mde
            mde_sub   = "Reached MDE ✓" if delta_vs_mde else "Below MDE ✗"
            mde_color = "#27ae60" if delta_vs_mde else "#888"

            _kpi_row([
                ("Group A Mean",   f"{f.mean_a:.4f}"),
                ("Group B Mean",   f"{f.mean_b:.4f}"),
                ("p-value",        f"{f.p_value:.4f}", sig_sub, sig_color),
                ("Effect Size",    f"{f.effect_size:.4f}"),
                ("delta vs MDE",   f"{f.delta:+.4f} vs {pipeline.mde:.4f}", mde_sub, mde_color),
                ("95% CI",         f"[{f.ci[0]:+.4f}, {f.ci[1]:+.4f}]"),
            ])
        if result.bayesian:
            b = result.bayesian
            _kpi_row([
                ("A Posterior Mean",  f"{b.mean_a:.4f}"),
                ("B Posterior Mean",  f"{b.mean_b:.4f}"),
                ("P(B > A)",          f"{b.prob_b_better:.1%}"),
                ("P(delta > MDE)",    f"{b.prob_practical:.1%}"),
                ("Exp. Loss (Keep A)", f"{b.expected_loss_a:.5f}"),
                ("Exp. Loss (Launch B)", f"{b.expected_loss_b:.5f}"),
            ])

        st.markdown("---")

        # ── Frequentist charts ────────────────────────────────────
        if result.frequentist and not result.sequential:
            st.markdown("#### Frequentist: Mean Comparison & Confidence Interval")
            st_echarts(
                options=freq_chart(result.frequentist, metric_label=m_label, mde=pipeline.mde),
                height="380px",
                key="chart_freq",
            )

        # ── Bayesian charts ───────────────────────────────────────
        if result.bayesian and not result.sequential:
            b = result.bayesian
            st.markdown("#### Bayesian: Posterior Distribution & Decision Metrics")

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
                st_echarts(
                    options=posterior_chart(b, metric_label=m_label),
                    height="400px",
                    key="chart_posterior",
                )

            with tab2:
                st.caption(
                    f"P(B > A) = {b.prob_b_better:.1%}  |  "
                    f"P(delta > MDE={pipeline.mde}) = {b.prob_practical:.1%}"
                )
                st_echarts(
                    options=delta_chart(b, mde=pipeline.mde, metric_label=m_label),
                    height="400px",
                    key="chart_delta",
                )

            with tab3:
                st.caption(
                    f"Expected loss (Keep A): {b.expected_loss_a:.6f}  |  "
                    f"Expected loss (Ship B): {b.expected_loss_b:.6f}  |  "
                    f"Threshold: {pipeline.loss_threshold}"
                )
                st_echarts(
                    options=loss_chart(b, loss_threshold=pipeline.loss_threshold),
                    height="380px",
                    key="chart_loss",
                )

        # ── Sequential testing charts ─────────────────────────────────
        if result.sequential:
            s = result.sequential
            st.markdown("#### Sequential: Test Boundaries & Metrics Over Time")

            # Key metrics for sequential test
            if s.looks:
                last_look = s.looks[-1]
                _kpi_row([
                    ("Current Look", f"{s.current_look} / {len(s.boundary_values)}"),
                    ("N (A)", f"{last_look.n_a:,}"),
                    ("N (B)", f"{last_look.n_b:,}"),
                    ("Test Statistic", f"{last_look.statistic:+.4f}"),
                    ("Naive p-value", f"{last_look.p_value:.4f}" if last_look.p_value else "-"),
                ])

            tab_seq1, tab_seq2 = st.tabs(["Boundary Chart", "Metrics Evolution"])

            with tab_seq1:
                st.caption(
                    "Test statistic path vs. rejection boundaries. "
                    "Crossing a boundary indicates a statistically significant result."
                )
                st_echarts(
                    options=sequential_chart(s),
                    height="420px",
                    key="chart_seq_boundary",
                )

            with tab_seq2:
                st_echarts(
                    options=sequential_metrics_chart(s),
                    height="400px",
                    key="chart_seq_metrics",
                )

            with st.expander("📋 View Sequential Test Details"):
                st.code(s.summary(), language=None)

        with st.expander("📋 View Full Text Summary"):
            st.code(result.summary(), language=None)


# ══════════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════════
user = st.session_state.get("user")

if not user:
    show_auth_page()
else:
    page = st.session_state.get("page", "Analysis")

    with st.sidebar:
        st.divider()
        col_name, col_pop = st.columns([3, 1])
        with col_name:
            st.markdown(f"👤 **{user['username']}**")
        with col_pop:
            # Only render the popover on the Analysis page so it unmounts
            # (and therefore closes) when navigating to sub-pages.
            if page == "Analysis":
                with st.popover("▾", use_container_width=True):
                    if st.button("📋 History", use_container_width=True):
                        st.session_state["page"] = "History"
                        st.rerun()
                    if st.button("🔑 Change Password", use_container_width=True):
                        st.session_state["page"] = "Change Password"
                        st.rerun()
                    st.divider()
                    if st.button("🚪 Log Out", use_container_width=True):
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.rerun()
            else:
                if st.button("▾", use_container_width=True):
                    st.session_state["page"] = "Analysis"
                    st.rerun()

    if page == "Analysis":
        show_analysis_page()
    elif page == "History":
        show_history_page()
    elif page == "Change Password":
        show_change_password_page()
