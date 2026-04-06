"""Analysis runner: background thread management, polling, and auto-save."""
import threading
import time
from datetime import datetime

import streamlit as st

from ab_testing import ABTestPipeline
from auth_db import save_analysis


# ── Background workers ────────────────────────────────────────────────────────

def _analysis_worker(pipeline, data_a, data_b, state: dict) -> None:
    try:
        result = pipeline.run(data_a, data_b)
        state["result"]   = result
        state["pipeline"] = pipeline
        state["status"]   = "done"
    except Exception as exc:
        state["error"]  = str(exc)
        state["status"] = "error"


def _sequential_worker(pipeline, data_a, data_b, reset_seq: bool, state: dict) -> None:
    try:
        if reset_seq or pipeline._sequential_test is None:
            pipeline.init_sequential()
        result = pipeline.run_sequential(data_a, data_b)
        state["result"]   = result
        state["pipeline"] = pipeline
        state["status"]   = "done"
    except Exception as exc:
        state["error"]  = str(exc)
        state["status"] = "error"


# ── Auto-save helper ──────────────────────────────────────────────────────────

def _auto_save(result, meta: dict, data_source_name: str, metric_col: str) -> None:
    user = st.session_state.get("user")
    if not user:
        return
    record = {
        "created_at":      datetime.now().isoformat(),
        "data_source":     data_source_name,
        "group_col":       meta.get("group_col"),
        "control_label":   meta.get("control_label"),
        "treatment_label": meta.get("treatment_label"),
        "metric_col":      metric_col,
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


# ── Main render function ──────────────────────────────────────────────────────

def _validate_data_for_metric_type(df, metric_col, metric_type, control_label, treatment_label, group_col) -> tuple[bool, str]:
    """Validate that the data matches the selected metric type."""
    if df is None or metric_col is None:
        return True, ""

    # Get combined data from both groups
    mask_a = df[group_col] == control_label
    mask_b = df[group_col] == treatment_label
    metric_data = df.loc[mask_a | mask_b, metric_col].dropna()

    if len(metric_data) == 0:
        return False, "No valid data found in the metric column for the selected groups."

    unique_vals = metric_data.unique()
    num_unique = len(unique_vals)

    # Try to convert to numeric
    try:
        numeric_data = pd.to_numeric(metric_data, errors='coerce').dropna()
        has_numeric = len(numeric_data) > 0
        num_unique_numeric = len(numeric_data.unique()) if has_numeric else 0
    except:
        has_numeric = False
        num_unique_numeric = 0

    if metric_type == "binary":
        # Check if data is binary (0/1, True/False, or only 2 unique values)
        # Try to convert to numeric to check
        if has_numeric:
            # Check if all values are 0 or 1
            is_01 = set(numeric_data.unique()).issubset({0, 1})
            if is_01:
                return True, ""

            # Check if there are exactly 2 unique numeric values
            if num_unique_numeric == 2:
                return True, ""

        # Check non-numeric binary data (exactly 2 unique values)
        if num_unique == 2:
            return True, ""

        # Check if data looks like continuous/float
        if has_numeric:
            # If there are many unique float values, it's likely continuous
            if num_unique_numeric > 10:
                return False, (
                    f"⚠️ Selected metric type is 'Binary', but the data in '{metric_col}' "
                    f"appears to be continuous ({num_unique_numeric} unique values). "
                    f"Please switch to 'Continuous' metric type, or ensure your binary data "
                    f"contains only 0/1 or two distinct values."
                )

        return False, (
            f"⚠️ Selected metric type is 'Binary', but the data in '{metric_col}' "
            f"doesn't appear to be binary. Binary data should contain only 0/1, "
            f"True/False, or exactly two distinct values (found {num_unique} unique values)."
        )

    elif metric_type == "continuous":
        # Check if data looks like binary (only 2 unique values)
        if num_unique <= 2:
            return False, (
                f"⚠️ Selected metric type is 'Continuous', but the data in '{metric_col}' "
                f"appears to be binary/categorical (only {num_unique} unique values: {list(unique_vals)[:5]}). "
                f"Please switch to 'Binary' metric type for binary/categorical data."
            )

    return True, ""


def render_runner(data_info: dict, params: dict) -> None:
    """Render the run-analysis section (button / slider, threading, polling)."""
    import pandas as pd

    df             = data_info["df"]
    group_col      = data_info["group_col"]
    metric_col     = data_info["metric_col"]
    control_label  = data_info["control_label"]
    treatment_label = data_info["treatment_label"]
    data_source_name = data_info["data_source_name"]

    method  = params["method"]
    ready   = df is not None and all([group_col, metric_col, control_label, treatment_label])
    running = st.session_state.get("running", False)

    status_placeholder = st.empty()

    # Initialize data variables
    data_a = None
    data_b = None
    full_data_a = None
    full_data_b = None
    n_a = 0
    n_b = 0

    # Validate data if ready
    data_valid = True
    validation_message = ""
    if ready:
        data_valid, validation_message = _validate_data_for_metric_type(
            df, metric_col, params["metric_type"],
            control_label, treatment_label, group_col
        )
        if not data_valid:
            st.warning(validation_message)

    # Handle pending sequential reset
    if method == "sequential" and ready:
        if st.session_state.pop("_reset_seq_pending", False):
            st.session_state.pop("pipeline", None)
            st.session_state.pop("result",   None)
            st.rerun()

    # ── Sequential mode UI ────────────────────────────────────────
    if method == "sequential" and ready:
        full_data_a = df[df[group_col] == control_label][metric_col].values.astype(float)
        full_data_b = df[df[group_col] == treatment_label][metric_col].values.astype(float)

        col_slider, _ = st.columns([3, 2])
        with col_slider:
            data_pct = st.slider(
                "Percentage of data to use",
                min_value=10, max_value=100, value=50, step=10,
                help="Select what percentage of the full dataset to use for this look",
            )
            n_a = max(10, int(len(full_data_a) * data_pct / 100))
            n_b = max(10, int(len(full_data_b) * data_pct / 100))
            st.caption(f"Current: {n_a:,} (A) + {n_b:,} (B) = {n_a + n_b:,} total")

            col_reset, col_run = st.columns(2)
            with col_reset:
                if st.button("New Test", width='stretch',
                             help="Reset and start a new sequential test"):
                    st.session_state["_reset_seq_pending"] = True
                    st.rerun()
            with col_run:
                run_btn = st.button(
                    "Add Look",
                    disabled=not ready or not data_valid or running,
                    width='stretch',
                )
        data_a = full_data_a[:n_a]
        data_b = full_data_b[:n_b]

    else:
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            run_btn = st.button(
                "🚀 Run Analysis", type="primary",
                disabled=not ready or not data_valid or running,
                width='stretch',
            )
        if ready and data_valid:
            data_a = df[df[group_col] == control_label][metric_col].values.astype(float)
            data_b = df[df[group_col] == treatment_label][metric_col].values.astype(float)

    # ── Launch background thread ──────────────────────────────────
    if not running and ready and data_valid and run_btn:
        pipeline_kwargs = dict(
            metric_type=params["metric_type"],
            method=method,
            alpha=params["alpha"],
            mde=params["mde"],
            loss_threshold=params["loss_threshold"],
        )
        if method != "frequentist":
            pipeline_kwargs["prior_strength"] = params["prior_strength"]
            pipeline_kwargs["n_samples"]      = params["n_samples"]
            if params["metric_type"] == "binary":
                pipeline_kwargs["historical_rate"] = params["historical_rate"]
            else:
                pipeline_kwargs["mcmc_draws"]       = int(params["mcmc_draws"])
                pipeline_kwargs["mcmc_tune"]        = int(params["mcmc_tune"])
                pipeline_kwargs["max_mcmc_samples"] = int(params["max_mcmc_samples"])
                pipeline_kwargs["nu_expected"]      = float(params["nu_expected"])
                pipeline_kwargs["historical_mean"]  = params["historical_mean"]
                pipeline_kwargs["historical_std"]   = params["historical_std"]

        if method == "sequential":
            pipeline_kwargs["sequential_method"]      = params["sequential_method"]
            pipeline_kwargs["sequential_looks"]       = params["sequential_looks"]
            pipeline_kwargs["sequential_n"]           = params["sequential_n"]
            pipeline_kwargs["sequential_wang_delta"]  = params["sequential_wang_delta"]

        pipeline = ABTestPipeline(**pipeline_kwargs)
        reset_for_worker = False
        if method == "sequential" and "pipeline" in st.session_state:
            pipeline = st.session_state["pipeline"]
        elif method == "sequential":
            reset_for_worker = True

        worker_state: dict = {"status": "running", "result": None, "pipeline": None, "error": None}

        target = _sequential_worker if method == "sequential" else _analysis_worker
        args   = (pipeline, data_a, data_b, reset_for_worker, worker_state) if method == "sequential" \
                 else (pipeline, data_a, data_b, worker_state)

        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()

        st.session_state["running"]          = True
        st.session_state["worker_state"]     = worker_state
        st.session_state["worker_thread"]    = thread
        st.session_state["metric_col"]       = metric_col
        st.session_state["data_source_name"] = data_source_name
        st.session_state["analysis_meta"] = {
            "metric_type":     params["metric_type"],
            "method":          method,
            "alpha":           params["alpha"],
            "mde":             params["mde"],
            "loss_threshold":  params["loss_threshold"],
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
            status_placeholder.info("Analysis in progress...")
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

            if method == "sequential":
                st.session_state["pipeline"] = pipeline

            status_placeholder.success("Analysis complete")
            _auto_save(
                result,
                st.session_state.get("analysis_meta", {}),
                st.session_state.get("data_source_name", ""),
                st.session_state.get("metric_col", ""),
            )
            st.rerun()

        elif status == "error":
            st.session_state["running"] = False
            status_placeholder.error(f"Analysis error: {worker_state.get('error')}")
            st.session_state.pop("worker_state",  None)
            st.session_state.pop("worker_thread", None)
            st.rerun()

    if st.session_state.get("last_saved_id"):
        st.toast(f"Result saved (Record ID: {st.session_state['last_saved_id']})")
        st.session_state.pop("last_saved_id", None)
