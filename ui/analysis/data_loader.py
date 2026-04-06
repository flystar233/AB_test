"""Data loading section: sample datasets or user-uploaded CSV."""
import os

import pandas as pd
import streamlit as st

SAMPLE_DATASETS: dict[str, dict] = {
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

_RESULT_KEYS = ["result", "pipeline", "worker_state", "worker_thread", "running"]


def _clear_results() -> None:
    for key in _RESULT_KEYS:
        st.session_state.pop(key, None)


def _guess_group_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if 2 <= df[c].nunique() <= 10 and df[c].dtype == object:
            return c
    return min(df.columns, key=lambda c: df[c].nunique())


def _guess_metric_col(df: pd.DataFrame, exclude: str) -> str:
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != exclude and df[c].nunique() < len(df) * 0.9
    ]
    return numeric_cols[0] if numeric_cols else df.columns[0]


def render_data_loader() -> dict:
    """Render the data loading UI and return a data_info dict.

    Returns
    -------
    dict with keys: df, group_col, metric_col, control_label,
                    treatment_label, data_source_name
    """
    prev_source = st.session_state.get("_prev_data_source")
    data_source = st.radio(
        "Data source",
        ["Try sample data", "Upload my CSV"],
        horizontal=True,
    )
    if prev_source is not None and prev_source != data_source:
        _clear_results()
    st.session_state["_prev_data_source"] = data_source

    df = None
    group_col = metric_col = control_label = treatment_label = None
    data_source_name = ""

    if data_source == "Try sample data":
        prev_sample = st.session_state.get("_prev_sample")
        selected_sample = st.selectbox(
            "Choose a sample dataset",
            list(SAMPLE_DATASETS.keys()),
            index=0,
            help="Quickly try different scenarios",
            width='stretch',
        )
        if prev_sample is not None and prev_sample != selected_sample:
            _clear_results()
        st.session_state["_prev_sample"] = selected_sample

        config   = SAMPLE_DATASETS[selected_sample]
        csv_path = os.path.join(os.path.dirname(__file__), "..", "..", config["file"])
        df = pd.read_csv(csv_path)

        group_col         = config["group_col"]
        metric_col        = st.selectbox("Select metric column", config["metric_cols"], width='stretch')
        control_label     = config["control_label"]
        treatment_label   = config["treatment_label"]
        data_source_name  = selected_sample

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.caption(f"Control: `{control_label}`  |  Treatment: `{treatment_label}`")
        with col_info2:
            st.caption(f"{len(df):,} total records")
        st.caption(config["description"])
        st.dataframe(df.head(5), width="stretch")

    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df               = pd.read_csv(uploaded)
            data_source_name = uploaded.name
            st.dataframe(df.head(5), width="stretch")

            cols           = df.columns.tolist()
            default_group  = _guess_group_col(df)
            default_metric = _guess_metric_col(df, exclude=default_group)

            c1, c2 = st.columns(2)
            with c1:
                group_col  = st.selectbox("Group column (A/B label)", cols, index=cols.index(default_group), width='stretch')
                metric_cols_no_group = [c for c in cols if c != group_col]
                default_idx = (
                    metric_cols_no_group.index(default_metric)
                    if default_metric in metric_cols_no_group
                    else 0
                )
                metric_col = st.selectbox("Metric column", metric_cols_no_group, index=default_idx, width='stretch')
            with c2:
                if group_col:
                    groups          = df[group_col].unique().tolist()
                    control_label   = st.selectbox("Control group (A)", groups, width='stretch')
                    treatment_label = st.selectbox(
                        "Treatment group (B)",
                        [g for g in groups if g != control_label],
                        width='stretch',
                    )

    return dict(
        df=df,
        group_col=group_col,
        metric_col=metric_col,
        control_label=control_label,
        treatment_label=treatment_label,
        data_source_name=data_source_name,
    )
