"""Analysis history page."""
import pandas as pd
import streamlit as st

from auth_db import get_user_history
from ui.components import page_header


def show_history_page() -> None:
    if st.button("← Back to Analysis"):
        st.session_state["page"] = "Analysis"
        st.rerun()

    user = st.session_state["user"]
    page_header("Analysis History", f"Showing saved analyses for {user['username']}")

    records = get_user_history(user["id"], limit=100)
    if not records:
        st.markdown(
            '<div style="background:#faf9f5;border:1px solid #f0eee6;border-radius:12px;'
            'padding:2rem;text-align:center;color:#87867f;font-family:Inter,sans-serif;">'
            'No analysis records yet. Run an analysis and it will be saved automatically.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    rows = [
        {
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
        }
        for r in records
    ]
    df_hist = pd.DataFrame(rows)

    st.markdown(
        '<p style="font-family:Inter,sans-serif;font-size:0.82rem;color:#87867f;'
        'margin-bottom:0.5rem;">Click a row to view its details.</p>',
        unsafe_allow_html=True,
    )
    selection = st.dataframe(
        df_hist,
        width='stretch',
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
    st.markdown(
        f'<h3 style="font-family:\'Lora\',Georgia,serif;font-weight:500;font-size:1.3rem;'
        f'color:#141413;margin-bottom:0.25rem;">'
        f'Record #{r["id"]}</h3>'
        f'<p style="font-family:Inter,sans-serif;font-size:0.82rem;color:#87867f;margin:0;">'
        f'{r["created_at"][:19].replace("T", " ")}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:500;'
            'letter-spacing:0.08em;text-transform:uppercase;color:#87867f;margin-bottom:0.5rem;">'
            'Experiment Config</p>',
            unsafe_allow_html=True,
        )
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
            st.markdown(
                '<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:500;'
                'letter-spacing:0.08em;text-transform:uppercase;color:#87867f;margin-bottom:0.5rem;">'
                'Frequentist Results</p>',
                unsafe_allow_html=True,
            )
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
            st.markdown(
                '<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:500;'
                'letter-spacing:0.08em;text-transform:uppercase;color:#87867f;margin-bottom:0.5rem;">'
                'Bayesian Results</p>',
                unsafe_allow_html=True,
            )
            st.json({
                f"Posterior Mean {ctrl_lbl} (A)":  r["bayes_mean_a"],
                f"Posterior Mean {treat_lbl} (B)": r["bayes_mean_b"],
                "P(B > A)":                        r["bayes_prob_b_better"],
                "P(delta > MDE)":                  r["bayes_prob_practical"],
                f"Exp. Loss (Keep {ctrl_lbl})":    r["bayes_loss_a"],
                f"Exp. Loss (Launch {treat_lbl})": r["bayes_loss_b"],
                "Decision":                        r["bayes_decision"],
            })
