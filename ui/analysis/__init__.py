import streamlit as st

from .sidebar import render_sidebar
from .data_loader import render_data_loader
from .runner import render_runner
from .results import render_results
from ui.components import page_header, section_label


def show_analysis_page() -> None:
    params = render_sidebar()

    captions = {
        "Quick Analysis":    "Fast, smart defaults — perfect for most analyses",
        "Monitor Over Time": "Watch results accumulate — stop early when evidence is clear",
        "Expert Mode":       "Full control for experts — tweak every parameter",
    }
    page_header("A/B Test Analysis", captions[params["mode"]])

    section_label("Step 1")
    st.markdown(
        '<h3 style="font-family:\'Lora\',Georgia,serif;font-weight:500;'
        'font-size:1.25rem;color:#141413;margin:0 0 0.75rem;">Load Data</h3>',
        unsafe_allow_html=True,
    )
    data_info = render_data_loader()
    st.divider()

    section_label("Step 2")
    st.markdown(
        '<h3 style="font-family:\'Lora\',Georgia,serif;font-weight:500;'
        'font-size:1.25rem;color:#141413;margin:0 0 0.75rem;">Run Analysis</h3>',
        unsafe_allow_html=True,
    )
    render_runner(data_info, params)

    render_results(params)
