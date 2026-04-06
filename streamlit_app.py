"""
A/B Test Analysis Platform — Streamlit + streamlit-echarts

Usage:
    streamlit run streamlit_app.py
"""
import streamlit as st

from auth_db import init_db
from ui.analysis import show_analysis_page
from ui.auth import show_auth_page, show_change_password_page
from ui.history import show_history_page
from ui.sample_size import show_sample_size_page
from ui.theme import inject_theme

# ── One-time initialisation ───────────────────────────────────────
init_db()

st.set_page_config(
    page_title="A/B Test Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Claude design-system theme on every render
inject_theme()

# ── Router ────────────────────────────────────────────────────────
user = st.session_state.get("user")

if not user:
    show_auth_page()
else:
    page = st.session_state.get("page", "Analysis")

    # ── Sidebar top: brand mark only ──────────────────────────────
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 1.25rem 0.5rem 0.75rem;
                        border-bottom: 1px solid #e8e6dc;
                        margin-bottom: 0.5rem;">
              <span style="font-family: 'Lora', Georgia, serif;
                           font-size: 1.1rem; font-weight: 500;
                           color: #141413; letter-spacing: -0.01em;">
                A/B Test Platform
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Page content (render_sidebar also writes to st.sidebar) ───
    if page == "Analysis":
        show_analysis_page()
    elif page == "Sample Size":
        show_sample_size_page()
    elif page == "History":
        show_history_page()
    elif page == "Change Password":
        show_change_password_page()

    # ── Sidebar bottom: three-dot menu at bottom ──
    with st.sidebar:
        # Spacer to push menu to bottom
        st.markdown('<div class="sidebar-bottom-spacer"></div>', unsafe_allow_html=True)

        if page in ("Analysis", "Sample Size"):
            with st.popover("⋮", use_container_width=False):
                st.markdown(
                    f'<p style="font-family:Inter,sans-serif;font-size:0.78rem;'
                    f'color:#87867f;margin:0 0 0.75rem 0;">@{user["username"]}</p>',
                    unsafe_allow_html=True,
                )
                if st.button("History", key="_m_history"):
                    st.session_state["page"] = "History"
                    st.rerun()
                if st.button("Change Password", key="_m_pwd"):
                    st.session_state["page"] = "Change Password"
                    st.rerun()
                st.divider()
                if st.button("Log Out", key="_m_logout"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
