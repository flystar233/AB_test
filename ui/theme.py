"""
Claude design-system theme injected as global CSS.

Call inject_theme() once per page render (idempotent in Streamlit).
"""
import streamlit as st

_CSS = """
/* ── Google Fonts: serif fallback ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=Inter:wght@400;500&display=swap');

/* ── Design tokens ───────────────────────────────────────────────── */
:root {
  --parchment:      #f5f4ed;
  --ivory:          #faf9f5;
  --pure-white:     #ffffff;
  --warm-sand:      #e8e6dc;
  --dark-surface:   #30302e;
  --near-black:     #141413;

  --terracotta:     #c96442;
  --coral:          #d97757;
  --focus-blue:     #3898ec;
  --error-crimson:  #b53333;

  --charcoal-warm:  #4d4c48;
  --olive-gray:     #5e5d59;
  --stone-gray:     #87867f;
  --dark-warm:      #3d3d3a;
  --warm-silver:    #b0aea5;

  --border-cream:   #f0eee6;
  --border-warm:    #e8e6dc;
  --border-dark:    #30302e;
  --ring-warm:      #d1cfc5;

  --serif:          'Lora', 'Georgia', serif;
  --sans:           'Inter', system-ui, -apple-system, sans-serif;
  --mono:           'SFMono-Regular', 'Consolas', 'Liberation Mono', monospace;

  --radius-sm:      6px;
  --radius-md:      8px;
  --radius-lg:      12px;
  --radius-xl:      16px;
  --radius-2xl:     24px;
  --radius-3xl:     32px;
}

/* ── Global reset & base ─────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background-color: var(--parchment) !important;
  font-family: var(--sans) !important;
  color: var(--near-black) !important;
}

/* Main content wrapper */
[data-testid="stMain"],
[data-testid="block-container"],
.main .block-container {
  background-color: var(--parchment) !important;
  padding-top: 2rem !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background-color: var(--warm-sand) !important;
  border-right: 1px solid var(--border-warm) !important;
}
[data-testid="stSidebar"] * {
  color: var(--olive-gray) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
  color: var(--near-black) !important;
  font-family: var(--serif) !important;
  font-weight: 500 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown strong {
  color: var(--olive-gray) !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label {
  color: var(--charcoal-warm) !important;
  font-family: var(--sans) !important;
  font-size: 0.875rem !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--border-cream) !important;
  opacity: 1 !important;
}
/* Sidebar radio options */
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
  color: var(--olive-gray) !important;
}
/* Sidebar username */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
  color: var(--near-black) !important;
  font-family: var(--sans) !important;
  font-size: 0.94rem !important;
}

/* ── Typography ──────────────────────────────────────────────────── */
h1 {
  font-family: var(--serif) !important;
  font-weight: 500 !important;
  line-height: 1.10 !important;
  color: var(--near-black) !important;
  letter-spacing: -0.01em !important;
  font-size: 2.5rem !important;
}
h2 {
  font-family: var(--serif) !important;
  font-weight: 500 !important;
  line-height: 1.20 !important;
  color: var(--near-black) !important;
  font-size: 2rem !important;
}
h3 {
  font-family: var(--serif) !important;
  font-weight: 500 !important;
  line-height: 1.30 !important;
  color: var(--near-black) !important;
  font-size: 1.5rem !important;
}
h4 {
  font-family: var(--serif) !important;
  font-weight: 500 !important;
  line-height: 1.30 !important;
  color: var(--near-black) !important;
  font-size: 1.2rem !important;
}

/* Streamlit title / header / subheader overrides */
[data-testid="stHeadingWithActionElements"] h1,
.stTitle {
  font-family: var(--serif) !important;
  font-weight: 500 !important;
  color: var(--near-black) !important;
}

/* Captions */
[data-testid="stCaptionContainer"],
.stCaption, .stCaption p {
  color: var(--stone-gray) !important;
  font-size: 0.875rem !important;
  line-height: 1.60 !important;
}

/* Body text */
p, li, .stMarkdown p {
  font-family: var(--sans) !important;
  color: var(--olive-gray) !important;
  line-height: 1.60 !important;
  font-size: 1rem !important;
}

/* ── Buttons ─────────────────────────────────────────────────────── */

/* Primary → Terracotta Brand */
.stButton button[kind="primary"],
button[data-testid="baseButton-primary"] {
  background-color: var(--terracotta) !important;
  color: var(--ivory) !important;
  border: none !important;
  border-radius: var(--radius-lg) !important;
  font-family: var(--sans) !important;
  font-weight: 500 !important;
  font-size: 0.94rem !important;
  padding: 0.55rem 1.2rem !important;
  box-shadow: var(--terracotta) 0px 0px 0px 0px, var(--terracotta) 0px 0px 0px 1px !important;
  transition: opacity 0.15s ease !important;
}
.stButton button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
  opacity: 0.88 !important;
  background-color: var(--terracotta) !important;
}

/* Secondary → Warm Sand */
.stButton button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
  background-color: var(--warm-sand) !important;
  color: var(--charcoal-warm) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-family: var(--sans) !important;
  font-weight: 500 !important;
  font-size: 0.94rem !important;
  padding: 0.5rem 1rem !important;
  box-shadow: var(--warm-sand) 0px 0px 0px 0px, var(--ring-warm) 0px 0px 0px 1px !important;
  transition: box-shadow 0.15s ease !important;
}
.stButton button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
  box-shadow: var(--warm-sand) 0px 0px 0px 0px, var(--ring-warm) 0px 0px 0px 2px !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton button {
  background-color: var(--ivory) !important;
  color: var(--charcoal-warm) !important;
  border: 1px solid var(--border-warm) !important;
  border-radius: var(--radius-md) !important;
}
[data-testid="stSidebar"] .stButton button:hover {
  background-color: var(--parchment) !important;
  color: var(--near-black) !important;
}

/* Form submit buttons (inherit primary styles) */
[data-testid="stFormSubmitButton"] button {
  background-color: var(--terracotta) !important;
  color: var(--ivory) !important;
  border: none !important;
  border-radius: var(--radius-lg) !important;
  font-family: var(--sans) !important;
  font-weight: 500 !important;
  font-size: 0.94rem !important;
  padding: 0.55rem 1.2rem !important;
  box-shadow: var(--terracotta) 0px 0px 0px 0px, var(--terracotta) 0px 0px 0px 1px !important;
  transition: opacity 0.15s ease !important;
}
[data-testid="stFormSubmitButton"] button:hover {
  opacity: 0.88 !important;
}

/* ── Inputs ──────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
  background-color: var(--ivory) !important;
  border: 1px solid var(--border-warm) !important;
  border-radius: var(--radius-lg) !important;
  color: var(--near-black) !important;
  font-family: var(--sans) !important;
  font-size: 0.94rem !important;
  padding: 0.45rem 0.75rem !important;
}
/* Selectbox - don't override padding to avoid text clipping */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background-color: var(--ivory) !important;
  border: 1px solid var(--border-warm) !important;
  border-radius: var(--radius-lg) !important;
  color: var(--near-black) !important;
  font-family: var(--sans) !important;
  font-size: 0.94rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--focus-blue) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(56, 152, 236, 0.15) !important;
}
/* Input labels */
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label,
[data-testid="stFileUploader"] label {
  color: var(--charcoal-warm) !important;
  font-family: var(--sans) !important;
  font-weight: 500 !important;
  font-size: 0.875rem !important;
}

/* ── Forms ───────────────────────────────────────────────────────── */
[data-testid="stForm"] {
  background-color: var(--ivory) !important;
  border: 1px solid var(--border-cream) !important;
  border-radius: var(--radius-xl) !important;
  padding: 1.5rem !important;
  box-shadow: rgba(0,0,0,0.05) 0px 4px 24px !important;
}

/* ── Cards / Expanders ───────────────────────────────────────────── */
[data-testid="stExpander"] {
  background-color: var(--ivory) !important;
  border: 1px solid var(--border-cream) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: rgba(0,0,0,0.04) 0px 2px 12px !important;
}
[data-testid="stExpander"] summary {
  color: var(--charcoal-warm) !important;
  font-family: var(--sans) !important;
  font-weight: 500 !important;
}

/* ── Tabs ────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  background-color: var(--warm-sand) !important;
  border-radius: var(--radius-xl) !important;
  padding: 4px !important;
  border: none !important;
  gap: 4px !important;
}
[data-testid="stTabs"] [role="tab"] {
  border-radius: var(--radius-lg) !important;
  color: var(--olive-gray) !important;
  font-family: var(--sans) !important;
  font-weight: 500 !important;
  font-size: 0.875rem !important;
  border: none !important;
  background: transparent !important;
  padding: 0.4rem 1rem !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background-color: var(--ivory) !important;
  color: var(--near-black) !important;
  box-shadow: rgba(0,0,0,0.08) 0px 1px 4px !important;
}

/* ── Metrics ─────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background-color: var(--ivory) !important;
  border: 1px solid var(--border-cream) !important;
  border-radius: var(--radius-lg) !important;
  padding: 1rem 1.25rem !important;
  box-shadow: rgba(0,0,0,0.04) 0px 2px 12px !important;
}
[data-testid="stMetric"] label {
  color: var(--stone-gray) !important;
  font-family: var(--sans) !important;
  font-size: 0.75rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
  color: var(--near-black) !important;
  font-family: var(--serif) !important;
  font-weight: 500 !important;
  font-size: 1.5rem !important;
}
[data-testid="stMetricDelta"] {
  font-family: var(--sans) !important;
  font-size: 0.8rem !important;
}

/* ── Dataframes / Tables ─────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border-cream) !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
}

/* ── Alerts / Info ───────────────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius-lg) !important;
  border: 1px solid var(--border-warm) !important;
  font-family: var(--sans) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="info"] {
  background-color: #f0ece3 !important;
  border-color: var(--border-warm) !important;
  color: var(--charcoal-warm) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="success"] {
  background-color: #e8f0e8 !important;
  color: #2d5a2d !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="error"] {
  background-color: #f8eaea !important;
  border-color: var(--error-crimson) !important;
  color: var(--error-crimson) !important;
}

/* ── Dividers ────────────────────────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid var(--border-cream) !important;
  margin: 1.5rem 0 !important;
}

/* ── Popovers ────────────────────────────────────────────────────── */
[data-testid="stPopover"] > div {
  background-color: var(--near-black) !important;
  border: 1px solid var(--border-dark) !important;
  border-radius: var(--radius-lg) !important;
}
[data-testid="stPopover"] button {
  background-color: var(--dark-surface) !important;
  color: var(--warm-silver) !important;
  border-radius: var(--radius-md) !important;
}
[data-testid="stPopover"] button:hover {
  background-color: var(--charcoal-warm) !important;
  color: var(--ivory) !important;
}

/* ── File Uploader ───────────────────────────────────────────────── */
[data-testid="stFileUploader"] section {
  background-color: var(--ivory) !important;
  border: 1.5px dashed var(--border-warm) !important;
  border-radius: var(--radius-lg) !important;
}

/* Ensure all sidebar buttons look clean */
[data-testid="stSidebar"] .stButton button {
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

/* Make sidebar use flexbox layout to push menu to bottom */
[data-testid="stSidebar"] > div:first-child {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

/* Push the menu to bottom */
.sidebar-bottom-spacer {
  flex-grow: 1;
}

/* ── Scrollbar (WebKit) ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--parchment); }
::-webkit-scrollbar-thumb { background: var(--ring-warm); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--olive-gray); }

/* ── Code blocks ─────────────────────────────────────────────────── */
code, pre {
  font-family: var(--mono) !important;
  background-color: var(--warm-sand) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--charcoal-warm) !important;
  font-size: 0.875rem !important;
}

/* ── JSON viewers ────────────────────────────────────────────────── */
[data-testid="stJson"] {
  background-color: var(--ivory) !important;
  border: 1px solid var(--border-cream) !important;
  border-radius: var(--radius-lg) !important;
}

/* ── Selectbox dropdown ──────────────────────────────────────────── */
[data-baseweb="popover"] ul {
  background-color: var(--ivory) !important;
  border: 1px solid var(--border-warm) !important;
  border-radius: var(--radius-lg) !important;
}
[data-baseweb="popover"] li {
  color: var(--near-black) !important;
  line-height: 1.5 !important;
  padding-top: 0.5rem !important;
  padding-bottom: 0.5rem !important;
}
[data-baseweb="popover"] li:hover {
  background-color: var(--warm-sand) !important;
}

/* ── Spinner ─────────────────────────────────────────────────────── */
[data-testid="stSpinner"] {
  color: var(--terracotta) !important;
}

/* ── Slider ──────────────────────────────────────────────────────── */
[data-baseweb="slider"] [data-testid="stThumbValue"],
[data-baseweb="slider"] div[role="slider"] {
  background-color: var(--terracotta) !important;
}
[data-baseweb="slider"] div[data-testid="stSliderTrackFill"] {
  background-color: var(--terracotta) !important;
}

/* ── Radio buttons ───────────────────────────────────────────────── */
[data-testid="stRadio"] label {
  color: var(--olive-gray) !important;
  font-family: var(--sans) !important;
  font-size: 0.9rem !important;
}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
  color: var(--charcoal-warm) !important;
}

/* ── Progress bar ────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
  background-color: var(--terracotta) !important;
}

/* ── Page header decoration ──────────────────────────────────────── */
.page-header {
  border-bottom: 1px solid var(--border-cream);
  margin-bottom: 2rem;
  padding-bottom: 1rem;
}
.page-title {
  font-family: var(--serif);
  font-weight: 500;
  font-size: 2.5rem;
  line-height: 1.10;
  color: var(--near-black);
  margin: 0 0 0.25rem 0;
}
.page-subtitle {
  font-family: var(--sans);
  font-size: 1rem;
  color: var(--stone-gray);
  line-height: 1.60;
  margin: 0;
}

/* ── Auth card ───────────────────────────────────────────────────── */
.auth-card {
  background: var(--ivory);
  border: 1px solid var(--border-cream);
  border-radius: var(--radius-3xl);
  padding: 2.5rem 2rem;
  box-shadow: rgba(0,0,0,0.06) 0px 8px 32px;
  max-width: 420px;
  margin: 4rem auto;
}
.auth-brand {
  font-family: var(--serif);
  font-weight: 500;
  font-size: 1.8rem;
  color: var(--near-black);
  text-align: center;
  margin-bottom: 0.25rem;
}
.auth-tagline {
  font-family: var(--sans);
  font-size: 0.9rem;
  color: var(--stone-gray);
  text-align: center;
  margin-bottom: 2rem;
}

/* ── Section label overline ──────────────────────────────────────── */
.section-overline {
  font-family: var(--sans);
  font-size: 0.65rem;
  font-weight: 500;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--stone-gray);
  margin-bottom: 0.4rem;
}
"""


def inject_theme() -> None:
    """Inject the Claude design-system CSS once per render."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
