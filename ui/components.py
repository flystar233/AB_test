"""Reusable UI components shared across pages."""
import streamlit as st

# ── Design tokens (mirrors theme.py) ─────────────────────────────
_IVORY        = "#faf9f5"
_BORDER_CREAM = "#f0eee6"
_NEAR_BLACK   = "#141413"
_STONE_GRAY   = "#87867f"
_OLIVE_GRAY   = "#5e5d59"
_TERRACOTTA   = "#c96442"
_WARM_SAND    = "#e8e6dc"

_SERIF = "'Lora', Georgia, serif"
_SANS  = "Inter, system-ui, sans-serif"


def _card(label: str, value: str, sub: str = "", sub_color: str = _OLIVE_GRAY) -> str:
    sub_html = (
        f'<div style="font-size:0.72rem;color:{sub_color};margin-top:2px;'
        f'font-family:{_SANS};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
        f'{sub}</div>'
    ) if sub else ""

    return (
        f'<div style="'
        f'background:{_IVORY};'
        f'border-radius:12px;'
        f'padding:0.9rem 1rem;'
        f'border:1px solid {_BORDER_CREAM};'
        f'min-width:0;overflow:hidden;'
        f'box-shadow:rgba(0,0,0,0.04) 0px 2px 12px;'
        f'">'
        # Label (overline style)
        f'<div style="'
        f'font-size:0.68rem;'
        f'font-family:{_SANS};'
        f'color:{_STONE_GRAY};'
        f'font-weight:500;'
        f'letter-spacing:0.08em;'
        f'text-transform:uppercase;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
        f'">{label}</div>'
        # Value (serif for gravitas)
        f'<div style="'
        f'font-size:1.25rem;'
        f'font-family:{_SERIF};'
        f'font-weight:500;'
        f'color:{_NEAR_BLACK};'
        f'margin-top:4px;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
        f'">{value}</div>'
        + sub_html
        + "</div>"
    )


def kpi_row(cards: list[tuple]) -> None:
    """Render a horizontal row of metric cards.

    Each card is a tuple of (label, value) or (label, value, sub) or
    (label, value, sub, sub_color).
    """
    n = len(cards)
    html = (
        f'<div style="display:grid;grid-template-columns:repeat({n},1fr);'
        f'gap:10px;margin-bottom:1rem;">'
    )
    for item in cards:
        html += _card(*item)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """Render a styled page header with serif title."""
    sub_html = (
        f'<p style="font-family:{_SANS};font-size:1rem;color:{_STONE_GRAY};'
        f'line-height:1.60;margin:0.25rem 0 0;">{subtitle}</p>'
    ) if subtitle else ""

    st.markdown(
        f'<div style="border-bottom:1px solid {_BORDER_CREAM};margin-bottom:1.75rem;padding-bottom:1rem;">'
        f'<h1 style="font-family:{_SERIF};font-weight:500;font-size:2.25rem;'
        f'line-height:1.10;color:{_NEAR_BLACK};margin:0;">{title}</h1>'
        + sub_html
        + "</div>",
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    """Render a small uppercase overline section label."""
    st.markdown(
        f'<p style="font-family:{_SANS};font-size:0.65rem;font-weight:500;'
        f'letter-spacing:0.5px;text-transform:uppercase;color:{_STONE_GRAY};'
        f'margin:0 0 0.35rem;">{text}</p>',
        unsafe_allow_html=True,
    )
