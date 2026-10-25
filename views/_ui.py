"""Small shared UI helpers."""
from __future__ import annotations

import html

import streamlit as st

import config


def load_css() -> None:
    try:
        st.markdown(f"<style>{config.CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except OSError:
        pass  # purely cosmetic


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="ft-header"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="ft-section">{html.escape(title)}</div>', unsafe_allow_html=True)


def metric_card(label: str, value, hint: str = "") -> str:
    return (f'<div class="ft-card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(str(value))}</div>'
            f'<div class="hint">{html.escape(hint)}&nbsp;</div></div>')


def chip(text: str, kind: str = "ok") -> str:
    return f'<span class="ft-chip {kind}">{html.escape(text)}</span>'


def note(text: str) -> None:
    st.markdown(f'<div class="ft-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def go(page: str) -> None:
    """on_click callback: switch the sidebar page."""
    st.session_state["nav"] = page
