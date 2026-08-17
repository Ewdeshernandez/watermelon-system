"""
core.reports_ext.ui — Selector tipo pestañas + kit visual de los reportes.

El selector es un radio horizontal estilizado como barra de pestañas (mismo
lenguaje enterprise que Calibración). Devuelve la clave de familia elegida; la
página de Reportes decide si sigue con el reporte del sistema (fall-through) o
renderiza uno de campo y hace st.stop().
"""
from __future__ import annotations

import streamlit as st

from core.reports_ext import REPORT_FAMILIES
from core.balance.ui import (  # noqa: F401  (re-export para la página)
    bal_section_header as rep_section_header,
    bal_status_banner as rep_status_banner,
    NAVY, CYAN, CYAN_DARK, GREEN, AMBER, GRAY, GRAY_LIGHT,
)

_LABELS = [lbl for _, lbl in REPORT_FAMILIES]
_KEYS = [key for key, _ in REPORT_FAMILIES]


def report_family_selector() -> str:
    """Barra de pestañas de familias de reporte. Devuelve la clave activa."""
    st.markdown(
        f"""
        <style>
        div[data-testid="stRadio"] div[role="radiogroup"] {{
            gap:6px; flex-wrap:wrap; border-bottom:2px solid {GRAY_LIGHT};
            padding-bottom:2px; margin-bottom:6px;
        }}
        div[data-testid="stRadio"] label {{
            background:{GRAY_LIGHT}; border:1px solid #e2e8f0; border-bottom:none;
            border-radius:9px 9px 0 0; padding:7px 16px !important; margin:0 !important;
            font-weight:600; color:{NAVY}; transition:all .15s;
        }}
        div[data-testid="stRadio"] label:hover {{ background:#e8f4fb; }}
        div[data-testid="stRadio"] label[data-baseweb="radio"] div:first-child {{
            display:none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    choice = st.radio("Tipo de reporte", _LABELS, horizontal=True,
                      key="rep_family_choice", label_visibility="collapsed")
    idx = _LABELS.index(choice) if choice in _LABELS else 0
    return _KEYS[idx]


__all__ = ["report_family_selector", "rep_section_header", "rep_status_banner",
           "NAVY", "CYAN", "CYAN_DARK", "GREEN", "AMBER", "GRAY", "GRAY_LIGHT"]
