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
        /* Barra de pestañas PLANA estilo st.tabs (Calibración): sin cajas,
           un solo color de acento (subrayado cian en la activa). */
        div[data-testid="stRadio"] > div[role="radiogroup"] {{
            flex-direction:row !important; gap:30px; flex-wrap:wrap;
            border-bottom:1px solid #E5EAF0; margin-bottom:16px;
        }}
        div[data-testid="stRadio"] label {{
            background:transparent !important; border:none !important;
            border-radius:0 !important; box-shadow:none;
            padding:6px 1px 11px 1px !important; margin:0 !important;
            font-weight:500; font-size:14px; color:#64748B; cursor:pointer;
            transition:color .15s;
        }}
        div[data-testid="stRadio"] label:hover {{ color:{NAVY}; }}
        /* ocultar el círculo del radio (primer DIV del label) */
        div[data-testid="stRadio"] label > div:first-of-type {{ display:none !important; }}
        /* pestaña activa: texto navy + subrayado cian (indicador de tab) */
        div[data-testid="stRadio"] label:has(input:checked) {{
            color:{NAVY}; box-shadow:inset 0 -3px 0 {CYAN};
        }}
        /* punto de color por pestaña (algo de vida, sin cajas) */
        div[data-testid="stRadio"] label::before {{
            content:"●"; font-size:13px; margin-right:8px; vertical-align:middle;
            position:relative; top:-1px; opacity:0.95;
        }}
        div[data-testid="stRadio"] label:nth-of-type(1)::before {{ color:#64748B; }}
        div[data-testid="stRadio"] label:nth-of-type(2)::before {{ color:#1AAEE5; }}
        div[data-testid="stRadio"] label:nth-of-type(3)::before {{ color:#8B5CF6; }}
        div[data-testid="stRadio"] label:nth-of-type(4)::before {{ color:#16A34A; }}
        div[data-testid="stRadio"] label:nth-of-type(5)::before {{ color:#D89B22; }}
        div[data-testid="stRadio"] label:nth-of-type(6)::before {{ color:#EF4444; }}
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
