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
        /* Barra de pestañas de familias de reporte (estilo Calibración) */
        div[data-testid="stRadio"] > div[role="radiogroup"] {{
            flex-direction:row !important; gap:4px; flex-wrap:wrap;
            border-bottom:2px solid #E5EAF0; margin-bottom:10px;
        }}
        div[data-testid="stRadio"] label {{
            background:{GRAY_LIGHT}; border:1px solid #E5EAF0; border-bottom:none;
            border-radius:10px 10px 0 0; padding:8px 18px !important; margin:0 !important;
            font-weight:600; color:{NAVY}; cursor:pointer; transition:all .15s;
        }}
        div[data-testid="stRadio"] label:hover {{ background:#E8F4FB; }}
        /* ocultar SOLO el círculo del radio (primer DIV del label; robusto
           aunque haya un <input> antes) */
        div[data-testid="stRadio"] label > div:first-of-type {{ display:none !important; }}
        /* pestaña activa: relleno navy */
        div[data-testid="stRadio"] label:has(input:checked) {{
            background:{NAVY}; color:#ffffff; border-color:{NAVY};
        }}
        div[data-testid="stRadio"] label:has(input:checked) * {{ color:#ffffff !important; }}
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
