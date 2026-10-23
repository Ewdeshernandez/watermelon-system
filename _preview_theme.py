"""Harness de preview del sistema de diseño (sin login). Renderiza los
componentes de core.ui_theme + core.ui_industrial con datos de ejemplo para
poder ver y pulir el look sin entrar a toda la app. NO es parte de producción."""
import streamlit as st

st.set_page_config(page_title="WM · Theme Preview", layout="wide")

# Mock de auth para que page_header muestre el hero (no el fallback simple).
try:
    import core.auth as _auth
    _auth.get_current_user = lambda: {"full_name": "Ewdes Hernández", "username": "ewdes"}
except Exception:
    pass

from core.ui_theme import apply_watermelon_page_style, page_header, card, kpi_card
from core.ui_industrial import inject_industrial_css, industrial_band, dot, html_table

apply_watermelon_page_style()
page_header("Live Monitoring", "3 activos · 2 en línea · 1 detenida · 0 alarmas")

st.markdown("### Tarjetas (card / kpi_card)")
col = st.columns(3)
with col[0]:
    card("SGT300A", "Siemens SGT-300 · Gas",
         "Turbina + gearbox + generador · Parex",
         chips=["API 670", "ISO 20816", "En línea"])
with col[1]:
    card("SGT300B", "Siemens SGT-300 · Gas",
         "Detenida (0 rpm) · última lectura 1/oct",
         chips=["Detenida", "N/A salud"])
with col[2]:
    kpi_card("Salud de flota", "Parex",
             [("92", "Salud media"), ("2", "En línea"), ("0", "Alarmas")])

st.markdown("### Botones")
b = st.columns(4)
b[0].button("Acción primaria", use_container_width=True)
b[1].button("Secundaria", use_container_width=True)
b[2].download_button("Descargar PDF", data="x", file_name="x.pdf", use_container_width=True)
b[3].button("Otra", use_container_width=True)

st.markdown("### Pestañas (tabs)")
t1, t2, t3, t4 = st.tabs(["Resumen", "Tabular", "Eventos", "Análisis"])
with t1:
    st.write("Contenido de resumen.")

st.markdown("### industrial_band + html_table + dot")
inject_industrial_css()
industrial_band("SIGA · Monitoreo en línea", "Live Monitoring",
                "Parex · SGT300A / SGT300B",
                kpis=[("● 2", "En línea"), ("● 1", "Detenida"), ("● 0", "Alarmas")])
html_table(
    ["Activo", "Estado", "Zona ISO", "Salud", "RPM"],
    [["SGT300A", dot("ok") + " En línea", "A", "92", "14 037"],
     ["SGT300B", dot("off") + " Detenida", "—", "—", "0"],
     ["Bomba BPC-2420", dot("warn") + " Alerta", "C", "61", "3 580"]],
    raw_cols=[1],
)
