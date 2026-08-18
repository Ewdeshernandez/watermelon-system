"""
pages/02_Remote_Monitoring.py — Página Remote Monitoring
========================================================

Módulo propio del sidebar, JUSTO debajo de Live Monitoring (sección
"Live Operations"). Independiente de Live Monitoring:

  · Live Monitoring   → escalares del Bently 3500 por Modbus (INTACTO).
  · Remote Monitoring → adquisición dinámica en vivo (maleta NI 9178 +
    9234 / simulado) → rotordinámica completa (waveform, spectrum, órbita,
    1X, tendencia).

Herramienta de analista (admin/specialist). El cliente no la ve — está en
CLIENT_BLOCKED_PAGES. La lógica vive en core/remote_monitoring/ui.py; esta
página solo hace el preámbulo de auth y delega el render.
"""
from __future__ import annotations

# set_page_config DEBE ir antes de render_user_menu() — si no, Streamlit
# resetea el CSS del sidebar (patrón canónico del repo, ver 01_Load_Data).
import streamlit as st

st.set_page_config(
    page_title="Watermelon System | Remote Monitoring",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.auth import require_login, render_user_menu, require_role
from core.ui_theme import apply_watermelon_page_style

require_login()
require_role(allowed_roles=("admin", "specialist"))
render_user_menu()
apply_watermelon_page_style()

from core.remote_monitoring.ui import render_remote_monitoring

render_remote_monitoring()
