"""
pages/20_License_Admin.py — Envoltorio fino (lógica en core.admin.licenses).

v3.31.491 — La gestión de licencias Watermelon Planta se movió a
core/admin/licenses.py como render() reutilizable, accesible desde el hub
pages/20_Administracion.py (pestañas de colores). Este archivo se conserva como
envoltorio para que la URL directa siga funcionando. Solo admin SIGA.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Watermelon · License Admin",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.auth import require_login, render_user_menu, require_role
from core.admin.licenses import render

require_login()
render_user_menu()
require_role(allowed_roles=("admin",))

render()
