"""
pages/_admin_clients.py — Envoltorio fino (la lógica vive en core.admin.clients).

v3.31.491 — La sección Clientes/Specialists/Admins se movió a core/admin/clients.py
como render() reutilizable, y ahora se accede desde el hub
pages/20_Administracion.py (pestañas de colores). Este archivo se conserva como
envoltorio para que la URL directa siga funcionando. Solo admin.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Admin · Clientes & Roles — Watermelon",
    page_icon="🔐",
    layout="wide",
)

from core.auth import require_login, render_user_menu, require_role
from core.admin.clients import render

require_login()
render_user_menu()
require_role(allowed_roles=("admin",))

render()
