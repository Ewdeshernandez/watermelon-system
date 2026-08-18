"""
pages/_admin_users.py — Envoltorio fino (la lógica vive en core.admin.users).

v3.31.491 — La gestión de usuarios se movió a core/admin/users.py como render()
reutilizable, accesible desde el hub pages/20_Administracion.py (pestañas de
colores). Este archivo se conserva como envoltorio para que la URL directa siga
funcionando. Guard interno: solo el admin único (is_admin_email).
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Watermelon · Admin Usuarios",
    page_icon="👥",
    layout="wide",
)

from core.auth import require_login, render_user_menu
from core.admin.users import render

require_login()
render_user_menu()

render()
