from __future__ import annotations

import streamlit as st

from core.auth import is_authenticated


# =============================================================
# Watermelon System — Entry point router (variant 1)
# =============================================================
# app.py es uno de los dos entry points válidos para `streamlit run`.
# Funciona idéntico a 00_Home.py: si hay sesión activa redirige a
# pages/_Home.py, si no a la página de login.
#
# La razón de tener Home en pages/_Home.py (no aquí) es que
# st.switch_page() en Streamlit moderno solo acepta el main file y
# archivos en pages/. Centralizar Home en pages/_Home.py hace que el
# switch tras login funcione independientemente de cuál archivo se
# use como entry point.
# =============================================================

st.set_page_config(
    page_title="Watermelon System",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if is_authenticated():
    st.switch_page("pages/_landing.py")
else:
    st.switch_page("pages/00_Login.py")
