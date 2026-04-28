from __future__ import annotations

import streamlit as st

from core.auth import is_authenticated


# =============================================================
# Watermelon System — Entry point router (variant 2)
# =============================================================
# 00_Home.py funciona idéntico a app.py: si hay sesión activa redirige
# a pages/_Home.py, si no a la página de login.
#
# Tener dos entry points equivalentes (app.py y 00_Home.py) permite
# que el deployment local o de producción use cualquiera de los dos
# sin afectar el flujo de navegación.
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
