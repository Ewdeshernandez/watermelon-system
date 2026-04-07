from __future__ import annotations

import streamlit as st

from core.auth import is_authenticated

st.set_page_config(
    page_title="Watermelon System",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Router inicial de la app
if is_authenticated():
    st.switch_page("00_Home.py")
else:
    st.switch_page("pages/00_Login.py")