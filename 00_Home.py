from __future__ import annotations

import streamlit as st

# DESACTIVADO TEMPORALMENTE PARA DEBUG
# from core.auth import render_user_menu, require_login
from core.ui.theme import apply_watermelon_theme
from core.version import VERSION

st.set_page_config(
    page_title="Watermelon System | Home",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# require_login()
# render_user_menu()
apply_watermelon_theme()

MODULES = [
    {"title": "Load Data", "page": "pages/01_Load_Data.py"},
    {"title": "Time Waveforms", "page": "pages/02_Time_Waveforms.py"},
    {"title": "Spectrum", "page": "pages/03_Spectrum.py"},
    {"title": "Trends", "page": "pages/04_Trends.py"},
    {"title": "Orbit Analysis", "page": "pages/05_Orbit_Analysis.py"},
    {"title": "Diagnostics", "page": "pages/15_Diagnostics.py"},
]

st.title("🍉 Watermelon System")
st.caption(f"Version: {VERSION}")

cols = st.columns(3)
for i, m in enumerate(MODULES):
    with cols[i % 3]:
        if st.button(m["title"], use_container_width=True):
            st.switch_page(m["page"])