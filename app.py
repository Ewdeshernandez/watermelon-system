from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Watermelon System", layout="wide")

st.title("🍉 Watermelon System")

if st.button("Load Data"):
    st.switch_page("pages/01_Load_Data.py")

if st.button("Time Waveforms"):
    st.switch_page("pages/02_Time_Waveforms.py")

if st.button("Spectrum"):
    st.switch_page("pages/03_Spectrum.py")

if st.button("Trends"):
    st.switch_page("pages/04_Trends.py")

if st.button("Orbit Analysis"):
    st.switch_page("pages/05_Orbit_Analysis.py")

if st.button("Diagnostics"):
    st.switch_page("pages/15_Diagnostics.py")